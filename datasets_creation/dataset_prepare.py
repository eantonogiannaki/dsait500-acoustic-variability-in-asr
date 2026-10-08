"""
Builds the test/training/validation split either for the whole dataset or by building dataset variants.

    python dataset_prepare.py configs/data.yaml
    python dataset_prepare.py configs/data.yaml --variants-config configs/variant_metric.yaml

Each variant in the variants config is built once per parameter set in its `params`, with
num_speakers speakers and seconds_per_speaker seconds of audio each, selected on variant_metric.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets_creation.data_loading.data_loading import (
    read_data,
    add_speaker_column,
    filter_training_speakers,
    get_audio_info,
    read_data,
    read_test_speakers,
    select_test_audio,
)
from datasets_creation.plotting.plots import plot_variant
from datasets_creation.variant_factory.DatasetVariantFactory import DatasetVariantFactory
from datasets_creation.variant_factory.FeatureSchema import FeatureSchema

def to_spec(metric):
    return tuple(metric) if isinstance(metric, list) else metric


def split_train_validation(utterances, audio_info, validation_frac, random_state):
    """Splits the utterances in `utterances` into training and validation sets by speaker, with
    `validation_frac` of the speakers going to validation."""
    audio_info = audio_info.loc[audio_info['ID'].isin(utterances)]
    speakers = audio_info['SpeakerID'].unique()
    sampled_speakers = pd.Series(speakers).sample(frac=validation_frac, random_state=random_state)
    validation_set = audio_info[audio_info['SpeakerID'].isin(sampled_speakers)]
    training_set = audio_info[~(audio_info['SpeakerID'].isin(sampled_speakers))]

    return training_set, validation_set


def save_csv(df, directory, filename):
    """Writes `df` to `directory/filename` without the index."""
    df.to_csv(os.path.join(directory, filename), index=False)


def _split_stats(calculator, metrics, features_df, filenames):
    """Std/mean/median/diversity_score of each metric in `metrics`, for the rows of
    `features_df` whose Filename is in `filenames` — one row per metric. Lets us check a split
    is representative of the full variant, or the test set is representative of the speaker
    population, and lets us report audio_duration alongside the variant metric."""
    split_features = features_df[features_df['Filename'].isin(filenames)]
    per_speaker = calculator.compute_per_speaker(split_features, metrics=metrics)
    return calculator.summary(per_speaker, metrics=metrics)


def _print_split_stats(split_name, num_speakers, split_stats):
    for _, row in split_stats.iterrows():
        print(f"    {split_name} [{row['metric']}]: {num_speakers} speakers, "
              f"mean={row['mean']:.2f}, std={row['std']:.2f}, "
              f"diversity_score={row['diversity_score']:.2f}")


def main():
    """Writes the base test/training/validation split, or, given a variants config, builds
    each configured variant and saves its splits, plots and per-split stats."""
    parser = argparse.ArgumentParser(description='Build the test split and dataset variants.')
    parser.add_argument('data_config', type=Path, help='path to the data config')
    parser.add_argument(
        '--variants_config', type=Path, default=None,
        help='path to a variants config; without it, only the base split is written',
    )
    args = parser.parse_args()

    with open(args.data_config, encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    paths = {
        k: f"{REPO_ROOT}/{v}" for k, v in data_cfg['paths'].items() if k != 'features_files'
    }
    data_root = data_cfg['data']['root']

    features_df = add_speaker_column(
        read_data(paths['features_dir'], data_cfg['paths']['features_files'])
    )
    recordings_file = f"{data_root}/{data_cfg['data']['recordings_file']}"
    utterances_dirs = [f"{data_root}/{folder}" for folder in data_cfg['data']['utterances_dirs']]

    audio_info = get_audio_info(
        recordings_file, utterances_dirs, features_df,
        drop_columns=['Duration (seconds)', 'Duration (days)'],
    )
    test_set = read_test_speakers(paths['test_speakers_file'])

    test_audio_info = select_test_audio(audio_info, test_set)
    validation_frac = data_cfg['split']['validation_frac']
    validation_random_state = data_cfg['split']['random_state']

    variants_config = args.variants_config
    if not variants_config:
        variants_dir = f"{paths['variants_dir']}/all/datasets"
        train_audio_info = audio_info.loc[~(audio_info['SpeakerID'].isin(test_set['SpeakerID']))]
        training_set, validation_set = split_train_validation(
            list(train_audio_info['ID'].unique()), audio_info,
            validation_frac=validation_frac, random_state=validation_random_state,
        )
        os.makedirs(variants_dir, exist_ok=True)
        save_csv(test_audio_info, variants_dir, 'test.csv')
        save_csv(training_set, variants_dir, 'training.csv')
        save_csv(validation_set, variants_dir, 'validation.csv')
    else:
        with open(variants_config, encoding='utf-8') as f:
            variants_config = yaml.safe_load(f)

        feature_schema_args = data_cfg['feature_schema']
        metrics = [to_spec(m) for m in variants_config['metrics']]
        variant_metric = to_spec(variants_config['variant_metric'])
        results_folder_name = variants_config['results_folder_name']
        seconds_per_speaker = variants_config['seconds_per_speaker']
        num_speakers = variants_config['num_speakers']
        normalization = variants_config['normalization']

        root = (
            f"{paths['variants_dir']}/{results_folder_name}/"
            f"{num_speakers}-speakers_{seconds_per_speaker}-seconds"
        )
        plots_dir = f"{root}/plots"
        variants_dir = f"{root}/datasets"

        os.makedirs(variants_dir, exist_ok=True)
        shutil.copy(args.variants_config, f"{root}/config.yaml")

        save_csv(test_audio_info, variants_dir, 'test.csv')

        training_features = filter_training_speakers(
            features_df, test_set, data_cfg['remove_speakers'],
        )

        schema = FeatureSchema(**feature_schema_args)
        factory = DatasetVariantFactory(schema, metrics=metrics)

        test_stats = _split_stats(
            factory.acoustics_calculator, metrics, features_df, test_audio_info['ID'],
        )
        test_num_speakers = test_audio_info['SpeakerID'].nunique()
        print("Test set:")
        _print_split_stats('test', test_num_speakers, test_stats)
        test_stats.insert(0, 'num_speakers', test_num_speakers)
        test_stats.insert(0, 'split', 'test')
        test_stats.insert(0, 'params', '')
        test_stats.insert(0, 'variant', 'test')

        balancer = factory.create_utterance_balancer(seconds_per_speaker=seconds_per_speaker)
        balanced_features = balancer.create_balanced_dataset(training_features)
        print(f"Speakers with >= {seconds_per_speaker}s of audio: "
              f"{balanced_features[schema.speaker_col].nunique()}")

        builder = factory.create_dataset_builder(
            num_speakers=num_speakers,
            metric=variant_metric,
            training_features=balanced_features,
            normalization=normalization,
        )

        all_stats = [test_stats]
        for variant in variants_config['variants']:
            variant_name = variant['variant']
            for params in variant['params']:
                label = ', '.join(f'{k}={v}' for k, v in params.items())
                file_stub = variant_name + '-' + '_'.join(f'{k}-{v}' for k, v in params.items())

                builder.build(variant_name, **params)
                print(f"\n{variant_name} ({label}): {len(builder.variant_df)} speakers")
                if variant['plot']:
                    os.makedirs(plots_dir, exist_ok=True)
                    plot_variant(
                        builder, kind=variant['plot'], save_path=f"{plots_dir}/{file_stub}.png",
                    )

                utterances = builder.variant_utterances()
                training_set, validation_set = split_train_validation(
                    utterances, audio_info,
                    validation_frac=validation_frac, random_state=validation_random_state,
                )
                splits = (
                    ('variant', utterances, builder.variant_df[builder.speaker_col].nunique()),
                    ('training', training_set['ID'], training_set['SpeakerID'].nunique()),
                    ('validation', validation_set['ID'], validation_set['SpeakerID'].nunique()),
                )

                split_stats_list = []
                for split_name, filenames, split_num_speakers in splits:
                    split_stats = _split_stats(
                        builder.calculator, metrics, balanced_features, filenames,
                    )
                    _print_split_stats(split_name, split_num_speakers, split_stats)
                    split_stats.insert(0, 'num_speakers', split_num_speakers)
                    split_stats.insert(0, 'split', split_name)
                    split_stats_list.append(split_stats)

                variant_stats = pd.concat(split_stats_list, ignore_index=True)
                variant_stats.insert(0, 'params', label)
                variant_stats.insert(0, 'variant', variant_name)
                all_stats.append(variant_stats)

                save_csv(training_set, variants_dir, f'{file_stub}_training.csv')
                save_csv(validation_set, variants_dir, f'{file_stub}_validation.csv')

        stats_df = pd.concat(all_stats, ignore_index=True)
        save_csv(stats_df, root, 'variant_stats.csv')


if __name__ == '__main__':
    main()
