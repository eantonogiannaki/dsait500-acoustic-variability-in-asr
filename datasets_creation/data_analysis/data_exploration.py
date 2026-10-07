import argparse
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets_creation.data_loading.data_loading import load_training_features 
from datasets_creation.plotting.plots import plot_variant
from datasets_creation.variant_factory.DatasetVariantFactory import DatasetVariantFactory 
from datasets_creation.variant_factory.FeatureSchema import FeatureSchema 
from variability_sweep import AcousticVariabilitySweep as Sweep

SCENARIO_MAPPING = {
    'low': 'Low',
    'extreme': 'Extreme',
    'bucketed-kmeans': 'Kmeans',
    'bucketed-kmeans-constrained': 'Kmeans - Equal sized clusters',
    'bucketed-qcut': 'Bucketing - Equal sized buckets',
    'bucketed-cut': 'Bucketing - Fixed range buckets',
}


def to_spec(metric):
    return tuple(metric) if isinstance(metric, list) else metric


def plot_sweep(data, hue, title, save_path=None, **kwargs):
    """Plots standard distance against diversity percentage, one panel per speaker count and
    seconds per speaker. Saves and closes the figure when `save_path` is given, shows it otherwise."""
    g = sns.relplot(
        data=data, x='sd', y='diversity pct',
        col='seconds_per_speaker', row='num',
        hue=hue, style='Scenarios', kind='scatter',
        alpha=0.85, palette='tab10', **kwargs,
    )
    for (num_speakers, seconds), ax in g.axes_dict.items():
        ax.text(
            0.05, 0.92, f'Total: {num_speakers * seconds / 3600:.1f} hrs',
            transform=ax.transAxes, fontsize=9.5, fontweight='bold', color='#374151',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2.5),
        )
    g.set_axis_labels('Standard Distance', 'Diversity Percentage')
    g.set_titles(col_template='{col_name} seconds per speaker', row_template='{row_name} speakers')
    g.figure.suptitle(title, fontsize=14, fontweight='bold')
    g.figure.subplots_adjust(top=0.92, hspace=0.3, wspace=0.2)
    if save_path:
        g.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(g.figure)
    else:
        plt.show()


def main():
    """Runs the variability sweep and builds the configured variants, saving stats and figures
    to the config's results directory."""
    parser = argparse.ArgumentParser(
        description='Run the variability exploration for one data_analysis config.',
    )
    parser.add_argument(
        'exploration_config', type=Path,
        help='path to an exploration config, e.g. configs/data_analysis/mean_pitch.yaml',
    )
    parser.add_argument('data_config', type=Path, help='path to the data config')
    args = parser.parse_args()

    with open(args.exploration_config, encoding='utf-8') as f:
        exploration_cfg = yaml.safe_load(f)
    with open(args.data_config, encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    paths = {k: f"{REPO_ROOT}/{v}" for k, v in data_cfg['paths'].items() if k != 'features_files'}
    results_root = f"{REPO_ROOT}/{exploration_cfg['results']['results_dir']}"
    figures_dir = f"{results_root}/figures"
    stats_dir = f"{results_root}/stats"

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    shutil.copy(args.exploration_config, f"{results_root}/config.yaml")

    metrics = [to_spec(m) for m in exploration_cfg['metrics']]
    variant_metric = to_spec(exploration_cfg['variant_metric'])
    feature_schema_args = data_cfg['feature_schema']

    training_features = load_training_features(
        paths['features_dir'],
        data_cfg['paths']['features_files'],
        paths['test_speakers_file'],
        data_cfg['remove_speakers'],
    )

    acoustic_calculator_args = {
        'metrics': metrics,
        'combined': [variant_metric] if isinstance(variant_metric, tuple) else [],
    }

    schema = FeatureSchema(**feature_schema_args)
    factory = DatasetVariantFactory(schema=schema, metrics=metrics)
    speakers = factory.acoustics_calculator.compute_per_speaker(training_features)
    population_summary = factory.acoustics_calculator.summary(speakers, **acoustic_calculator_args)
    population_summary.to_csv(f"{stats_dir}/summary.csv", index=False)

    start, stop, step = exploration_cfg['sweep']['seconds_per_speaker']
    seconds_per_speaker = range(start, stop, step)
    speaker_counts = exploration_cfg['sweep']['speaker_counts']
    bucket_counts = exploration_cfg['sweep']['bucket_counts']
    cluster_counts = exploration_cfg['sweep']['cluster_counts']
    normalization = exploration_cfg['sweep']['normalization']
    num_repeats = exploration_cfg['sweep']['num_repeats']
    vs = Sweep(
        factory,
        metric=variant_metric,
        seconds_per_speaker=seconds_per_speaker,
        speaker_counts=speaker_counts,
        bucket_counts=bucket_counts,
        cluster_counts=cluster_counts,
        normalization=normalization,
        num_repeats=num_repeats,
    )
    datasets_df = vs.run(training_features)
    datasets_df.to_csv(f"{stats_dir}/sweep_summary.csv", index=False)
    datasets_df['diversity pct'] = datasets_df['diversity pct'] * 100

    plot_data = (
        datasets_df
        .groupby(['var', 'buckets', 'num', 'seconds_per_speaker', 'norm'])
        .mean()
        .reset_index()
    )
    plot_data['Buckets'] = plot_data['buckets']
    plot_data['Scenarios'] = plot_data['var'].map(SCENARIO_MAPPING)

    for norm, data in plot_data.groupby('norm'):
        plot_sweep(
            data, hue='Buckets', s=120,
            title=f'Normalization: {norm}',
            save_path=f"{figures_dir}/{norm}-sweep.png",
        )

    build_seconds = exploration_cfg['build']['seconds_per_speaker']
    balancer = factory.create_utterance_balancer(seconds_per_speaker=build_seconds)
    balanced_features = balancer.create_balanced_dataset(training_features)
    balanced_speakers = factory.acoustics_calculator.compute_per_speaker(balanced_features)
    balanced_summary = factory.acoustics_calculator.summary(
        balanced_speakers, **acoustic_calculator_args,
    )
    balanced_summary.to_csv(
        f"{stats_dir}/{build_seconds}s-per-speaker_summary.csv", index=False,
    )

    builder = factory.create_dataset_builder(
        num_speakers=exploration_cfg['build']['num_speakers'],
        metric=variant_metric,
        training_features=balanced_features,
        normalization=exploration_cfg['build']['normalization'],
    )
    rows = []
    for variant in exploration_cfg['build']['variants']:
        variant_name = variant['variant']
        params = variant['params']
        builder.build(variant_name, **params)
        stats = builder.summary().iloc[0]
        rows.append({
            'variant': variant_name,
            'params': ', '.join(f'{k}={val}' for k, val in params.items()),
            'target': builder.target_speaker,
            **stats.drop('metric'),
        })
        kind = variant['plot']
        if kind:
            file = variant_name + '-' + '_'.join(f'{k}-{v}' for k, v in params.items())
            plot_variant(builder, kind=kind, save_path=f"{figures_dir}/{file}.png")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(f"{stats_dir}/variants_summary.csv", index=False)


if __name__ == '__main__':
    main()
