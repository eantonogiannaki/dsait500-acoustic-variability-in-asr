"""
Example of the dataset-building pipeline, run against synthetic data so it
works without the real feature-extraction CSVs. Run with: python example_usage.py
"""

import os

import numpy as np
import pandas as pd

from variant_factory.FeatureSchema import FeatureSchema
from variant_factory.DatasetVariantFactory import DatasetVariantFactory
from plots import plot_variant

METRIC = 'articulation_rate'
NUM_SPEAKERS = 50
NUM_BUCKETS = 10
SECONDS_PER_SPEAKER = 200
PLOTS_DIR = 'example-plots'


def make_synthetic_features(num_speakers=150, utterances_per_speaker=250, random_state=0):
    """Fakes an utterance-level features dataframe shaped like the real extraction output."""
    rng = np.random.default_rng(random_state)
    rows = []
    for speaker_idx in range(num_speakers):
        speaker_id = f"S{speaker_idx:03d}"
        for utt_idx in range(utterances_per_speaker):
            phoneme_duration = rng.uniform(0.5, 3.0)
            rows.append({
                'Speaker': speaker_id,
                'Filename': f"{speaker_id}-utt{utt_idx:03d}",
                'Audio Duration': phoneme_duration + rng.uniform(0.1, 0.5),
                'Number of Phonemes': rng.integers(3, 20),
                'Duration Phonemes': phoneme_duration,
            })
    return pd.DataFrame(rows)


def main():
    schema = FeatureSchema()  # default column names, matching make_synthetic_features above
    training_features = make_synthetic_features()

    # DatasetVariantFactory owns the schema and hands out helpers that all agree on column
    # names, so you never construct UtteranceBalancer / AcousticMetricsCalculator /
    # DatasetVariantBuilder directly
    factory = DatasetVariantFactory(schema, metrics=[METRIC])

    balancer = factory.create_utterance_balancer(seconds_per_speaker=SECONDS_PER_SPEAKER)
    balanced_features = balancer.create_balanced_dataset(training_features)
    print(f"Speakers with >= {SECONDS_PER_SPEAKER}s of audio: "
          f"{balanced_features['Speaker'].nunique()}")

    builder = factory.create_dataset_builder(
        num_speakers=NUM_SPEAKERS, metric=METRIC, training_features=balanced_features
    )
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\nLow-variability variants:")
    for quantile in (0.25, 0.5, 0.75):
        builder.build('low_variability', quantile=quantile)
        stats = builder.summary().loc[0]
        print(f"  quantile={quantile}: target speaker {builder.target_speaker}, "
              f"{len(builder.variant_df)} speakers, std={stats['std']:.2f}, "
              f"diversity_score={stats['diversity_score']:.2f}")
        plot_variant(
            builder, kind='swarm',
            save_path=os.path.join(PLOTS_DIR, f'low_variability_q{quantile}.png')
        )

    print("\nExtreme-spread variants:")
    for random_state in (0, 1, 2):
        builder.build('extreme_spread', random_state=random_state)
        stats = builder.summary().loc[0]
        print(f"  random_state={random_state}: target speaker {builder.target_speaker}, "
              f"{len(builder.variant_df)} speakers, std={stats['std']:.2f}, "
              f"diversity_score={stats['diversity_score']:.2f}")
        plot_variant(
            builder, kind='swarm',
            save_path=os.path.join(PLOTS_DIR, f'extreme_spread_rs{random_state}.png')
        )

    print(f"\nBucketed variants ({NUM_BUCKETS} buckets) with equal range:")
    for random_state in (0, 1, 2):
        builder.build('bucketed', num_buckets=NUM_BUCKETS, bucket_method='cut', random_state=random_state)
        stats = builder.summary().loc[0]
        print(f"  random_state={random_state}: {len(builder.variant_df)} speakers, "
              f"std={stats['std']:.2f}, diversity_score={stats['diversity_score']:.2f}")
        plot_variant(
            builder, kind='swarm',
            save_path=os.path.join(PLOTS_DIR, f'bucketed-equal-range-{NUM_BUCKETS}_rs{random_state}.png')
        )

    print(f"\nBucketed variants ({NUM_BUCKETS} clusters) with KMeans:")
    for random_state in (0, 1, 2):
        builder.build('bucketed_kmeans', num_clusters=NUM_BUCKETS, random_state=random_state)
        stats = builder.summary().loc[0]
        print(f"  random_state={random_state}: {len(builder.variant_df)} speakers, "
              f"std={stats['std']:.2f}, diversity_score={stats['diversity_score']:.2f}")
        plot_variant(
            builder, kind='swarm',
            save_path=os.path.join(PLOTS_DIR, f'bucketed-kmeans-{NUM_BUCKETS}_rs{random_state}.png')
        )


if __name__ == '__main__':
    main()
