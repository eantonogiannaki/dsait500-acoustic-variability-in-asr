import itertools
import math


import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans

class DatasetVariantBuilder:
    """
    Builds dataset variants by selecting speakers from a base dataset according to 
    defined acoustic `metric`.
    A `metric` can be a signle metric or a group of metrics. Each speaker is reduced to
    their per-speaker value of these metric(s). 
    A variant is a set of `num_speakers` speakers, picked by one of these strategies:
    - 'low_variability': the speakers closest to a speaker, giving a tightly clustered variant.
    - 'extreme_spread': half the speakers closest to a randomly chosen speaker and half the
      furthest from them, giving a two-sided, high-contrast variant.
    - 'bucketed': speakers sampled evenly across a grid of per-metric bins, covering the
      metric's full range.
    - 'bucketed_kmeans': speakers sampled evenly across K-Means clusters.
    - 'bucketed_kmeans_constrained': speakers sampled evenly across size-balanced K-Means
      clusters.
    The goal of each strategy is to create a dataset of varying degrees of variability.
    """

    _VARIANT_SELECTORS = {
        'low_variability': 'select_low_variability_speakers',
        'extreme_spread': 'select_extreme_spread_speakers',
        'bucketed': 'select_bucketed_speakers',
        'bucketed_kmeans': 'select_kmeans_speakers',
        'bucketed_kmeans_constrained': 'select_kmeans_constrained_speakers',
    }

    def __init__(self, schema, num_speakers, metric, calculator, training_features, normalization=False):
        """Initialize the builder and compute each speaker's metric value(s)
        
        Args
            schema: names the speaker and filename columns of `training_features`.
            num_speakers: number of speakers each variant contains.
            metric: a single metric name or a combined metric group.
            calculator: computes per-speaker values of `metric` from `training_features`.
            training_features: the base dataset, one row per utterance, holding the speaker 
            and filename columns plus the feature columns `calculator` needs for `metric`.
            normalization: if True, the distance-based selection runs on standardized metric
            values so differently-scaled metrics contribute equally
        """
        self.schema = schema
        self.speaker_col = self.schema.speaker_col
        self.metric = metric
        self.num_speakers = num_speakers
        self.calculator = calculator
        self.training_features = training_features

        self.per_speaker_df = calculator.compute_per_speaker(
            training_features, metrics=calculator.component_specs(metric),
            normalization=normalization
        )
        # when normalization is True, the selecetion runs on the standardized columns (`self.cluster_cols`), so
        # differently-scaled metrics don't dominate distances. `self.metric_cols` stay on actual units and
        # are used to compute the variant's summary stats
        self.metric_cols = calculator.metric_columns(metric)
        self.cluster_cols = (
            [calculator.normalized_column(col) for col in self.metric_cols]
            if normalization else self.metric_cols
        )
        self.variant_df = None
        self.target_speaker = None
        self.num_buckets = None
        self.bucket_method = None
        self.cluster_labels = None
        self.cluster_centers_ = None

    def summary(self):
        """Summary stats for the current variant's metric(s)."""
        if self.calculator.is_combined_spec(self.metric):
            return self.calculator.summary(self.variant_df, metrics=[], combined=[self.metric])
        return self.calculator.summary(self.variant_df, metrics=self.metric)

    def variant_utterances(self):
        """Utterance filenames for the current variant's speakers."""
        if self.variant_df is None:
            raise ValueError("No speakers selected yet — call build() first.")
        return self.training_features[
            self.training_features[self.speaker_col].isin(self.variant_df[self.speaker_col])
        ][self.schema.file_col]

    def select_extreme_spread_speakers(self, random_state=None):
        """Selects speakers closest to and furthest from a random target using  Euclidean distance"""
        half = self.num_speakers // 2

        target_row = self.per_speaker_df.sample(1, random_state=random_state)
        target_point = target_row[self.cluster_cols].values[0]
        target_speaker = target_row[self.speaker_col].iloc[0]

        distances = np.linalg.norm(self.per_speaker_df[self.cluster_cols].values - target_point, axis=1)
        order = pd.Series(distances, index=self.per_speaker_df.index).sort_values().index
        closest = self.per_speaker_df.loc[order[:half]]
        furthest = self.per_speaker_df.loc[order[-half:]]
        extreme_spread_df = pd.concat([closest, furthest])

        return extreme_spread_df, target_speaker

    def select_low_variability_speakers(self, quantile=0.5):
        """Selects speakers closest (using Euclidiean distance) to a real speaker's
        profile: the speaker nearest the point formed by each column's own `quantile`-th
        percentile."""
        quantile_point = self.per_speaker_df[self.cluster_cols].quantile(quantile)
        quantile_distances = np.linalg.norm(
            self.per_speaker_df[self.cluster_cols].values - quantile_point.values, axis=1
        )
        target_idx = np.argmin(quantile_distances)
        target_point = self.per_speaker_df[self.cluster_cols].iloc[target_idx].values
        target_speaker = self.per_speaker_df[self.speaker_col].iloc[target_idx]

        distances = np.linalg.norm(self.per_speaker_df[self.cluster_cols].values - target_point, axis=1)
        closest_index = pd.Series(distances, index=self.per_speaker_df.index).sort_values().index[:self.num_speakers]
        low_variability_df = self.per_speaker_df.loc[closest_index]

        return low_variability_df, target_speaker

    def select_bucketed_speakers(self, num_buckets, bucket_method='qcut', random_state=None):
        """Selected speakers evenly across a `num_buckets`**`len(metric_cols)` grid. 
        Each metric is independently binned into `num_buckets` bins, forming the grid cells. 
        `bucket_method` is either 'qcut' (equal-population quantile bins, the default),
        or 'cut' (equal-width bins). 
        Only cells that actually contain speakers participate. 
        See `_select_samples` to see how the selection is performed after the cells are created."""

        self._check_population(f"{num_buckets}-bucket grid")

        bucketed = self.per_speaker_df.copy()
        bucket_cols = [f'_bucket_{i}' for i in range(len(self.metric_cols))]
        for bucket_col, col in zip(bucket_cols, self.metric_cols):
            bucketed[bucket_col] = self.bin(bucketed[col], num_buckets, bucket_method)

        groups = bucketed.groupby(bucket_cols)
        bucketed_df = self._select_samples(groups, bucket_cols, random_state)
        return bucketed_df, None

    def bin(self, series, num_buckets, bucket_method, retbins=False):
        """Bins `series` into num_buckets bins via 'qcut' (equal-population)
        or 'cut' (equal-width)"""
        if bucket_method == 'qcut':
            return pd.qcut(series, q=num_buckets, labels=False, retbins=retbins)
        if bucket_method == 'cut':
            return pd.cut(series, bins=num_buckets, labels=False, retbins=retbins)
        raise ValueError(f"Unknown bucket_method '{bucket_method}'. Available: ['qcut', 'cut']")

    def select_kmeans_speakers(self, num_clusters, random_state=None):
        """Clusters speakers using K-Means, then selects samples across clusters.
        See `_select_samples` to see how the selection is performed after the clusters are created.
        """
        self._check_population("K-Means clustering")
        num_clusters = min(num_clusters, len(self.per_speaker_df))

        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
        clustered = self.per_speaker_df.copy()
        clustered['_cluster'] = kmeans.fit_predict(clustered[self.cluster_cols].values)
        self.cluster_labels = clustered['_cluster']
        self.cluster_centers_ = kmeans.cluster_centers_

        groups = clustered.groupby('_cluster')
        kmeans_df = self._select_samples(groups, ['_cluster'], random_state)
        return kmeans_df, None

    def select_kmeans_constrained_speakers(self, num_clusters, size_min=None, size_max=None, random_state=None):
        """Clusters speakers by metric_cols using a size-constrained K-Means,  then selects samples
        across clusters.
        If size_min/size_max aren't given, they default to an even
        split of the population across num_clustered,
        so clusters come out as close to equal-sized as possible.
        See `_select_samples` to see how the selection is performed after the clusters are created.
        """

        self._check_population("constrained K-Means clustering")
        num_clusters = min(num_clusters, len(self.per_speaker_df))
        population = len(self.per_speaker_df)
        if size_min is None:
            size_min = population // num_clusters
        if size_max is None:
            size_max = math.ceil(population / num_clusters)

        kmeans = KMeansConstrained(
            n_clusters=num_clusters, size_min=size_min, size_max=size_max, random_state=random_state,
        )
        clustered = self.per_speaker_df.copy()
        clustered['_cluster'] = kmeans.fit_predict(clustered[self.cluster_cols].values)
        self.cluster_labels = clustered['_cluster']
        self.cluster_centers_ = kmeans.cluster_centers_

        groups = clustered.groupby('_cluster')
        kmeans_df = self._select_samples(groups, ['_cluster'], random_state)
        return kmeans_df, None

    def _check_population(self, strategy_label):
        population = len(self.per_speaker_df)
        if self.num_speakers > population:
            raise ValueError(
                f"Cannot build a {strategy_label} for num_speakers={self.num_speakers}: "
                f"{self.num_speakers} needed, but only "
                f"{population} speakers are available. Try smaller num_speakers."
            )

    def _select_samples(self, groups, group_cols, random_state=None):
        """Selects samples up to self.num_speakers across `groups` via three
        phases:
        1. Every group claims up to self.num_speakers divided by the number of groups (rounded up).
        2. Groups left short then fill the gap with the nearest speakers (using Euclidean
           distance) that were not selected yet.
        3. Since the amount of speaker per group is rounded up, the combined total can exceed 
        self.num_speakers. If so, groups are trimmed one random member at a time until the total 
        matches num_speakers exactly.
        The total number of borrowed and trimmed speakers is printed. """

        all_rows = pd.concat([group for _, group in groups])
        per_cell = math.ceil(self.num_speakers / len(groups))

        # Phase 1: each group claims up to per_cell from its members
        cell_frames = {}
        short_cells = []
        selected_ids = set()
        for key, group in groups:
            if len(group) >= per_cell:
                sampled = group.sample(per_cell, random_state=random_state)
            else:
                sampled = group
                short_cells.append((key, group))
            cell_frames[key] = sampled
            selected_ids.update(sampled[self.speaker_col])

        # Phase 2: fill each shortfall from speakers nobody claimed in phase 1.
        total_borrowed = 0
        total_unfilled = 0
        for key, group in short_cells:
            needed = per_cell - len(group)
            candidates = all_rows.loc[~all_rows[self.speaker_col].isin(selected_ids)]
            if len(candidates) == 0:
                total_unfilled += needed
                continue
            member_points = group[self.cluster_cols].values
            candidate_points = candidates[self.cluster_cols].values
            distances = np.min(
                np.linalg.norm(candidate_points[:, None, :] - member_points[None, :, :], axis=2),
                axis=1,
            )
            order = np.argsort(distances)[:needed]
            borrowed = candidates.iloc[order]
            total_borrowed += len(borrowed)
            total_unfilled += needed - len(borrowed)
            cell_frames[key] = pd.concat([cell_frames[key], borrowed])
            selected_ids.update(borrowed[self.speaker_col])

        # Phase 3: per_cell is rounded up, so the combined total can exceed num_speakers when
        # it doesn't divide evenly across groups. 
        # Trim the excess one random member at a time across groups.
        excess = sum(len(df) for df in cell_frames.values()) - self.num_speakers
        total_trimmed = max(excess, 0)
        if excess > 0:
            rng = np.random.default_rng(random_state)
            for key in itertools.cycle(cell_frames):
                if excess == 0:
                    break
                df = cell_frames[key]
                if len(df) == 0:
                    continue
                drop_pos = rng.integers(len(df))
                cell_frames[key] = df.drop(df.index[drop_pos])
                excess -= 1

        print(f"Borrowed {total_borrowed} speaker(s), trimmed {total_trimmed} speaker(s).")
        if total_unfilled:
            print(f"{total_unfilled} slot(s) left unfilled — no unclaimed speakers remained.")

        return pd.concat(cell_frames.values()).reset_index(drop=True).drop(columns=group_cols)

    def build(self, variant, **variant_kwargs):
        """Selects speakers for the given variant type, and remembers the result."""
        if variant not in self._VARIANT_SELECTORS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(self._VARIANT_SELECTORS)}"
            )

        self.num_buckets = variant_kwargs.get('num_buckets')
        self.bucket_method = variant_kwargs.get('bucket_method', 'qcut')
        self.cluster_labels = None
        self.cluster_centers_ = None
        selector = getattr(self, self._VARIANT_SELECTORS[variant])
        self.variant_df, self.target_speaker = selector(**variant_kwargs)
        # cluster_cols may be normalized columns alongside the actual-value ones —
        # keep variant_df to just the actual-value columns callers expect.
        normalized_cols = [
            col for col in self.variant_df.columns if col.endswith(self.calculator.NORMALIZED_SUFFIX)
        ]
        if normalized_cols:
            self.variant_df = self.variant_df.drop(columns=normalized_cols)