import pandas as pd


class AcousticVariabilitySweep:
    """Builds dataset variants across a set of settings and collects their variability statistics."""

    _BUCKET_VARIANTS = {'qcut': 'bucketed-qcut', 'cut': 'bucketed-cut'}
    _CLUSTER_VARIANTS = {'bucketed_kmeans': 'bucketed-kmeans', 'bucketed_kmeans_constrained': 'bucketed-kmeans-constrained'}

    def __init__(self, pipeline, metric,
                 seconds_per_speaker=range(120, 500, 120), speaker_counts=(50, 100),
                 bucket_counts=(5, 10, 25, 50), cluster_counts=(5, 10, 25, 50), 
                 num_repeats=100, normalization=[False]):
        self.pipeline = pipeline
        self.acoustics_calculator = pipeline.acoustics_calculator
        self.acoustics_calculator.metric_columns(metric)
        self.seconds_per_speaker = seconds_per_speaker
        self.speaker_counts = speaker_counts
        self.metric = metric
        self.bucket_counts = bucket_counts
        self.cluster_counts = cluster_counts
        self.num_repeats = num_repeats
        self.normalization = normalization

    def run(self, training_features):
        """Returns the resulting per-variant stats as a dataframe."""
        records = []

        for seconds_per_speaker in self.seconds_per_speaker:
            balancer = self.pipeline.create_utterance_balancer(seconds_per_speaker=seconds_per_speaker)
            balanced_features = balancer.create_balanced_dataset(training_features)
            
            for norm in self.normalization:
                
                for num in self.speaker_counts:
                    builder = self.pipeline.create_dataset_builder(
                        num_speakers=num, metric=self.metric, 
                        training_features=balanced_features,
                        normalization=norm
                    )
                    infeasible_buckets = set()
                    infeasible_clusters = set()

                    for i in range(self.num_repeats):
                        builder.build('low_variability', quantile=i / self.num_repeats)
                        records.append(self._summarize(
                            builder, balancer, seconds_per_speaker, norm, 'low', 0, builder.variant_df
                        ))

                        builder.build('extreme_spread', random_state=i)
                        records.append(self._summarize(
                            builder, balancer, seconds_per_speaker, norm, 'extreme', 0, builder.variant_df
                        ))

                        for bucket_method, label_prefix in self._BUCKET_VARIANTS.items():
                            for bucket in self.bucket_counts:
                                if bucket in infeasible_buckets:
                                    continue
                                try:
                                    builder.build(
                                        'bucketed', num_buckets=bucket, bucket_method=bucket_method,
                                        random_state=i,
                                    )
                                except ValueError as e:
                                    print(
                                        f"Skipping {label_prefix}-{bucket} at "
                                        f"seconds_per_speaker={seconds_per_speaker}, num_speakers={num}: {e}"
                                    )
                                    # Feasibility only depends on population size, not bucket_method,
                                    # so a failure here rules out this bucket count for both methods.
                                    infeasible_buckets.add(bucket)
                                    continue
                                records.append(self._summarize(
                                    builder, balancer, seconds_per_speaker, norm, 
                                    f"{label_prefix}", bucket, builder.variant_df,
                                ))

                        for variant, label_prefix in self._CLUSTER_VARIANTS.items():
                            for cluster in self.cluster_counts:
                                if cluster in infeasible_clusters:
                                    continue
                                try:
                                    builder.build(variant, num_clusters=cluster, random_state=i)
                                except ValueError as e:
                                    print(
                                        f"Skipping {label_prefix}-{cluster} at "
                                        f"seconds_per_speaker={seconds_per_speaker}, num_speakers={num}: {e}"
                                    )
                                    # Feasibility only depends on population size, not the clustering
                                    # method, so a failure here rules out this cluster count for both.
                                    infeasible_clusters.add(cluster)
                                    continue
                                records.append(self._summarize(
                                    builder, balancer, seconds_per_speaker, norm,
                                    label_prefix, cluster, builder.variant_df,
                                ))

        return pd.DataFrame(records)

    def _summarize(self, builder, balancer, seconds_per_speaker, norm, variant, buckets, subset_df):
        stats = builder.summary().iloc[0]
        cv = (stats['std'] / stats['mean']) * 100 if pd.notna(stats['mean']) else float('nan')
        return {
            'seconds_per_speaker': seconds_per_speaker,
            'norm': norm,
            'var': variant,
            'buckets': buckets,
            'num': len(subset_df),
            'cv': cv,
            'sd': stats['std'],
            'vd': stats['vendi_score'],
            'diversity pct': stats['diversity_score'],
            'duration': subset_df[balancer.audio_duration_col].sum() / 3600
        }
