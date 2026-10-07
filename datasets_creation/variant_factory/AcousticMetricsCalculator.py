from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.preprocessing import StandardScaler
from vendi_score import vendi


class AcousticMetricsCalculator:
    """
    Calculates per-speaker acoustic metrics and their summary statistics (std, mean, median,
    vendi score, diversity score), for single metrics or several combined.

    Attributes:
        phonemes_count_col: column name for the number of phonemes, for articulation_rate.
        phonemes_duration_col: column name for the total phoneme duration, for articulation_rate.
        audio_duration_col: column name for the audio duration, for audio_duration.
        F0_sum_col: column name for the summed F0 values, for mean_pitch.
        voiced_timeframes_num_col: column name for the number of voiced frames, for mean_pitch.
        file_col: the file column name.
        speaker_col: the speaker column name that features are aggregated by.
        formants: the formants that formant specs may use.
        phonemes: the vowels that formant specs may use.
        sum_col_pattern: pattern of the summed-formant column names, filled with formant and phoneme.
        count_col_pattern: pattern of the formant-count column names, filled with formant and phoneme.
        NORMALIZED_SUFFIX: suffix added to the names of standardized columns.
    """

    _ALL_METRICS = {
        'articulation_rate': '_articulation_rate',
        'audio_duration': '_audio_duration',
        'mean_pitch': '_mean_pitch',
    }
    _FORMANT_METRIC = 'mean_formant_frequency'
    _VOWEL_FORMANT_METRIC = 'mean_vowel_formant_frequency'
    NORMALIZED_SUFFIX = '__normalized'

    def __init__(self, schema, metrics=None):
        """
        Sets up a calculator for the given metrics, reading column names from `schema`.

        Args:
            schema: a FeatureSchema with the column names and patterns of the feature
                dataframe, plus its available formants and phonemes.
            metrics: the metric spec, or list of metric specs, this calculator computes. Defaults to the simple metrics.
        """
        self.phonemes_count_col = schema.phonemes_count_col
        self.phonemes_duration_col = schema.phonemes_duration_col
        self.audio_duration_col = schema.audio_duration_col
        self.F0_sum_col = schema.F0_sum_col
        self.voiced_timeframes_num_col = schema.voiced_timeframes_num_col
        self.file_col = schema.file_col
        self.speaker_col = schema.speaker_col
        self.formants = schema.formants
        self.phonemes = schema.phonemes
        self.sum_col_pattern = schema.sum_col_pattern
        self.count_col_pattern = schema.count_col_pattern

        if metrics is None:
            metrics = list(self._ALL_METRICS)
        elif isinstance(metrics, (str, tuple)):
            metrics = [metrics]
        self._metrics = {metric: self._build_metric(metric) for metric in metrics}

    def _build_metric(self, metric):
        """Returns the column name and the function neccessary to compute the given metric"""
        if isinstance(metric, tuple):
            if self.is_combined_spec(metric):
                raise ValueError(
                    f"{metric!r} is a combined group, not a computable metric — pass it to "
                    f"summary(..., combined=[{metric!r}]) instead of metrics=."
                )
            if len(metric) == 3:
                name, formant_arg, phoneme_arg = metric
                if name != self._VOWEL_FORMANT_METRIC:
                    raise ValueError(f"Unknown parameterized metric '{name}'.")
                if formant_arg not in self.formants:
                    raise ValueError(f"Unknown formant '{formant_arg}'. Available: {self.formants}")
                if phoneme_arg not in self.phonemes:
                    raise ValueError(f"Unknown vowel '{phoneme_arg}'. Available: {self.phonemes}")
                column = f'mean_{formant_arg}_{phoneme_arg}_frequency'
                return column, partial(
                    self._mean_vowel_formant_frequency, formant=formant_arg, phoneme=phoneme_arg
                )
            name, formant_arg = metric
            if name != self._FORMANT_METRIC:
                raise ValueError(f"Unknown parameterized metric '{name}'.")
            if formant_arg not in self.formants:
                raise ValueError(f"Unknown formant '{formant_arg}'. Available: {self.formants}")
            return f'mean_{formant_arg}_frequency', partial(
                self._mean_formant_frequency, formant=formant_arg
            )
        if metric not in self._ALL_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Available: {list(self._ALL_METRICS)}")
        return metric, getattr(self, self._ALL_METRICS[metric])

    def _combined_measure_columns(self, spec):
        """Returns the names of the metric columns that the `spec` wants to combine together."""
        if not self.is_combined_spec(spec):
            raise ValueError(
                f"Invalid combined spec {spec!r}; expected (name, [formant, ...]) or "
                f"(name, [formant, ...], [phoneme, ...])."
            )
        name = spec[0]
        if name != self._FORMANT_METRIC:
            raise ValueError(f"Unknown parameterized metric '{name}'.")
        if len(spec) == 3:
            _, formants_arg, phonemes_arg = spec
            unknown_formants = [f for f in formants_arg if f not in self.formants]
            if unknown_formants:
                raise ValueError(f"Unknown formant(s) {unknown_formants}. Available: {self.formants}")
            unknown_phonemes = [p for p in phonemes_arg if p not in self.phonemes]
            if unknown_phonemes:
                raise ValueError(f"Unknown vowel(s) {unknown_phonemes}. Available: {self.phonemes}")
            return [f'mean_{f}_{p}_frequency' for f in formants_arg for p in phonemes_arg]
        _, formants_arg = spec
        if len(formants_arg) < 2:
            raise ValueError(f"combined spec's formants must be a list of 2+ formants, got {formants_arg!r}.")
        unknown = [f for f in formants_arg if f not in self.formants]
        if unknown:
            raise ValueError(f"Unknown formant(s) {unknown}. Available: {self.formants}")
        return [f'mean_{f}_frequency' for f in formants_arg]

    @staticmethod
    def is_combined_spec(spec):
        """Whether `spec` is a combined spec (naming several metric columns to score together)"""
        if not (isinstance(spec, tuple) and len(spec) in (2, 3)):
            return False
        return all(isinstance(arg, (list, tuple)) for arg in spec[1:])

    def component_specs(self, spec):
        """Returns the metric specs `spec` is made of, one per column."""
        if self.is_combined_spec(spec):
            if len(spec) == 3:
                _, formants_arg, phonemes_arg = spec
                return [
                    (self._VOWEL_FORMANT_METRIC, f, p) for f in formants_arg for p in phonemes_arg
                ]
            name, formants_arg = spec
            return [(name, f) for f in formants_arg]
        return [spec]

    def metric_columns(self, spec):
        """Returns the column names `spec` refers to: one for a metric spec, several for a
        combined spec.
       
        Raises ValueError if any of those columns isn't among the metrics this calculator
        computes."""
        for component in self.component_specs(spec):
            self._validate_metric(component)
        if self.is_combined_spec(spec):
            return self._combined_measure_columns(spec)
        return [self._metric_column(spec)]

    def _metric_column(self, metric):
        return self._metrics[metric][0]

    def available_metrics(self):
        """The metric specs this calculator computes, as passed to the constructor."""
        return list(self._metrics)

    def _metric_label(self, spec):
        """Returns the row label for `summary()`'s output table."""
        if isinstance(spec, tuple):
            if self.is_combined_spec(spec):
                name = spec[0]
                if len(spec) == 3:
                    _, formants_arg, phonemes_arg = spec
                    return f"{name}[{','.join(formants_arg)}]x[{','.join(phonemes_arg)}]"
                _, formants_arg = spec
                return f"{name}[{','.join(formants_arg)}]"
            if len(spec) == 3:
                _, formant_arg, phoneme_arg = spec
                return f'mean_{formant_arg}_{phoneme_arg}_frequency'
            _, formant_arg = spec
            return f'mean_{formant_arg}_frequency'
        return spec

    def _articulation_rate(self, df):
        return df[self.phonemes_count_col] / (df[self.phonemes_duration_col] / 60)

    def _audio_duration(self, df):
        return df[self.audio_duration_col]

    def _mean_pitch(self, df):
        return df[self.F0_sum_col] / df[self.voiced_timeframes_num_col]

    def _mean_formant_frequency(self, df, formant):
        total = 0
        for phoneme in self.phonemes:
            sum = df[self.sum_col_pattern.format(formant=formant, phoneme=phoneme)]
            count = df[self.count_col_pattern.format(formant=formant, phoneme=phoneme)]
            mean = (sum / count).where(count != 0, 0)
            total = total + mean
        return total / len(self.phonemes)

    def _mean_vowel_formant_frequency(self, df, formant, phoneme):
        sum = df[self.sum_col_pattern.format(formant=formant, phoneme=phoneme)]
        count = df[self.count_col_pattern.format(formant=formant, phoneme=phoneme)]
        return (sum / count).where(count != 0, 0)

    def _validate_metric(self, metric):
        try:
            known = metric in self._metrics
        except TypeError:
            known = False  # unhashable (e.g. a combined group's list) — definitely not a computable metric
        if not known:
            if self.is_combined_spec(metric):
                raise ValueError(
                    f"{metric!r} is a combined group, not a computable metric — pass it to "
                    f"summary(..., combined=[{metric!r}]) instead of metrics=."
                )
            raise ValueError(f"Unknown metric '{metric}'. Available: {self.available_metrics()}")

    def _per_speaker(self, df):
        grouped = df.groupby(self.speaker_col)
        num_cols = df.select_dtypes(include=np.number).columns
        str_cols = (
            df.select_dtypes(exclude=np.number)
            .drop(columns=grouped.keys, errors='ignore')
            .columns
        )

        agg_dict = {**{col: 'sum' for col in num_cols}, **{col: 'count' for col in str_cols}}

        return grouped.agg(agg_dict).reset_index()

    def normalized_column(self, column):
        """The column name holding `column`'s standardized values"""
        return f'{column}{self.NORMALIZED_SUFFIX}'

    def compute_per_speaker(self, df, metrics=None, normalization=False):
        """Computes each metric's value. When `normalization` is set, each
        column also gets a sibling `normalized_column(column)` holding its standardized
        value."""
        per_speaker = self._per_speaker(df)
        metric_columns = []
        if metrics:
            if isinstance(metrics, (str, tuple)):
                metrics = [metrics]
            for metric in metrics:
                self._validate_metric(metric)
                metric_column, compute = self._metrics[metric]
                per_speaker[metric_column] = compute(per_speaker)
                metric_columns.append(metric_column)
        else:
            for metric_column, compute in self._metrics.values():
                per_speaker[metric_column] = compute(per_speaker)
                metric_columns.append(metric_column)

        if normalization:
            scaler = StandardScaler()
            scaler.fit(per_speaker[metric_columns])
            per_speaker = per_speaker.copy()
            normalized_columns = [self.normalized_column(column) for column in metric_columns]
            per_speaker[normalized_columns] = scaler.transform(per_speaker[metric_columns])
        return per_speaker

    def summary(self, df, metrics=None, groups=None, combined=[]):
        """Returns summary statistics, one row per metric and per combined spec."""
        if metrics is None:
            metrics = self._metrics.keys()
        elif isinstance(metrics, (str, tuple)):
            metrics = [metrics]

        summaries = {}
        for metric in metrics:
            self._validate_metric(metric)
            metric_column = self._metric_column(metric)
            summaries[self._metric_label(metric)] = self._summary_frame(df, [metric_column], groups)

        for spec in combined:
            metric_columns = self._combined_measure_columns(spec)
            missing = [c for c in metric_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{missing} not computed yet — request the individual formant(s) via metrics= first."
                )
            summaries[self._metric_label(spec)] = self._summary_frame(df, metric_columns, groups)

        return pd.concat(summaries, names=['metric']).reset_index().drop(columns=['level_1'])

    def _summary_frame(self, df, metric_columns, groups):
        if groups is None:
            return pd.DataFrame([self._stats_row(df, metric_columns)])
        return df.groupby(groups).apply(lambda g: self._stats_row(g, metric_columns))

    def _stats_row(self, df, metric_columns):
        """Returns a summary row for `metric_columns` (a list of >=1 column names).
        For a single column, std, mean, median, vendi_score and diversity_score are computed.
        For multiple columns, returns the standard distance, vendi_score and diversity_score."""
        single = metric_columns[0] if len(metric_columns) == 1 else None
        if single:
            std = df[single].std()
        else:
            centroid = df[metric_columns].mean(axis=0)
            distances = np.linalg.norm(df[metric_columns].values - centroid.values, axis=1)
            std = np.sqrt(np.mean(distances ** 2))
        return pd.Series({
            'std': std,
            'mean': df[single].mean() if single else np.nan,
            'median': df[single].median() if single else np.nan,
            'vendi_score': self.vendi_score(df, metric_columns),
            'diversity_score': self.diversity_score(df, metric_columns),
        })

    def vendi_score(self, df, metric_columns):
        """`metric_columns` is a single column name, or a list of column names to score
        together (speakers are compared by their mean absolute difference across all of them)."""
        if isinstance(metric_columns, str):
            metric_columns = [metric_columns]
        return self._vendi_score_values(df[list(metric_columns)].values)

    def diversity_score(self, df, metric_columns):
        """Vendi score normalized by the number of rows."""
        return self.vendi_score(df, metric_columns) / len(df)

    @staticmethod
    def _vendi_score_values(values):
        values = np.asarray(values)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return vendi.score_K(laplacian_kernel(values))
