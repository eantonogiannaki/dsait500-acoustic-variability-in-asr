import numpy as np
import pandas as pd
from vendi_score import vendi


class AcousticProperties:
    """Computes acoustic proporties per speaker and their summary statistics."""

    def __init__(self, measures=None, phonemes_col='Number of Phonemes', phoneme_duration_col='Duration Phonemes',
                 audio_duration_col='Audio Duration', speaker_col='Speaker'):
        self.phonemes_column = phonemes_col
        self.phoneme_duration_column = phoneme_duration_col
        self.audio_duration_column = audio_duration_col
        self.speaker_column = speaker_col
        self._measures = {
            'articulation_rate': self._articulation_rate,
            'audio_duration': self._audio_duration,
        }
        if measures:
            self._measures.update(measures)

    def _articulation_rate(self, df):
        return df[self.phonemes_column] / (df[self.phoneme_duration_column] / 60)

    def _audio_duration(self, df):
        return df[self.audio_duration_column]
    
    def available_measures(self):
        return list(self._measures)

    def _validate_measure(self, measure):
        if measure not in self._measures:
            raise ValueError(f"Unknown measure '{measure}'. Available: {self.available_measures()}")

    def _per_speaker(self, df):
        grouped = df.groupby(self.speaker_column)
        return grouped.sum(numeric_only=True).reset_index()

    def compute_per_speaker(self, df, measures=None):
        per_speaker = self._per_speaker(df)
        if measures:
            if isinstance(measures, str):
                measures = [measures]
            for measure in measures:
                self._validate_measure(measure)
                per_speaker[measure] = self._measures[measure](per_speaker)
        else:
            for measure, function in self._measures.items():
                per_speaker[measure] = function(per_speaker)

        return per_speaker

    def summary(self, df, measures=None, groups=None):
       
        if measures:
            if isinstance(measures, str):
                measures = [measures]
        else:
            measures = self._measures.keys()  
        summaries = {}
        for measure in measures:
            if groups is None:
                values = df[measure]
                stats = pd.DataFrame({
                    'std': [values.std()],
                    'mean': [values.mean()],
                    'median': [values.median()],
                    'vendi_score': [self.vendi_score(df, measure)],
                    'diversity_score': [self.diversity_score(df, measure)],
                })
            else:
                grouped = df.groupby(groups)[measure]
                stats = grouped.agg(['std', 'mean', 'median', 'count'])
                stats['vendi_score'] = grouped.apply(lambda s: self._vendi_score_values(s.values))
                stats['diversity_score'] = stats['vendi_score'] / stats['count']
                stats = stats.drop(columns='count')
            summaries[measure] = stats
            summary = pd.concat(summaries, names=['measure']).reset_index().drop(columns=['level_1'])
        return summary

    def vendi_score(self, df, measure):
        return self._vendi_score_values(df[measure].values)

    def diversity_score(self, df, measure):
        """Vendi score normalized by the number of rows."""
        return self.vendi_score(df, measure) / len(df)

    @staticmethod
    def _vendi_score_values(values):
        kernel = lambda a, b: np.exp(-np.abs(a - b))
        return vendi.score(values, kernel)
