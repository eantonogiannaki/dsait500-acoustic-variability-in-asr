import pandas as pd


class UtteranceBalancer:
    """
    Trims each speaker's utterances down to a fixed `seconds_per_speaker` of audio,
    keeping their longest utterances first, and drops speakers who can't reach that
    target.
    """

    def __init__(self, schema, seconds_per_speaker):
        self.seconds_per_speaker = seconds_per_speaker
        self.schema = schema
        self.speaker_col = self.schema.speaker_col
        self.audio_duration_col = self.schema.audio_duration_col
        self.file_col = self.schema.file_col

    def create_balanced_dataset(self, df):
        """
        Trims each speaker down to `seconds_per_speaker` seconds of audio, keeping their
        longest utterances first, and drops speakers who can't reach that target.
        """
        balanced_rows = []
        seen_files = set()
        for _, group in df.groupby(self.speaker_col):
            group = group.sort_values(self.audio_duration_col, ascending=False)
            running_total = 0.0
            for _, row in group.iterrows():
                file_id = row[self.file_col]
                if file_id in seen_files:
                    continue
                if running_total >= self.seconds_per_speaker:
                    break
                balanced_rows.append(row)
                seen_files.add(file_id)
                running_total += row[self.audio_duration_col]

        balanced_features = pd.DataFrame(balanced_rows)

        totals = balanced_features.groupby(self.speaker_col)[self.audio_duration_col].sum()
        valid_speakers = totals[totals >= self.seconds_per_speaker].index
        return balanced_features.loc[balanced_features[self.speaker_col].isin(valid_speakers)]
