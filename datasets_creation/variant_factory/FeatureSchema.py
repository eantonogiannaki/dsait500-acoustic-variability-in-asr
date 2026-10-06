from dataclasses import dataclass, field


@dataclass()
class FeatureSchema:
    """
    Column names for the training-features dataframe, used by the dataset-variant-building
    pipeline: DatasetVariantFactory and the AcousticMetricsCalculator, UtteranceBalancer, and
    DatasetVariantBuilder it creates. A single instance is shared across all of them so they
    agree on column names, and can be reconfigured for a differently-named dataset without
    changing any of their code.

    """
    speaker_col: str = 'Speaker'
    audio_duration_col: str = 'Audio Duration'
    phonemes_count_col: str = 'Number of Phonemes'
    phonemes_duration_col: str = 'Duration Phonemes'
    F0_sum_col: str = 'Sum of F0 (Hz)'
    voiced_timeframes_num_col: str = 'Number of Voiced Frames'
    formants: list = field(default_factory=lambda: ['F1', 'F2'])
    phonemes: list = field(default_factory=lambda: ['I', 'E', 'A', 'O', 'Y', '@', 'i', 'y', 'e', '2', 'a', 'o', 'u'])
    sum_col_pattern: str = "Sum of {formant} {phoneme} (bark)"
    count_col_pattern: str = "Number of {formant} {phoneme}"
    phoneme_formants_sum_col: list = field(init=False)
    phoneme_count_col: list = field(init=False)
    # mean_pitch_col: str = 'Mean Pitch (Hz)'
    file_col: str = 'Filename'

    def __post_init__(self):
        self.phoneme_formants_sum_col = [
            self.sum_col_pattern.format(formant=formant, phoneme=phoneme)
            for formant in self.formants for phoneme in self.phonemes
        ]
        self.phoneme_count_col = [
            self.count_col_pattern.format(formant=formant, phoneme=phoneme)
            for formant in self.formants for phoneme in self.phonemes
        ]
