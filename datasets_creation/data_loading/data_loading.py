import contextlib
import os
import wave

import pandas as pd


def read_data(root: str, filenames: list[str]) -> pd.DataFrame:
    """Reads and concatenates the feature-extraction CSVs, as-is."""
    return pd.concat(
        [pd.read_csv(os.path.join(root, filename)) for filename in filenames]
    )


def add_speaker_column(features_df: pd.DataFrame) -> pd.DataFrame:
    """Derives the speaker column from each row's filename."""
    features_df['Speaker'] = features_df['Filename'].str.split('-').str[0]
    return features_df


def read_test_speakers(test_set_path: str) -> pd.DataFrame:
    """Reads the held-out test-speaker list, as-is."""
    return pd.read_csv(test_set_path)


def filter_training_speakers(
    features_df: pd.DataFrame, test_set: pd.DataFrame, remove_speakers: list[str]
) -> pd.DataFrame:
    """Drops held-out test speakers and any explicitly excluded speakers from the training set."""
    training_features = features_df.loc[~features_df['Speaker'].isin(test_set['SpeakerID'])]
    training_features = training_features.loc[~training_features['Speaker'].isin(remove_speakers)]
    return training_features


def _load_utterance_manifest(root: str) -> pd.DataFrame:
    """Reads a Kaldi-style utterance directory: `wav.scp` maps utterance IDs to audio file
    locations, `text` maps utterance IDs to their transcription (`<id> <text>` per line)."""
    wav_location = pd.read_csv(f"{root}/wav.scp", sep=' ', names=['ID', 'location'])
    records = []
    with open(f"{root}/text", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            uttid, _, text = line.partition(" ")
            text = text.replace("ggg", "[LAUGH]").replace("xxx", "[FIL]")
            records.append((uttid, text))
    audio_text = pd.DataFrame(records, columns=["ID", "text"])
    return pd.merge(wav_location, audio_text, on='ID')


def get_audio_info(
    recordings_path: str, utterances_dirs: list[str], features_df: pd.DataFrame, drop_columns: list[str] = None
) -> pd.DataFrame:
    """Builds a per-utterance table (location, duration, speaker metadata) by combining the
    Kaldi-style utterance manifests in `utterances_dirs`, JASMIN's recordings metadata, and
    each utterance's duration from the already-loaded feature-extraction dataframe."""
    recordings_df = pd.read_excel(recordings_path)
    if drop_columns:
        recordings_df = recordings_df.drop(columns=drop_columns)

    audio_info = pd.concat(
        [_load_utterance_manifest(utterances_dir) for utterances_dir in utterances_dirs]
    )
    audio_info['root'] = audio_info['ID'].str.split("-").str[1]
    audio_info = pd.merge(audio_info, recordings_df, left_on='root', right_on='Root')
    audio_info = audio_info.drop(columns='Root')

    durations = features_df[['Filename', 'Audio Duration']].rename(
        columns={'Filename': 'ID', 'Audio Duration': 'duration'}
    )
    audio_info = pd.merge(audio_info, durations, on='ID', how='left')
    LOCATION_DIR = '/tudelft.net/staff-bulk/ewi/insy/SpeechLab/yuanyuanzhang/ESPnet202402/espnet/egs2/JASMIN/whisper_test'
    audio_info['location'] = audio_info['location'].\
        apply(lambda location: f'{LOCATION_DIR}/{location}')
    return audio_info


def select_test_audio(audio_info: pd.DataFrame, test_set: pd.DataFrame) -> pd.DataFrame:
    return audio_info.loc[audio_info['SpeakerID'].isin(test_set['SpeakerID'])]


def get_duration(file_path: str) -> float:
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

def load_training_features(features_dir: str, features_files: list[str], test_speakers_file: str, remove_speakers: list[str]) -> pd.DataFrame:
    features_df = add_speaker_column(read_data(features_dir, features_files))
    test_set = read_test_speakers(test_speakers_file)
    training_features = filter_training_speakers(features_df, test_set, remove_speakers)
    return training_features
