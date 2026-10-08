"""Explore the effect of the balancing the per-speaker duration of the speakers on the amount of speakers"""

import sys
from pathlib import Path
import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets_creation.data_loading.data_loading import load_training_features
from datasets_creation.variant_factory.FeatureSchema import FeatureSchema
from datasets_creation.variant_factory.UtteranceBalancer import UtteranceBalancer

DATA_CONFIG = f"{REPO_ROOT}/datasets_creation/configs/data.yaml"
RESULTS_DIR = f"{REPO_ROOT}/results/data_analysis/generic"
FIGURES_DIR = f"{RESULTS_DIR}/plots"
STATS_DIR = f"{RESULTS_DIR}/stats"
COLOR_LINE ='#2563EB'
START = 30
FINISH = 1201
STEP = 30

def main():
    with open(DATA_CONFIG, encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    # prep the directories to save the results in
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    # get the paths to find the features and test file
    paths = {k: f"{REPO_ROOT}/{v}" for k, v in data_cfg['paths'].items() if k != 'features_files'}
    # load training data (includes removing uneccessary speakers)
    training_features = load_training_features(paths['features_dir'], 
                                            data_cfg['paths']['features_files'],
                                            paths['test_speakers_file'],
                                            data_cfg['remove_speakers'])
    # set up the data schema of the features file
    feature_schema_args = data_cfg['feature_schema']
    schema = FeatureSchema(**feature_schema_args)
    # loop through different amounts of speech per speaker and 
    # find the amount of speakers and amount of audio left
    # and save the results
    speaker_column = feature_schema_args['speaker_col']
    audio_duration_column = feature_schema_args['audio_duration_col']
    plot_data = {'dataset': [], 
                'seconds_per_speaker': [],
                'speakers_left': [],
                'audio_duration': []}
    for seconds in range(START, FINISH, STEP):
        balancer = UtteranceBalancer(schema=schema, seconds_per_speaker=seconds)
        sec_df = balancer.create_balanced_dataset(training_features)
        plot_data['dataset'].append(f"{seconds} seconds per speaker")
        plot_data['seconds_per_speaker'].append(seconds)
        plot_data['speakers_left'].append(sec_df[speaker_column].nunique())
        plot_data['audio_duration'].append(sec_df[audio_duration_column].sum())
    df = pd.DataFrame(plot_data)
    # save the results
    df.to_csv(f'{STATS_DIR}/speakers_left_after_balancing.csv', index=False)
    # plot the results and save the figure
    fig, ax = plt.subplots(figsize=(20, 7))
    x = df['seconds_per_speaker'].to_numpy()
    y = df['speakers_left'].to_numpy()

    ax.plot(
        x,
        y,
        marker='o',
        markersize=10,
        linewidth=3.5,
        color=COLOR_LINE
    )
    ax.set_xlabel('Seconds per speaker', fontsize=12, labelpad=10)
    ax.set_ylabel('Number of speakers', fontsize=12, labelpad=10)
    x_ticks = np.arange(START, FINISH, STEP) 
    y_ticks = np.arange(0, max(y)+20, 20) 
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    plt.title('Speakers with sufficient data vs. Amount of speech per speaker', fontsize=14, pad=20)

    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/speakers_left_after_balancing.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    main()

