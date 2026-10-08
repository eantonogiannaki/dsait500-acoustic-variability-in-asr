# Evaluating the effect of inter-speaker acoustic variability on ASR performance and bias

## Aim
This project is part of my MSc thesis in Data Science and AI Technology at TU Delft where I aim to evaluate the effect of inter-speaker acoustic variability on both ASR performance and bias.

It contains the code responsible for generating datasets of varying degrees of inter-speaker variability (in terms of standand distance and diversity) given a specified acoustic metric (e.g, articulation  rate) or a combination of those (format frequencies of all vowels) called dataset variants from now on. It also stores the configurations and training scripts of the experiments run (using SpeechBrain).

## Structure

```
├── datasets_creation/        # Builds the dataset variants
│   ├── dataset_prepare.py    # Prepares and stores the dataset variants (specific to the dataset used in this project)
│   ├── example_usage.py      # Demo of building the dataset variants on synthetic data (no corpus needed)
│   ├── configs/              # Data paths and settings to prepare the dataset variants and perform data analysis tasks
│   ├── variant_factory/      # Core pipeline: balancing, calculating the metrics, building the variants
│   ├── data_loading/         # Loading the feature CSVs (specific to the dataset used in this project)
│   ├── data_analysis/        # Data analysis and exploration scripts
│   └── plotting/             # Scripts to plot the dataset variants
│   ├── ... 
└── training/                 # SpeechBrain recipes for the ASR experiments
```

## Requirements

Each of the `datasets_creation\` and `training\` can be run independently therefore each folder contains its own `requirements.txt` file.

### Dataset variants generation

To run the scripts generating the dataset variants or utilize the core pipeline for your own project (check `example_usage.py` for reference), you need to have Anaconda installed on your device.

After installing Anaconda, create and activate a new environment with Python 3.12:

```bash
conda create -n acoustic-variability python=3.12
conda activate acoustic-variability
```

Then install the dependencies:

```bash
pip install -r datasets_creation/requirements.txt
```

To check that everything works, run the demo from inside `datasets_creation/`. It builds dataset variants from synthetic data and saves their plots to `example-plots/` folder:

```bash
cd datasets_creation
python example_usage.py
```

For details on running the preparation and data analysis scripts, see the [`datasets_creation` README](datasets_creation/README.md).

### Training experiments

#TODO