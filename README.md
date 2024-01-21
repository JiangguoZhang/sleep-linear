
<div align="center">
<h1>Do not 😴 on traditional ML</h1>
<h2>Simple and Interpretable Techniques Are Competitive to Deep Learning for Sleep Scoring</h2>
</div>

## Inspiration


## What it does
### Data Wrangling
* Clean the dataset
  * Remove NaN data points.
  * Remove not scored sleep stage.
* Manipulate
  * Using bandpass filters to extract meaning data.
  * Normalize data to around 0.
  * Balance the dataset by removing some Waking stage data, which are far away from the sleep stages (> 30 minutes).
  * Calculate time domain statistics, including std, iqr, skewness, kurtosis, number of zero-crossings, Hjorth mobility, Hjorth complexity, higuch fractal dimension, petrosial fractal dimension, permutation entropy, binned entropy (4); and frequency-domain statistics including spectral Fourier statistics (4), binned Fourier entropy (7), absolute spectral power in the 0.4-30 Hz band, relative spectral power in the applied frequency bands (6), fast delta+theta spectral power, alpha/theta spectral power, delta/beta spectral power, delta/sigma spectral power and delta/theta spectral power.

### Data Visualization
* Visualize the data structure.
* Visualize different channels including EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, Resp oro-nasal, EMG submental, Temp rectal and the sleep stage labels. Visualize the processed data.
* Visualize the count of data points in different sleep stages.
* Visualize the model structure.
* Visualize the results in a confusion matrix and metric tables.

### Process
1. Collect data information
2. Load data
3. Bandpass filter and normalize
4. Calculate statistics
5. Save data
6. Train the model
7. Evaluate the model
8. Select the best model and make predictions

### Machine Learning Models
* Implemented Linear Classifier and Multi-Layer Perceptron Classifier with sklearn.
* Implemented Catboost Classifier from catboost library.
* The models are trained to convert statistical features calculated from 6 feature channels into 6 distinct sleep stages.

### Evaluation

## How we built it
### Data process
* We built helper functions to load data.
* We used packages and code from [paper](https://arxiv.org/abs/2207.07753) to calculate statistics.
### Visualization
* The process can be visualized in jupyter notebook.
### Model building
* We used machine learning packages including sklearn and scipy.

## Challenges we ran into
* Model evaluation
  * We calculated F1 score, balanced accuracy, accuracy, and log loss for each model on training set and test set. Due to the time limit, we used the result from 2-fold cross validation.

| Model  |Dataset | F1     | Balanced accuracy | Accuracy | Log loss |
|--------|--------|--------|-------------------|----------|----------|
|Linear  |train   | 0.8378 | 0.8529            | 0.8826   | 0.7567   |
|Linear  |train   | 0.7242 | 0.7334            | 0.8005   |1.6495|
|Catboost|train   | 0.9297 | 0.9193            | 0.9501   | 0.1736|
|Catboost|test    | 0.7490 | 0.7440            | 0.8229   |0.4717|
| MLP    |train   | 0.9061 | 0.8920            | 0.9397   |0.1700|
| MLP    |test  | 0.7139 | 0.7125            | 0.7979   |0.6820|

* Overfitting
  * According to the result, the MLP model is overfitted to the training set, with a better score than the other 2 models on training set, while a worse score than the Catboost model on test set.
  * The MLP model is overfitted because it is a complex model with too many parameters, so it can fit to the variation of data statistics, while leading to large bias. The overfitting is a result of the bias-variance trade off.


## Accomplishments that we're proud of
* Finished a meaningful project in 36 hours.
* Selected the best model and performed accurate prediction of sleep stage.

## What we learned
* The data processing for time series data.
  * The time series data can be analyzed as chunks of statistics.
  * Time series data has dependencies on nearby data points.
* Data visualization is important for the pipeline design.
* Model building
  * Select proper evaluation metrics.
  * There are a lot of models on the shelf and we have to pick the best one based on the evaluation results.

## What's next for Stimulus
* Check the contribution of each statistic, remove redundant statistics to save memory.
* Try to integrate the model into devices that can report real-time sleep stage prediction. The model will change because we will not have datapoints after the predicted time point.

This project highly relies on the ideas from the paper *Do Not Sleep on Traditional Machine Learning: Simple and Interpretable Techniques Are Competitive to Deep Learning for Sleep Scoring*.

Preprint: https://arxiv.org/abs/2207.07753  
Published article: https://doi.org/10.1016/j.bspc.2022.104429

Citation:
```bibtex
@article{vanderdonckt2023donotsleep,
  title={Do not sleep on traditional machine learning: Simple and interpretable techniques are competitive to deep learning for sleep scoring},
  author={Van Der Donckt, Jeroen and Van Der Donckt, Jonas and Deprost, Emiel and Vandenbussche, Nicolas and Rademaker, Michael and Vandewiele, Gilles and Van Hoecke, Sofie},
  journal={Biomedical Signal Processing and Control},
  volume={81},
  pages={104429},
  year={2023},
  publisher={Elsevier}
}
```

---

## How is the code structured?

For each dataset you can find a separate notebook in the [`notebooks`](notebooks) folder.  

The notebooks allow to reproduce the results as they contain;  
1. data loading (see code in [`src` folder](src))
2. pre-processing & feature extraction
3. (seeded) machine learning experiments

| notebook | dataset |
|---|---|
| [SleepEDF-SC +- 30min.ipynb](notebooks/SleepEDF-SC%20%2B-%2030min.ipynb) | `SC-EDF-20` & `SC-EDF-78` |
| [SleepEDF-ST](notebooks/SleepEDF-ST.ipynb) | `SC-EDF-ST` |
| [MASS-SS3](notebooks/MASS-SS3.ipynb) | `MASS SS3` | 

---

## Additional experiments

The [`notebooks/other`](notebooks/other) folder contains some additional experiments;

| notebook | experiment description |
|---|---|
| [inputs_SleepEDF-SC +- 30min.ipynb](notebooks/other/inputs_SleepEDF-SC%20%2B-%2030min.ipynb) | evaluate impact of signal combination on performance for `SC-EDF-20` & `SC-EDF-78`|
| [inputs_SleepEDF-ST.ipynb](notebooks/other/inputs_SleepEDF-ST.ipynb) | evaluate impact of signal combination on performance for `SC-EDF-ST` |
| [inputs_SleepEDF-MASS.ipynb](notebooks/other/inputs_SleepEDF-MASS.ipynb) | evaluate impact of signal combination on performance for `MASS SS3` |
| [feature_selection.ipynb](notebooks/other/feature_selection.ipynb) | show the (little to no) impact of feature selection on performance |
| [feature_space_visualization.ipynb](notebooks/other/feature_space_visualization.ipynb) | `PCA` and `t-SNE` visualization of the feature vector for `SleepEDF-SC +/- 30min`|

A table showing the impact of signal combination on performance can be found in [notebooks/other/signal_combination_impact.md](notebooks/other/signal_combination_impact.md). 

---

## How to install the requirements?

This repository uses [poetry](https://python-poetry.org/) as dependency manager.  
A specification of the dependencies is provided in the [`pyproject.toml`](pyproject.toml) and [`poetry.lock`](poetry.lock) files.

You can install the dependencies in your Python environment by executing the following steps;
1. Install poetry: https://python-poetry.org/docs/#installation
2. Install the dependencies by calling `poetry install`


## How to download the datasets?

This work uses 4 (sub)sets of data;
- `SC-EDF-20`: first 20 patients (40 recordings) of Sleep-EDFx - Sleep Cassette
- `SC-EDF-78`: : all 78 patients (153 recordings) of Sleep-EDFx - Sleep Cassette
- `ST-EDF`: all 22 patients (44 recordings) of Sleep-EDFx - Sleep Telemetry
- `MASS SS3`: all 62 patients (62 recordings) of the MASS - SS3 subset

### [Sleep-EDFx](https://www.physionet.org/content/sleep-edfx/1.0.0/)

Contains the the `SC-EDF-20`, `SC-EDF-78`, and `ST-EDF` subset.

You can download & extract the data via the following commands;
```sh
mkdir data
# Download the data
wget https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip -P data
# Extract all data
unzip data/sleep-edf-database-expanded-1.0.0.zip -d data
```

### [MASS](http://ceams-carsm.ca/mass/)

Contains the [`MASS SS3` subset](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/9MYUCS).

In order to access the data you should submit a request as is described here; http://ceams-carsm.ca/mass/

