# Neural Latent State and Dynamics Inference Workshop

Neural recordings are high-dimensional and complex.
 We aim to find spatiotemporal structure in data in order to "understand" it better, but what is the right kind of structure to look for?
In this workshop, we will introduce the statistical problem of inferring latent state trajectories from high-dimensional neural time series and how it is related to dimensionality reduction methods such as principal component analysis (PCA).
Subsequently, we will introduce the statistically more difficult, but theoretically more satisfying inference of the latent nonlinear dynamical system.
There will be hands-on components to try some of the methods.

## Schedule: 11 October 2022

 - 13:30-14:30 Lecture 1: Latent state trajectories and dimensionality reduction
 - 14:30-14:45 Break & conda/python/code installation
 - 14:45-16:00 Exercise session 1: dimensionality reduction
 - 16:00-17:00 Lecture 2: Latent state dynamics
 - 17:00-17:15 Break
 - 17:15-18:15 Exercise session 2: inferring latent dynamics
 - 18:15-18:30 Summary and discussions

---
## Code setup:

1. Clone the repo using `git clone --recursive`

1. If you missed the `--recursive` option, your `nlb_tools` folder will be empty. Use `git submodule update --init --recursive` to download the submodules

1. Make a conda environment using the requirements.txt with 
    `conda env create -f env.yml`

1. Activate the conda environment using `conda activate lvmworkshop`

## Datasets
We will be focusing on two datasets – a toy dataset of spiking data with low dimensional dynamics governed by
a Van der Pol Oscillator and electrophysiological recordings from the motor cortex (M1) and dorsal premotor cortex (PMd) 
of a monkey during a delayed reaching task.

### Van der Pol Oscillator

  - To setup this dataset move to the code pack folder using `cd code_pack/` then run `python generate_vdp_data.py`

### Monkey reaching task

  - To setup the dataset install dandi using either `pip install dandi` or `conda install -c conda-forge dandi`
  - Then navigate to the data folder in the repo `cd mc_maze/data/`
  - Download the data using `dandi download https://dandiarchive.org/dandiset/000128`

---
## Starting Jupyter Notebook or JupyterLab
Start Jupyter Notebook by typing `jupyter notebook`
or JupyterLab by typing `jupyter lab`

---
Contributors

 - Matt Dowling
 - Tushar Arora
