# Neural Latent State and Dynamics Inference Workshop

Neural recordings are high-dimensional and complex.
 We aim to find spatiotemporal structure in data in order to "understand" it better, but what is the right kind of structure to look for?
In this workshop, we will introduce the statistical problem of inferring latent state trajectories from high-dimensional neural time series and how it is related to dimensionality reduction methods such as principal component analysis (PCA).
Subsequently, we will introduce the statistically more difficult, but theoretically more satisfying inference of the latent nonlinear dynamical system.
There will be hands-on components to try some of the methods.

---
## Conda installation

For installation of conda follow the instructions here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#

---
## Code setup:

1. Clone this repo **with submodules** (required for XFADS, neurofisherSNR, and NLB tools):
   ```bash
   git clone --recurse-submodules https://github.com/catniplab/latent_dynamics_workshop.git
   cd latent_dynamics_workshop
   ```
   If you already cloned without submodules, initialize them with:
   ```bash
   git submodule update --init --recursive
   ```

2. Create the conda environment (Linux, macOS, and Windows):
   ```bash
   conda env create -f env.yml
   ```

3. Activate the environment: `conda activate lvmworkshop`

   The environment installs editable packages from `external/`:
   - [XFADS](https://github.com/catniplab/xfads/) — [Dowling, Zhao, Park. 2024](https://arxiv.org/abs/2403.01371)
   - [neurofisherSNR](https://github.com/catniplab/neurofisherSNR) — Fisher-information SNR bounds

   If pip editable installs failed during env creation, run manually from the repo root:
   ```bash
   pip install -e external/xfads/ -e external/neurofisherSNR/
   ```

4. (Contributors only) This repo strips notebook outputs on commit via
   [nbstripout](https://github.com/kynan/nbstripout). After cloning, register the
   filter locally once so `*.ipynb` diffs stay clean:
   ```bash
   pip install nbstripout   # or: uv tool install nbstripout
   nbstripout --install
   ```

## Datasets

We will be focusing on two datasets – a toy dataset of spiking data with low dimensional dynamics governed by
a simulated system and electrophysiological recordings from the motor cortex (M1) and dorsal premotor cortex (PMd) of a monkey during a delayed reaching task.
The simulated system is a continuous attractor system with a ring topology in 2D - i.e., an abstract ring attractor system.

Notebook 00 expects pre-generated Van der Pol data at `vanderpol/data/poisson_obs.h5`. Generate it with:
```bash
python code_pack/generate_vdp_data.py
```

---
## Starting Jupyter Notebook or JupyterLab
Start Jupyter Notebook by typing `jupyter notebook`
or JupyterLab by typing `jupyter lab`

---
## Contributors

 - Matt Dowling
 - Tushar Arora
 - Ayesha Vermani
 - Abel Sagodi
 - Mahmoud Elmakki

## Lecture history
 - Cajal course on Neuro-AI (2025)
 - Neural Latent State and Dynamics Inference Workshop (2022)
