# Neural Latent State and Dynamics Inference Workshop

Neural recordings are high-dimensional and complex.
 We aim to find spatiotemporal structure in data in order to "understand" it better, but what is the right kind of structure to look for?
In this workshop, we will introduce the statistical problem of inferring latent state trajectories from high-dimensional neural time series and how it is related to dimensionality reduction methods such as principal component analysis (PCA).
Subsequently, we will introduce the statistically more difficult, but theoretically more satisfying inference of the latent nonlinear dynamical system.
There will be hands-on components to try some of the methods.

---
## Conda installation

For installation of conda follow the instructions here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#

## Git installation (for Windows)

Windows users would need to install and work in the Git BASH terminal.
For installing git, see: https://git-scm.com/downloads

---
## Code setup:

1. Clone or download this repo. \
   Use this command to clone the repo along with all its submodules, ensuring you get the full project, including any nested dependencies: \
   `git clone --recurse-submodules https://github.com/catniplab/latent_dynamics_workshop.git`

2. Make a conda environment using the requirements.txt with 
    - For Linux and MacOS use   `conda env create -f env.yml`
    - For Windows use `conda env create -f env_windows.yml` and then \
        GPU: `pip install jax==0.3.13 https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.7+cuda11.cudnn82-cp39-none-win_amd64.whl` \
        or \
        CPU: `pip install "jax[cpu]===0.3.14" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver` \
        and finally
        `pip install git+https://github.com/yuanz271/vlgpax.git`

3. Activate the conda environment using `conda activate lvmworkshop`

4. cd to the project main directory (`cd latent_dynamics_workshop`), after cloning the repo, and run the following command in the terminal to install [XFADS](https://github.com/catniplab/xfads/) (eXponential FAmily Dynamical Systems), [Dowling, Zhao, Park. 2024](https://arxiv.org/abs/2403.01371),  and its dependencies \
   `pip install -e xfads/` \
   (`xfads/` is the submodule folder that contains the `pyproject.toml` file and the `xfads` package folder)

## Datasets

We will be focusing on two datasets – a toy dataset of spiking data with low dimensional dynamics governed by
a Van der Pol Oscillator and electrophysiological recordings from the motor cortex (M1) and dorsal premotor cortex (PMd)
of a monkey during a delayed reaching task.

### Van der Pol Oscillator

  - To setup this dataset move to the code pack folder using `cd code_pack/` then run `python generate_vdp_data.py`

### Monkey reaching task

  - No action required. Included in the github.com repo.

---
## Starting Jupyter Notebook or JupyterLab
Start Jupyter Notebook by typing `jupyter notebook`
or JupyterLab by typing `jupyter lab`


---
Contributors

 - Matt Dowling
 - Tushar Arora
 - Ayesha Vermani
 - Abel Sagodi
 - Mahmoud Elmakki
