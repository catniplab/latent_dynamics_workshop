# Neural Latent State and Dynamics Inference Workshop

Neural recordings are high-dimensional and complex.
 We aim to find spatiotemporal structure in data in order to "understand" it better, but what is the right kind of structure to look for?
In this workshop, we will introduce the statistical problem of inferring latent state trajectories from high-dimensional neural time series and how it is related to dimensionality reduction methods such as principal component analysis (PCA).
Subsequently, we will introduce the statistically more difficult, but theoretically more satisfying inference of the latent nonlinear dynamical system.
There will be hands-on components to try some of the methods.

---
## Code setup:

We use [uv](https://docs.astral.sh/uv/) to manage the environment. Prefer conda?
See [INSTALL_conda.md](INSTALL_conda.md).

1. Install uv (one line, no admin needed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
   ```
   Windows and other options: https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repo **with submodules** (required for XFADS, neurofisherSNR, and NLB tools):
   ```bash
   git clone --recurse-submodules https://github.com/catniplab/latent_dynamics_workshop.git
   cd latent_dynamics_workshop
   ```
   If you already cloned without submodules, initialize them with:
   ```bash
   git submodule update --init --recursive
   ```

3. Create the environment. `UV_TORCH_BACKEND=auto` lets uv detect your hardware
   (NVIDIA CUDA, Apple Silicon, or CPU) and fetch the matching PyTorch build:
   ```bash
   UV_TORCH_BACKEND=auto uv sync
   ```
   This creates `.venv/` with all dependencies, including editable installs of
   the `external/` packages:
   - [XFADS](https://github.com/catniplab/xfads/) — [Dowling, Zhao, Park. 2024](https://arxiv.org/abs/2403.01371)
   - [neurofisherSNR](https://github.com/catniplab/neurofisherSNR) — Fisher-information SNR bounds

   Run commands in the environment with `uv run`, e.g. `uv run jupyter lab`, or
   activate it directly with `source .venv/bin/activate`.

4. (Contributors only) This repo strips notebook outputs on commit via
   [nbstripout](https://github.com/kynan/nbstripout). After cloning, register the
   filter locally once so `*.ipynb` diffs stay clean:
   ```bash
   uv tool install nbstripout
   nbstripout --install
   ```

5. (Contributors only) Each notebook is paired to a `py:percent` script via
   [jupytext](https://jupytext.readthedocs.io/) (see `jupytext.toml`), so you can
   edit the `.py` in an editor or the `.ipynb` in Jupyter and keep them in sync.
   jupytext ships with the environment (`uv sync`), so just run:
   ```bash
   uv run jupytext --sync <notebook>.ipynb   # sync after editing either file
   ```
   Commit both the `.ipynb` and its paired `.py`.

## Datasets

We will be focusing on two datasets – a toy dataset of spiking data with low dimensional dynamics governed by
a simulated system and electrophysiological recordings from the motor cortex (M1) and dorsal premotor cortex (PMd) of a monkey during a delayed reaching task.
The simulated system is a continuous attractor system with a ring topology in 2D - i.e., an abstract ring attractor system.

Notebook 00 expects pre-generated Van der Pol data at `vanderpol/data/poisson_obs.h5`. Generate it with:
```bash
uv run python code_pack/generate_vdp_data.py
```

---
## Starting Jupyter Notebook or JupyterLab
Start JupyterLab with `uv run jupyter lab` (or `uv run jupyter notebook`).
If you activated the environment (`source .venv/bin/activate`), you can drop the
`uv run` prefix.

---
## Contributors

Matt Dowling, Tushar Arora, Ayesha Vermani, Abel Sagodi, Mahmoud Elmakki, Hyungju Jeon

## Lecture history
 - Cajal course: Computational Neuroscience (2026)
 - Cajal course: Neuro-AI (2025)
 - Neural Latent State and Dynamics Inference Workshop (2022)
