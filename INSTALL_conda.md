# Conda installation (fallback)

The recommended installation uses [uv](https://docs.astral.sh/uv/) - see the
README. This conda path is kept for environments where conda is preferred (e.g.
institutional setups or when installing PyTorch through conda-forge).

For installation of conda follow the instructions here:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

## Setup

1. Clone with submodules (see README step 1).

2. Create the environment (Linux, macOS, and Windows):
   ```bash
   conda env create -f env.yml
   ```

3. Activate it:
   ```bash
   conda activate lvmworkshop
   ```
   The environment installs the editable packages from `external/`
   ([XFADS](https://github.com/catniplab/xfads/) and
   [neurofisherSNR](https://github.com/catniplab/neurofisherSNR)).

   If the pip editable installs failed during env creation, run manually from
   the repo root:
   ```bash
   pip install -e external/xfads/ -e external/neurofisherSNR/
   ```

For contributor tooling (nbstripout, jupytext) and dataset generation, see the
README.
