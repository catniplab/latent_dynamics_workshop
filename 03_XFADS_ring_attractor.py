# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="ALTbmnGoemgT"
# # Nonlinear State Space Modeling via XFADS - simulated ring attractor
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/03_XFADS_ring_attractor.ipynb)
#
# Dowling, M., Zhao, Y., & Park, I. M. (2024). eXponential FAmily Dynamical Systems (XFADS): Large-scale nonlinear Gaussian state-space modeling. The Thirty-Eighth Annual Conference on Neural Information Processing Systems. NeurIPS. https://openreview.net/forum?id=Ln8ogihZ2S
#
# XFADS is our favorite variational autoencoder for nonlinear state space modeling.

# %% id="Ok-TCt-4IWj8"
try:
    import google.colab
    _in_colab = True
except:
    _in_colab = False

# %% [markdown] id="0fIUowr3Ikkl"
# # Installation
#
# Create a `build-system` for the `xfads` package from the `pyproject.toml`
#
# (If you are local, make sure to run this command in the terminal after cd'íng to the project/ workshop main directory and activating the conda environment)
#
# `pip install -e external/xfads/`

# %% colab={"base_uri": "https://localhost:8080/"} id="i34FFj7SIbHE" outputId="57b2daa2-b0e7-4d9f-9107-cdb7e99dcce8"
if _in_colab:
    # !git clone --recurse-submodules https://github.com/catniplab/latent_dynamics_workshop.git

# %% id="fGE66k4UIbJI"
import sys
import os

cwd = os.getcwd()
if _in_colab:
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop"))
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop/external/xfads"))

# %% colab={"base_uri": "https://localhost:8080/"} id="Qh_s-NIiI4AD" outputId="9b16fd62-e8fe-4432-e530-fa735146401d"
if _in_colab:
    # !pip install -e latent_dynamics_workshop/external/xfads/

# %% id="fDaXapziIaka"
import torch
import pytorch_lightning as pl

import torch.nn as nn
import matplotlib.pyplot as plt

import xfads.utils as utils
import xfads.plot_utils as plot_utils

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint

from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn

from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
from xfads.smoothers.nonlinear_smoother import NonlinearFilterSmallL, LowRankNonlinearStateSpaceModel

# %% [markdown] id="2ee29d5d4f2a7536"
# ## ⚙️ 2. Initialize Configuration
#
# We use Hydra to load experiment configs and set up deterministic behavior for reproducibility.

# %% colab={"base_uri": "https://localhost:8080/"} id="2d7c27e664155363" outputId="abec47c0-6528-4728-ee81-c9a60a33d77d"
"""config"""

cfg_dict = {
    # --- graphical model --- #
    'n_latents': 2,
    'n_latents_read': 2,
    'rank_local': 2,
    'rank_backward': 2,
    'n_hidden_dynamics': 64,

    # --- inference network --- #
    'n_samples': 5,
    'n_hidden_local': 128,
    'n_hidden_backward': 64,

    # --- hyperparameters --- #
    'use_cd': False,
    'p_mask_a': 0.8,
    'p_mask_b': 0.0,
    'p_mask_apb': 0.0,
    'p_mask_y_in': 0.0,
    'p_local_dropout': 0.4,
    'p_backward_dropout': 0.0,
    'lr_gamma_decay': 0.99,

    # --- training --- #
    'device': 'cpu',
    'data_device': 'cpu',
    'lr': 1e-3,
    'n_epochs': 5,
    'batch_sz': 128,

    # --- misc --- #
    'bin_sz': 20e-3,
    'bin_sz_ms': 20,
    'seed': 1234,
    'default_dtype': torch.float32,
}

class Cfg(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"'Cfg' object has no attribute '{attr}'")

cfg = Cfg(cfg_dict)

# Set devices and seed
if not torch.cuda.is_available():
    cfg.device = 'cpu'
    cfg.data_device = 'cpu'

pl.seed_everything(cfg.seed, workers=True)
torch.set_default_dtype(cfg.default_dtype)

if cfg.device == 'cuda':
    torch.cuda.empty_cache()

# %% colab={"base_uri": "https://localhost:8080/"} id="oyjxd98x7zpy" outputId="062248b9-3bc5-41eb-daf1-6f9eb45a957a"
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

cfg.device = device
print(f"Using device: {device}")

# A quick test
matrix_a = torch.randn(1024, 1024, device=device)
result = torch.matmul(matrix_a, matrix_a)

# cfg['n_epochs'] = 50  # reduced epochs for testing

# %% [markdown] id="5166b21e3be10cb2"
# ## 📈 3. Simulate Data
#
# We generate data from a 2D ring attractor latent dynamic system, projecting into 100-dimensional observations using a fixed linear readout.

# %% id="ffb244a2edaa6d8"
n_trials = 3000
n_neurons = 100
n_time_bins = 75

mean_fn = utils.RingAttractorDynamics(bin_sz=1e-1, w=0.0)
C = utils.FanInLinear(cfg.n_latents, n_neurons, device=cfg.device).requires_grad_(False)

Q_diag = 5e-3 * torch.ones(cfg.n_latents, device=cfg.device)
Q_0_diag = 1.0 * torch.ones(cfg.n_latents, device=cfg.device)
R_diag = 1e-1 * torch.ones(n_neurons, device=cfg.device)
m_0 = torch.zeros(cfg.n_latents, device=cfg.device)

z = utils.sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins)
y = C(z) + torch.sqrt(R_diag) * torch.randn((n_trials, n_time_bins, n_neurons), device=cfg.device)
y = y.detach()

# %% [markdown] id="8807079c3ba83cee"
# ## 📈 4. Visualize Latent Trajectories
#
# Let's look at some sample trajectories from the 2D latent space.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 564} id="861d352ca0ed99d5" outputId="5585cf1f-07ee-45b6-96f9-09473bfee6ac"
fig, axs = plt.subplots(figsize=(6, 6))
for i in range(40):
    axs.plot(z[i, :, 0].cpu(), z[i, :, 1].cpu(), alpha=0.6, linewidth=0.5)

plot_utils.plot_two_d_vector_field(mean_fn, axs, min_xy=-2, max_xy=2)
axs.set_title("Sample Latent Trajectories (2D Ring Attractor)")
axs.set_xlabel("Latent dim 1")
axs.set_ylabel("Latent dim 2")
axs.set_xlim(-2, 2)
axs.set_ylim(-2, 2)
axs.set_box_aspect(1.0)
plt.show()


# %% [markdown] id="f9cfaaf7d41583a4"
# ##  5. Prepare Train/Validation Dataloaders
#
# Split the simulated data into training and validation sets and prepare PyTorch dataloaders.
#

# %% id="3f0486dd107fdfad"
def collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, (tuple, list)):
        return tuple(torch.stack([b[i] for b in batch]).to(cfg.device) for i in range(len(elem)))
    else:
        return torch.stack(batch).to(cfg.device)

y_train, z_train = y[:2*n_trials//3], z[:2*n_trials//3]
y_valid, z_valid = y[2*n_trials//3:], z[2*n_trials//3:]

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(y_train), batch_size=cfg.batch_sz, shuffle=True, collate_fn=collate_fn
)
valid_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(y_valid), batch_size=cfg.batch_sz, shuffle=False, collate_fn=collate_fn
)

# %% [markdown] id="e0a019425d36406e"
# ## 🧱 6. Define Model Components
#
# We define the following:
# - A Gaussian likelihood with a fixed observation noise
# - A nonlinear Gaussian dynamics module
# - A prior over the initial condition
# - Local and backward encoders for amortized inference

# %% id="2e9a0d03984446ba"
# Likelihood
H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
readout_fn = nn.Sequential(H, C)
likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons, R_diag, device=cfg.device, fix_R=True)

# Dynamics
dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

# Initial condition
initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

# Encoders
backward_encoder = BackwardEncoderLRMvn(
    cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
    rank_local=cfg.rank_local, rank_backward=cfg.rank_backward, device=cfg.device
)
local_encoder = LocalEncoderLRMvn(
    cfg.n_latents, n_neurons, cfg.n_hidden_local, cfg.n_latents,
    rank=cfg.rank_local, device=cfg.device, dropout=cfg.p_local_dropout
)

# Nonlinear filtering
nl_filter = NonlinearFilterSmallL(dynamics_mod, initial_condition_pdf, device=cfg.device)

# %% [markdown] id="40405b7f1966f90a"
# ## 🧠 7. Assemble the State Space Model
#
# We combine dynamics, likelihood, encoders, and filtering into a complete latent variable model.

# %% id="d449ba34d5b2aaa2"
ssm = LowRankNonlinearStateSpaceModel(
    dynamics_mod, likelihood_pdf, initial_condition_pdf,
    backward_encoder, local_encoder, nl_filter, device=cfg.device
)

# %% [markdown] id="d726c69556e898d1"
# ## 🔁 8. Train the Model Using PyTorch Lightning
#
# We use `LightningNonlinearSSM` for training. Logging and checkpointing are included.
#

# %% id="mkmlKUzQmbTz"
from pytorch_lightning.callbacks import Timer
timer = Timer()

# %% colab={"base_uri": "https://localhost:8080/", "height": 393, "referenced_widgets": ["b548f1bb9e7b4ffea5cc574ca00755e9", "2d970c81e4ec4b26916da8450ca9e068"]} id="529df5e92cc355f9" outputId="1ed11b8c-759d-47f4-bdb1-3f769ce69a0a"
train_from_scratch = False

if cfg.device == 'cuda':
    torch.cuda.empty_cache()

if _in_colab:
    log_path = 'latent_dynamics_workshop/logs/ring_attractor'
    ckpts_path = 'latent_dynamics_workshop/ckpts/ring_attractor'
else:
    log_path = './logs/ring_attractor'
    ckpts_path = './ckpts/ring_attractor'

if train_from_scratch:
    seq_vae = LightningNonlinearSSM(ssm, cfg)

    csv_logger = CSVLogger(log_path, name=f'r_y_{cfg.rank_local}_r_b_{cfg.rank_backward}', version='noncausal')
    ckpt_callback = ModelCheckpoint(
        save_top_k=3, monitor='valid_loss', mode='min',
        dirpath=ckpts_path, filename='{epoch:0}_{valid_loss:.2f}'
    )

    trainer = pl.Trainer(
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.0,
        default_root_dir='lightning/',
        callbacks=[RichProgressBar(), ckpt_callback, timer],
        accelerator=cfg.device,  # disable autodetection (no MPS support!)
        logger=csv_logger
    )

    trainer.fit(model=seq_vae, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    torch.save(ckpt_callback.best_model_path, f'{ckpts_path}/best_model_path.pt')

else:
    seq_vae = LightningNonlinearSSM.load_from_checkpoint(f'{ckpts_path}/example_model.ckpt', ssm=ssm, cfg=cfg)

# %% colab={"base_uri": "https://localhost:8080/"} id="aXzym_zRdSzV" outputId="faa12d91-9fba-4b2b-e552-483ff6f763fa"
print(timer.time_elapsed("train"))  # total training time
print(timer.time_elapsed("validate"))  # validation time

# %% [markdown] id="1bb7d9f2fd34e1af"
# ## ✅ Done!
#
# The model is now trained. You can proceed with:
# - Plotting smoothed trajectories.
# - Visualizing uncertainty.
# - Comparing inferred vs. ground truth latents (since this was a synthetic dataset).
#

# %% [markdown] id="89ef34e7cb93bc02"
# ## 🌀 9. Visualize Learned Dynamics and Simulated Trajectories
#
# Now that training is complete, we can explore what the model has learned.  
# This section:
# - Seeds the latent space with initial conditions.
# - Rolls out the learned dynamics forward in time.
# - Overlays those trajectories onto the learned dynamics vector field.
# python
# Copy
# Edit
#

# %% id="19d7254c298d909a"
# Define number of rollout samples and rollout length
n_ex_samples = 1
n_ex_trials = 50
n_ex_time_bins = 50

# Sample initial latent states (z_0): a mix of small and large amplitude noise
z_0 = torch.zeros((n_ex_samples, n_ex_trials, 2))
z_0[:, ::2] = 0.2 * torch.randn_like(z_0[:, ::2])   # small amplitude for even-indexed trials
z_0[:, 1::2] = 2.0 * torch.randn_like(z_0[:, 1::2])  # large amplitude for odd-indexed trials

# Predict forward using the learned dynamics (no encoder or data used here)
z_prd = seq_vae.ssm.predict_forward(z_0, n_ex_time_bins).detach()

# %% [markdown] id="996d07535c3a411d"
# ### 🧭 Plot: Learned Dynamics Vector Field + Predicted Latent Trajectories
#
# The vector field shows the learned mean dynamics function.
# Each curve shows a rollout of the model's latent trajectory starting from a different `z_0`.
#     

# %% colab={"base_uri": "https://localhost:8080/", "height": 545} id="ec996ab8b81f07bd" outputId="434ff5a9-e594-4652-9e77-7b535247db08"
fig, axs = plt.subplots(figsize=(6, 6))
axs.set_box_aspect(1.0)
axs.set_xlim(-2.0, 2.0)
axs.set_ylim(-2.0, 2.0)
axs.set_title("Learned Dynamics and Autonomous Latent Trajectories")

# Plot learned vector field over the 2D latent space
plot_utils.plot_two_d_vector_field(
    seq_vae.ssm.dynamics_mod.mean_fn,
    axs,
    min_xy=-2,
    max_xy=2,
)

# Overlay predicted trajectories
for i in range(50):  # plot 50 of the 50
    axs.plot(z_prd[0, i, :, 0].cpu(), z_prd[0, i, :, 1].cpu(), lw=0.5, alpha=0.6)

plt.show()

# %% id="aeb212b3aeaa9f77"
