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
# # XFADS learns a ring attractor from data
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/05_xfads_ring_attractor.ipynb)
#
# **Takeaway:** given only noisy high-dimensional observations, XFADS recovers the
# *nonlinear latent vector field* that generated them - here, a 2D ring attractor.
#
# XFADS is a structured variational autoencoder for nonlinear Gaussian
# state-space models. It ties together three ideas you have already met in this
# workshop:
#
# - **Poisson / Gaussian observations** from a latent (lecture notes: exponential-family
#   observations, `sec:expfam`),
# - **filtering and smoothing** to infer the latent path (`sec:smoothing`),
# - **variational inference** with an **amortized recognition network**
#   (`sec:vi`, `sec:amortized`).
#
# The XFADS model itself is summarized in the lecture notes section **XFADS**
# (`sec:xfads`). See also Dowling, Zhao & Park (2024),
# [XFADS](https://arxiv.org/abs/2403.01371) (NeurIPS 2024).
#
# > **Branch points.**
# > - This core notebook *loads a trained checkpoint*. To train it yourself, run
# >   the clearly-marked optional cell in Section 6. A deeper companion notebook
# >   explores the inference-network hyperparameters (ranks, masking, dropout).
# > - **Next core notebook:** XFADS applied to real neural spike trains (MC-Maze).

# %% [markdown]
# ## Setup
#
# On Colab we clone the repo and install the `xfads` package. Locally, install it
# once with `pip install -e external/xfads/` and skip these cells.

# %% id="Ok-TCt-4IWj8"
try:
    import google.colab
    _in_colab = True
except ImportError:
    _in_colab = False

# %% colab={"base_uri": "https://localhost:8080/"} id="i34FFj7SIbHE"
if _in_colab:
    # !git clone --recurse-submodules https://github.com/catniplab/latent_dynamics_workshop.git
    pass

# %% id="fGE66k4UIbJI"
import sys
import os

cwd = os.getcwd()
if _in_colab:
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop"))
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop/external/xfads"))

# %% colab={"base_uri": "https://localhost:8080/"} id="Qh_s-NIiI4AD"
if _in_colab:
    # !pip install -e latent_dynamics_workshop/external/xfads/
    pass

# %% id="fDaXapziIaka"
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import xfads.utils as utils
import xfads.plot_utils as plot_utils

from xfads.ssm_modules.dynamics import DenseGaussianDynamics, DenseGaussianInitialCondition
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
from xfads.smoothers.nonlinear_smoother import NonlinearFilterSmallL, LowRankNonlinearStateSpaceModel

# %% [markdown]
# ## Configuration
#
# We expose only the three knobs you might touch: latent dimension, number of
# training epochs, and learning rate. Everything else (encoder ranks, masking
# probabilities, dropout, decay schedule) is inference-network plumbing kept in a
# default dictionary - the deeper companion notebook is where you tune those.

# %%
# The three knobs worth touching.
n_latents = 2       # dimension of the latent state z
n_epochs = 5        # only used on the optional from-scratch path
lr = 1e-3           # learning rate for from-scratch training

# Inference-network defaults (plumbing - safe to ignore on first read).
cfg_dict = {
    'n_latents': n_latents,
    'n_latents_read': n_latents,
    'rank_local': 2,
    'rank_backward': 2,
    'n_hidden_dynamics': 64,
    'n_samples': 5,
    'n_hidden_local': 128,
    'n_hidden_backward': 64,
    'use_cd': False,
    'p_mask_a': 0.8,
    'p_mask_b': 0.0,
    'p_mask_apb': 0.0,
    'p_mask_y_in': 0.0,
    'p_local_dropout': 0.4,
    'p_backward_dropout': 0.0,
    'lr_gamma_decay': 0.99,
    'device': 'cpu',
    'data_device': 'cpu',
    'lr': lr,
    'n_epochs': n_epochs,
    'batch_sz': 128,
    'bin_sz': 20e-3,
    'bin_sz_ms': 20,
    'seed': 1234,
    'default_dtype': torch.float32,
}


class Cfg(dict):
    """Dict with attribute access, so cfg.n_latents works like the xfads API expects."""

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'Cfg' object has no attribute '{attr}'")


cfg = Cfg(cfg_dict)

# Device selection: XFADS uses CPU or CUDA. We intentionally avoid MPS on Apple
# silicon (unsupported by the filtering kernels), so this runs CPU-only there.
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg.data_device = cfg.device
print(f"Using device: {cfg.device}")

pl.seed_everything(cfg.seed, workers=True)
torch.set_default_dtype(cfg.default_dtype)

# %% [markdown]
# ## 1. Simulate ring-attractor data
#
# The generator is a 2D latent with **ring-attractor** dynamics: any initial state
# relaxes onto the unit circle. A fixed linear readout `C` lifts the 2D latent
# into 100 noisy observation channels (think 100 neurons).
#
# We name the latent `z` to match the xfads library and the workshop's z-convention.
# In the lecture notes the same latent is written `x`; read **z (notebook/library)
# = x (notes)** throughout.

# %% id="ffb244a2edaa6d8"
n_trials = 3000
n_neurons = 100
n_time_bins = 75

# Ground-truth generative model.
mean_fn = utils.RingAttractorDynamics(bin_sz=1e-1, w=0.0)   # w = tangential rotation speed
C = utils.FanInLinear(cfg.n_latents, n_neurons, device=cfg.device).requires_grad_(False)

Q_diag = 5e-3 * torch.ones(cfg.n_latents, device=cfg.device)   # process noise
Q_0_diag = 1.0 * torch.ones(cfg.n_latents, device=cfg.device)  # initial-state spread
R_diag = 1e-1 * torch.ones(n_neurons, device=cfg.device)       # observation noise
m_0 = torch.zeros(cfg.n_latents, device=cfg.device)            # initial-state mean

# z: latent trajectories; y: noisy observations y = C z + noise.
z = utils.sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins)
y = C(z) + torch.sqrt(R_diag) * torch.randn((n_trials, n_time_bins, n_neurons), device=cfg.device)
y = y.detach()

# %% [markdown]
# ## 2. Look at the object first
#
# Before any model, plot the ground-truth latent trajectories over the true vector
# field. This is the target XFADS must recover from `y` alone.
#
# > **Predict (do this before running the cell):** the generator uses
# > `RingAttractorDynamics(w=0.0)`. Where do the 40 trajectories end up, and do
# > they circulate?
#
# <details>
# <summary>Solution</summary>
#
# They collapse onto the unit-radius ring (radius = 1). With `w = 0.0` there is no
# tangential component, so the flow is purely radial - trajectories move straight in
# toward the circle and stop; they do not rotate around it.
#
# </details>

# %% colab={"base_uri": "https://localhost:8080/", "height": 564} id="861d352ca0ed99d5"
fig, axs = plt.subplots(figsize=(6, 6))
for i in range(40):
    axs.plot(z[i, :, 0].cpu(), z[i, :, 1].cpu(), alpha=0.6, linewidth=0.5)

plot_utils.plot_two_d_vector_field(mean_fn, axs, min_xy=-2, max_xy=2)
axs.set_title("Ground-truth latent trajectories (2D ring attractor)")
axs.set_xlabel("latent z1")
axs.set_ylabel("latent z2")
axs.set_xlim(-2, 2)
axs.set_ylim(-2, 2)
axs.set_box_aspect(1.0)
plt.show()

# %% [markdown]
# > **Tweak and observe:** re-create `mean_fn` with `w=0.5` instead of `w=0.0`,
# > rebuild the vector field, and re-plot (you can copy the two lines below).
# > How does the field change?
#
# <details>
# <summary>Solution</summary>
#
# ```python
# mean_fn_rot = utils.RingAttractorDynamics(bin_sz=1e-1, w=0.5)
# fig, axs = plt.subplots(figsize=(6, 6))
# plot_utils.plot_two_d_vector_field(mean_fn_rot, axs, min_xy=-2, max_xy=2)
# axs.set_box_aspect(1.0); plt.show()
# ```
#
# `w` adds a tangential (rotational) component: the flow still contracts onto the
# unit ring, but now also circulates around it, turning the point attractor-ring
# into a rotating limit cycle. Leave the data generation at `w=0.0` for the rest of
# the notebook so it matches the checkpoint.
#
# </details>

# %% [markdown]
# ## 3. Train / validation split
#
# Standard two-thirds / one-third split. Only `y` is fed to the model; `z` is held
# aside so we can check the recovered dynamics against the truth at the end.

# %% id="3f0486dd107fdfad"
def collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, (tuple, list)):
        return tuple(torch.stack([b[i] for b in batch]).to(cfg.device) for i in range(len(elem)))
    return torch.stack(batch).to(cfg.device)


y_train = y[:2 * n_trials // 3]
y_valid = y[2 * n_trials // 3:]

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(y_train), batch_size=cfg.batch_sz, shuffle=True, collate_fn=collate_fn
)
valid_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(y_valid), batch_size=cfg.batch_sz, shuffle=False, collate_fn=collate_fn
)

# %% [markdown]
# ## 4. Assemble the state-space model
#
# XFADS has four learnable-or-fixed pieces (lecture notes `sec:xfads`):
#
# - **Likelihood** `p(y | z)`: a Gaussian readout. **Important:** here we hand the
#   model the ground-truth readout `C` (frozen, `requires_grad_(False)`) and fix the
#   observation noise `R` (`fix_R=True`). So *both* the readout matrix and the noise
#   are pinned to the true generative values - the model does **not** learn them.
#   This is deliberate: it isolates the hard part, learning the nonlinear dynamics
#   and the inference network. In practice you would let `C` (and often `R`) be
#   learned.
# - **Readout mask** `H`: lets the readout use a subset of latents. Here
#   `n_latents_read == n_latents == 2`, so `H` is the identity (a no-op) and the
#   effective map is just `C`.
# - **Dynamics** `p(z_t | z_{t-1})`: a GRU-based nonlinear Gaussian transition -
#   **this is what XFADS learns.**
# - **Encoders**: local + backward recognition networks that amortize the posterior
#   over `z` (`sec:amortized`).

# %% id="2e9a0d03984446ba"
# Likelihood: readout is H then C (H is identity here), observation noise R fixed.
H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
readout_fn = nn.Sequential(H, C)
likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons, R_diag, device=cfg.device, fix_R=True)

# Dynamics: the nonlinear latent transition XFADS learns.
dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

# Prior over the initial condition.
initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

# Amortized recognition network: local + backward encoders.
backward_encoder = BackwardEncoderLRMvn(
    cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
    rank_local=cfg.rank_local, rank_backward=cfg.rank_backward, device=cfg.device
)
local_encoder = LocalEncoderLRMvn(
    cfg.n_latents, n_neurons, cfg.n_hidden_local, cfg.n_latents,
    rank=cfg.rank_local, device=cfg.device, dropout=cfg.p_local_dropout
)

# Nonlinear filter tying dynamics + initial condition together.
nl_filter = NonlinearFilterSmallL(dynamics_mod, initial_condition_pdf, device=cfg.device)

# The full low-rank nonlinear state-space model.
ssm = LowRankNonlinearStateSpaceModel(
    dynamics_mod, likelihood_pdf, initial_condition_pdf,
    backward_encoder, local_encoder, nl_filter, device=cfg.device
)

# %% [markdown]
# ## 5. Load the trained model
#
# The core path loads a checkpoint we trained for you, so the notebook runs in
# seconds. Training XFADS maximizes a variational ELBO (`sec:vi`); the from-scratch
# recipe is in the optional cell just below.

# %% id="checkpoint-load"
ckpts_path = 'latent_dynamics_workshop/ckpts/ring_attractor' if _in_colab else './ckpts/ring_attractor'

seq_vae = LightningNonlinearSSM.load_from_checkpoint(
    f'{ckpts_path}/example_model.ckpt', ssm=ssm, cfg=cfg
)

# %% [markdown]
# ## 6. (Optional) Train it yourself
#
# Skip this on a first pass - the checkpoint above is already loaded. Set
# `train_from_scratch = True` to fit the model with PyTorch Lightning (5 epochs on
# CPU is minutes, not seconds). This overwrites `seq_vae` with your freshly trained
# model. To explore the encoder ranks, masking, and dropout that drive training,
# use the deeper companion notebook.

# %% colab={"base_uri": "https://localhost:8080/"} id="529df5e92cc355f9"
train_from_scratch = False  # <- flip to True to train

if train_from_scratch:
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, Timer

    log_path = 'latent_dynamics_workshop/logs/ring_attractor' if _in_colab else './logs/ring_attractor'
    timer = Timer()

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
        accelerator=cfg.device,  # CPU or CUDA only (no MPS)
        logger=csv_logger,
    )

    trainer.fit(model=seq_vae, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    torch.save(ckpt_callback.best_model_path, f'{ckpts_path}/best_model_path.pt')
    print(f"training time (s): {timer.time_elapsed('train'):.1f}")

# %% [markdown]
# ## 7. Did XFADS recover the ring?
#
# Now the payoff. We take the **learned** dynamics (no data, no encoder) and roll it
# forward from a spread of initial states, then overlay the trajectories on the
# learned vector field. If XFADS worked, we should see the ring re-emerge.

# %% id="19d7254c298d909a"
n_ex_samples = 1
n_ex_trials = 50
n_ex_time_bins = 50

# Seed initial latent states z_0: a mix of small- and large-amplitude starts.
z_0 = torch.zeros((n_ex_samples, n_ex_trials, cfg.n_latents))
z_0[:, ::2] = 0.2 * torch.randn_like(z_0[:, ::2])   # small-amplitude, even-indexed trials
z_0[:, 1::2] = 2.0 * torch.randn_like(z_0[:, 1::2])  # large-amplitude, odd-indexed trials

# Roll the LEARNED dynamics forward. The second argument is the rollout horizon.
z_prd = seq_vae.ssm.predict_forward(z_0, n_ex_time_bins).detach()

# %% [markdown]
# > **Stretch (optional) - fill one line:** the rollout above uses only the learned
# > dynamics; no observations `y` and no encoder are involved. Rewrite the
# > `predict_forward` call so *you* supply the horizon:
# >
# > ```python
# > z_prd = seq_vae.ssm.predict_forward(z_0, ___)  # YOUR CODE HERE: rollout length
# > ```
#
# <details>
# <summary>Solution</summary>
#
# ```python
# z_prd = seq_vae.ssm.predict_forward(z_0, n_ex_time_bins).detach()
# ```
#
# The horizon is `n_ex_time_bins` (50 steps). Because `predict_forward` consumes only
# `z_0` and the learned transition, the ring you see is generated by the dynamics
# alone - proof that XFADS captured the vector field, not just fit the data.
#
# </details>

# %% colab={"base_uri": "https://localhost:8080/", "height": 545} id="ec996ab8b81f07bd"
fig, axs = plt.subplots(figsize=(6, 6))
axs.set_box_aspect(1.0)
axs.set_xlim(-2.0, 2.0)
axs.set_ylim(-2.0, 2.0)
axs.set_title("Learned dynamics and autonomous rollouts")

# Learned mean vector field.
plot_utils.plot_two_d_vector_field(seq_vae.ssm.dynamics_mod.mean_fn, axs, min_xy=-2, max_xy=2)

# Overlay the autonomous rollouts.
for i in range(n_ex_trials):
    axs.plot(z_prd[0, i, :, 0].cpu(), z_prd[0, i, :, 1].cpu(), lw=0.5, alpha=0.6)

plt.show()

# %% [markdown]
# ## You can now...
#
# ...fit a **nonlinear** latent state-space model with XFADS, load a trained
# checkpoint, and roll out its learned dynamics to check whether it recovered the
# generating vector field.
#
# **Transfer prompt:** swap the ring-attractor generator for a dynamical system you
# care about (a limit cycle, a bistable switch, or your own recorded population),
# refit, and roll out the learned dynamics. Does the autonomous flow reproduce the
# phenomenon you started from?
#
# **Next:** the following core notebook applies this same XFADS machinery to real
# neural spike trains (MC-Maze) with Poisson observations (`sec:expfam`).
