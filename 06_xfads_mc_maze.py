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
#     display_name: Python (lvmworkshop)
#     language: python
#     name: lvmworkshop
# ---

# %% [markdown]
# # XFADS on MC_Maze: infer, forecast, and reconstruct population activity
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/06_xfads_mc_maze.ipynb)
#
# **Takeaway:** a trained XFADS state-space model turns spikes into a low-dimensional
# latent `z` that we can *smooth*, *filter*, *forecast*, and read back out as
# single-neuron firing rates.
#
# This notebook is the core through-line: load real reaching data, build the model,
# load a pretrained checkpoint, run the three inference modes, and reconstruct rates.
# We do **not** train here (the checkpoint is provided) and we do **not** decode
# behavior here.
#
# **Branch point:**
# - Optional companion: **`07_decoding_and_evaluation`** takes the `z` and rates from
#   this notebook and asks *how good are they?* - ridge decoding of hand velocity,
#   k-step forecasting curves, a PCA-vs-R2 sweep, and predictive log-likelihood.
# - This is the last XFADS applications notebook; the earlier `03_XFADS_ring_attractor`
#   shows the same machinery on a synthetic system where ground truth is known.
#
# Background in the lecture notes: *Poisson observations* (`sec:expfam`),
# *RTS smoothing and forecasting* (`sec:smoothing`), *variational inference*
# (`sec:vi`), *amortized inference / VAE* (`sec:amortized`), and *XFADS* (`sec:xfads`).
#
# Reference: [Dowling, Zhao, Park. 2024](https://arxiv.org/abs/2403.01371).

# %%
try:
    import google.colab
    _in_colab = True
except ImportError:
    _in_colab = False

# %% [markdown]
# # Installation
#
# The XFADS package is installed editable from the submodule. Locally, run this once
# in your terminal from the workshop root (with the conda environment active):
#
# `pip install -e external/xfads/`
#
# On Colab the next two cells clone the repo and install it for you.

# %%
if _in_colab:
    # !git clone --recurse-submodules https://github.com/catniplab/latent_dynamics_workshop.git
    pass

# %%
import sys
import os

cwd = os.getcwd()
if _in_colab:
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop"))
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop/external/xfads"))

# %%
if _in_colab:
    # !pip install -e latent_dynamics_workshop/external/xfads/
    pass

# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as lightning

import warnings
warnings.filterwarnings("ignore")  # silence matplotlib/lightning chatter for teaching

# %% [markdown]
# # Model and training parameters
#
# One config dict drives the graphical model size, the inference network, and
# training. We only *use* these to rebuild the model that matches the checkpoint;
# the shapes here must match the ones the pretrained model was fit with.

# %%
"""config"""

cfg = {
    # --- graphical model --- #
    'n_latents': 40,
    'n_latents_read': 35,

    'rank_local': 15,
    'rank_backward': 5,

    'n_hidden_dynamics': 128,

    # --- inference network --- #
    'n_samples': 25,
    'n_hidden_local': 256,
    'n_hidden_backward': 128,

    # --- hyperparameters --- #
    'use_cd': False,
    'p_mask_a': 0.0,
    'p_mask_b': 0.0,
    'p_mask_apb': 0.0,
    'p_mask_y_in': 0.0,
    'p_local_dropout': 0.4,
    'p_backward_dropout': 0.0,

    # --- training --- #
    'device': 'cpu',
    'data_device': 'cpu',

    'lr': 1e-3,
    'lr_gamma_decay': 0.997,
    'n_epochs': 3,
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

cfg = Cfg(cfg)

if not torch.cuda.is_available():
    cfg.device = 'cpu'
    cfg.data_device = 'cpu'

lightning.seed_everything(cfg.seed, workers=True)
torch.set_default_dtype(torch.float32)

if cfg.device == 'cuda':
    torch.cuda.empty_cache()

# %% [markdown]
# # Load the data
#
# <p align="center">
#   <img src="https://github.com/catniplab/latent_dynamics_workshop/blob/main/img/maze.png?raw=1"/>
# </p>
#
# [MC_Maze](https://neurallatents.github.io/datasets.html) is a delayed center-out
# reaching task through a maze of barriers, giving straight and curved reaches.
#
# **Neural activity:** binned at 20 ms, 45 bins/trial, window -240 ms to +660 ms
# relative to movement onset.
#
# **Kinematics (hand velocity):** binned at 20 ms, 35 bins/trial, window -40 ms to
# +660 ms relative to movement onset.
#
# Note the two windows do not start at the same time: the neural window begins
# 200 ms (10 bins) *before* the velocity window. `07` uses this alignment when
# decoding, so keep it in mind.

# %%
data_splits_path = './external/xfads/examples/monkey_reaching/data' if not _in_colab else 'latent_dynamics_workshop/external/xfads/examples/monkey_reaching/data'

train_data = torch.load(data_splits_path + f'/data_train_{cfg.bin_sz_ms}ms.pt')
valid_data = torch.load(data_splits_path + f'/data_valid_{cfg.bin_sz_ms}ms.pt')
test_data = torch.load(data_splits_path + f'/data_test_{cfg.bin_sz_ms}ms.pt')

train_data.keys()

# %%
y_train_obs = train_data['y_obs'].type(torch.float32).to(cfg.data_device)
y_valid_obs = valid_data['y_obs'].type(torch.float32).to(cfg.data_device)
y_test_obs = test_data['y_obs'].type(torch.float32).to(cfg.data_device)

vel_train = train_data['velocity'].type(torch.float32).to(cfg.data_device)
vel_valid = valid_data['velocity'].type(torch.float32).to(cfg.data_device)
vel_test = test_data['velocity'].type(torch.float32).to(cfg.data_device)

print(y_train_obs.shape)  # trials x time bins x neurons
print(vel_valid.shape)    # trials x time bins x (vx, vy)
print(vel_test.shape)


# %% [markdown]
# ## The object, before the model
#
# Show what we are modeling first: single-trial hand reaches (colored by reach angle)
# and, for a few trials, the spike raster next to the hand velocity.

# %%
def plot_single_reaches(reaches, n_trials_to_plot):
    """Plumbing: integrate velocity to hand position and color by reach angle."""
    trial_plt_dx = torch.randperm(reaches.shape[0])[:n_trials_to_plot]

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle('hand reaches')
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis('off')

    for n in trial_plt_dx:
        traj = torch.cumsum(reaches[n], dim=0)
        reach_angle = torch.atan2(traj[-1, 0], traj[-1, 1])
        reach_color = plt.cm.hsv(reach_angle / (2 * np.pi) + 0.5)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, alpha=0.8, color=reach_color)


plot_single_reaches(vel_train.cpu(), n_trials_to_plot=120)

# %%
move_onset_bin = 12

# at t = bin_prd_start we start forecasting (rolling latents forward, no data)
bin_prd_start = 10

_, n_bins, n_neurons_obs = y_train_obs.shape
n_bins_prd = n_bins - bin_prd_start

n_bins_enc = train_data['n_time_bins_enc']


# %%
def plot_spikes_and_behavior(spikes, velocity, binsize, trials_inds, event_bin):
    """Plumbing: raster (top) and hand velocity (bottom) for a few trials."""
    n_trials = len(trials_inds)
    fig, axes = plt.subplots(nrows=2, ncols=n_trials, figsize=(4 * n_trials, 6), sharex=False, sharey='row')
    if n_trials == 1:
        axes = axes.reshape(2, 1)

    for col, trial_idx in enumerate(trials_inds):
        trial = spikes[trial_idx]
        reach = velocity[trial_idx]
        ax_spikes = axes[0, col]
        ax_vel = axes[1, col]

        for neuron_idx in range(trial.shape[-1]):
            spike_times = np.where(trial[:, neuron_idx].cpu() == 1)[0]
            ax_spikes.scatter(spike_times, [neuron_idx] * len(spike_times), s=4, color='gray', marker='|')

        ax_spikes.axvline(x=event_bin, linestyle='--', color='purple', alpha=0.4)
        ax_spikes.set_ylabel('neurons')
        ax_spikes.set_title(f'Trial {trial_idx}\n# spikes: {int(torch.sum(trial))}', fontsize=10)
        ax_spikes.set_xlabel('time bins')

        time_axis = torch.arange(reach.shape[0]) * binsize
        ax_vel.plot(time_axis, reach[:, 0], color='navy', label='vel x')
        ax_vel.plot(time_axis, reach[:, 1], color='coral', label='vel y')
        ax_vel.axvline(x=event_bin * binsize, linestyle='--', color='purple', alpha=0.4)
        ax_vel.set_xlabel('time (ms)')
        ax_vel.set_title('hand velocity', fontsize=10)
        ax_vel.legend(fontsize=8)

        if col == 0:
            _, y_top = ax_spikes.get_ylim()
            ax_spikes.annotate("movement\nonset", xy=(event_bin, y_top), xytext=(event_bin - 10, y_top + 3),
                               arrowprops=dict(facecolor='black', alpha=0.4, arrowstyle='->'),
                               fontsize=7, ha='center', alpha=0.8)

    fig.tight_layout()
    plt.show()


plot_spikes_and_behavior(y_train_obs.cpu(), vel_train.cpu(), cfg.bin_sz_ms,
                         torch.randperm(y_train_obs.size(0))[:4],
                         event_bin=move_onset_bin)

# %%
"""prepare data for torch"""
y_train_dataset = torch.utils.data.TensorDataset(y_train_obs, vel_train)
y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, vel_valid)
y_test_dataset = torch.utils.data.TensorDataset(y_test_obs, vel_test)

train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(y_val_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)
test_dataloader = torch.utils.data.DataLoader(y_test_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)

# %% [markdown]
# # Building the state-space model
#
# A state-space model needs two pieces: a **dynamics model** for how the latent `z`
# evolves, and an **observation (likelihood) model** for how `z` generates spikes.
# XFADS organizes both, plus the inference network, as swappable modules. Here:
# a GRU nonlinear dynamics, a Poisson likelihood (spikes), and low-rank Gaussian
# local/backward encoders for amortized inference.
#
# For the full construction see the paper's Methods and the lecture notes on
# *amortized inference / VAE* (`sec:amortized`) and *XFADS* (`sec:xfads`).
#
# <p align="center">
#   <img src="https://github.com/catniplab/latent_dynamics_workshop/blob/main/img/ssm_diagram.png?raw=1" width=1000/>
# </p>

# %%
import torch.nn as nn

import xfads.utils as utils
import xfads.prob_utils as prob_utils
import xfads.plot_utils as plot_utils

from xfads.smoothers.nonlinear_smoother_causal import LowRankNonlinearStateSpaceModel, NonlinearFilter
from xfads.ssm_modules.dynamics import DenseGaussianDynamics, DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.ssm_modules.likelihoods import PoissonLikelihood

from xfads.smoothers.lightning_trainers import LightningMonkeyReaching

# %%
if cfg.device == 'cuda':
    torch.cuda.empty_cache()

"""likelihood module (Poisson spikes; see sec:expfam)"""
H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))
likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

"""dynamics module"""
Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

"""initial condition"""
m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
Q_0_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

"""local/backward encoder"""
backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                        rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                        device=cfg.device)
local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_obs, cfg.n_hidden_local, cfg.n_latents, rank=cfg.rank_local,
                                  device=cfg.device, dropout=cfg.p_local_dropout)

"""nonlinear filter"""
nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

"""sequential vae"""
ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                      local_encoder, nl_filter, device=cfg.device)

seq_vae = LightningMonkeyReaching(ssm, cfg, n_bins_enc, bin_prd_start)
seq_vae.ssm.eval()

# %% [markdown]
# # Load the pretrained checkpoint
#
# Training XFADS on CPU is slow, so we ship a pretrained checkpoint and load it.
# Set `train_from_scratch = True` only if you want to (slowly) retrain; the default
# path just loads the model that produced all figures below.

# %%
train_from_scratch = False  # set True to train from scratch (slow; pretrained ckpt below)

if cfg.device == 'cuda':
    torch.cuda.empty_cache()

if _in_colab:
    log_path = 'latent_dynamics_workshop/logs/mc_maze'
    ckpts_path = 'latent_dynamics_workshop/ckpts/mc_maze'
else:
    log_path = './logs/mc_maze'
    ckpts_path = './ckpts/mc_maze'

if train_from_scratch:
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint

    csv_logger = CSVLogger(log_path,
                           name=f'sd_{cfg.seed}_r_y_{cfg.rank_local}_r_b_{cfg.rank_backward}_mask_{cfg.p_mask_a}',
                           version='smoother_causal')
    ckpt_callback = ModelCheckpoint(save_top_k=3, monitor='r2_valid_enc', mode='max',
                                    dirpath=f'{ckpts_path}/', save_last=True,
                                    filename='{epoch:0}_{valid_loss:0.2f}_{r2_valid_enc:0.2f}_{r2_valid_prd:0.2f}_{valid_bps_enc:0.2f}')
    trainer_kwargs = dict(
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.0,
        default_root_dir="lightning/",
        callbacks=[RichProgressBar(), ckpt_callback],
        logger=csv_logger,
        enable_progress_bar=True,
    )

    if cfg.device == 'cuda':
        trainer_kwargs.update(accelerator="gpu", devices=1)
    if cfg.device == 'cpu':
        trainer_kwargs.update(accelerator="cpu")

    trainer = lightning.Trainer(**trainer_kwargs)

    seq_vae.train()
    trainer.fit(model=seq_vae, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    best_model_path_saved = ckpt_callback.best_model_path
    torch.save(best_model_path_saved, f'{ckpts_path}/best_model_path.pt')

else:
    """loading the trained model"""
    best_model_path = f'{ckpts_path}/epoch=827_valid_loss=1415.56_r2_valid_enc=0.89_r2_valid_bhv=0.00_valid_bps_enc=0.42.ckpt'
    seq_vae = LightningMonkeyReaching.load_from_checkpoint(best_model_path, ssm=ssm, cfg=cfg,
                                                           n_time_bins_enc=n_bins_enc, n_time_bins_bhv=bin_prd_start,
                                                           strict=False)
    seq_vae.ssm = seq_vae.ssm.to(cfg.device)

if cfg.device == 'cuda':
    torch.cuda.empty_cache()

# %% [markdown]
# # Inference network: one encoder, three modes
#
# Inspired by conjugate Bayesian inference, XFADS writes the natural parameters of
# the approximate posterior as a **sum** of a prior term and a data-dependent term:
#
# $$\lambda_{\phi}(z_{t-1},\, y_{t:T}) = \lambda_{\theta}(z_{t-1}) + \tilde{\lambda}_{\theta}(y_{t:T}).$$
#
# The data-dependent term $\tilde{\lambda}_{\theta}$ (the "pseudo-observation") can be
# set to zero for missing observations, which is exactly what lets us **forecast**.
# The encoder splits into a **local** encoder (current observation) and a **backward**
# encoder (future observations). The derivation lives in the lecture notes:
# *variational inference* (`sec:vi`) and *amortized inference / VAE* (`sec:amortized`).
#
# With the trained model we can now run three inference modes over the same encoder:
# **smoothing** (use all time), **filtering** (use the past only), and **forecasting**
# (roll the dynamics forward with no data). See *RTS smoothing and forecasting*
# (`sec:smoothing`).

# %% [markdown]
# ## Smoothing: $q(z_t \mid y_{1:T})$ uses the whole trial
#
# <p align="center">
#   <img src="https://github.com/catniplab/latent_dynamics_workshop/blob/main/img/smoothing.png?raw=1" width=600/>
# </p>

# %%
with torch.no_grad():
    loss, z_s_train, stats = seq_vae.ssm(y_train_obs, cfg.n_samples)
    loss, z_s_valid, stats = seq_vae.ssm(y_valid_obs, cfg.n_samples)
    loss, z_s_test, stats = seq_vae.ssm(y_test_obs, cfg.n_samples)

# %% [markdown]
# ## Filtering: $q(z_t \mid y_{1:t})$ uses only the past
#
# <p align="center">
#   <img src="https://github.com/catniplab/latent_dynamics_workshop/blob/main/img/filtering.png?raw=1" width=600/>
# </p>

# %%
with torch.no_grad():
    loss, z_f_train, stats = seq_vae.ssm.forward_filter(y_train_obs, cfg.n_samples)
    loss, z_f_valid, stats = seq_vae.ssm.forward_filter(y_valid_obs, cfg.n_samples)
    loss, z_f_test, stats = seq_vae.ssm.forward_filter(y_test_obs, cfg.n_samples)

# %% [markdown]
# ## Forecasting: roll the dynamics forward with no data
#
# `predict_forward()` starts from the filtered latent at `bin_prd_start` and rolls the
# learned dynamics forward for `n_bins_prd` bins. Each step adds process noise:
#
# $$z_t = \mu_\theta(z_{t-1}) + Q^{1/2}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I),$$
#
# where $\mu_\theta$ is the learned transition (`dynamics_mod.mean_fn`) and $Q^{1/2}$
# is the per-dimension process-noise std. This is an open-loop generative rollout: no
# data enters after `bin_prd_start`. The nonlinear generalization of the Kalman predict
# step is derived in the notes, *RTS smoothing and forecasting* (`sec:smoothing`).

# %%
cat_f_p = lambda f, p: torch.cat([f, p], dim=2)  # stitch filtered prefix + forecast

with torch.no_grad():
    z_p_train = cat_f_p(z_f_train[:, :, :bin_prd_start], seq_vae.ssm.predict_forward(z_f_train[:, :, bin_prd_start], n_bins_prd))
    z_p_valid = cat_f_p(z_f_valid[:, :, :bin_prd_start], seq_vae.ssm.predict_forward(z_f_valid[:, :, bin_prd_start], n_bins_prd))
    z_p_test = cat_f_p(z_f_test[:, :, :bin_prd_start], seq_vae.ssm.predict_forward(z_f_test[:, :, bin_prd_start], n_bins_prd))

# %% [markdown]
# ## Visualize the three modes on the same latent dimensions
#
# For four test trials, plot the first three latent dimensions under smoothing,
# filtering, and forecasting. The vertical line marks where forecasting takes over.

# %%
# plt.get_cmap replaces the removed matplotlib.cm.get_cmap (matplotlib >= 3.9)
blues = plt.get_cmap("winter", z_s_test.shape[0])
reds = plt.get_cmap("summer", z_s_test.shape[0])
springs = plt.get_cmap("spring", z_s_test.shape[0])

trial_list = [28, 202, 8, 285]
color_map_list = [blues, reds, springs]

# %%
"""smoothed latent states"""
with torch.no_grad():
    fig, axs = plt.subplots(len(trial_list), 1, figsize=(4, 4))
    fig.suptitle('smoothed\n')
    plot_utils.plot_z_samples(fig, axs, z_s_test[:, trial_list, ..., :3].cpu(), color_map_list)
    axs[0].lines[-1].set_label('movement\nonset')
    axs[0].legend(bbox_to_anchor=(0.125, 0.96), fontsize=8, frameon=False)
    plt.show()

# %%
"""filtered latent states"""
with torch.no_grad():
    fig, axs = plt.subplots(len(trial_list), 1, figsize=(4, 4))
    fig.suptitle('filtered')
    plot_utils.plot_z_samples(fig, axs, z_f_test[:, trial_list, ..., :3].cpu(), color_map_list)
    plt.show()

# %%
"""forecasted latent states"""
with torch.no_grad():
    fig, axs = plt.subplots(len(trial_list), 1, figsize=(4, 4))
    fig.suptitle('forecasted')
    [axs[i].axvline(bin_prd_start, linestyle='--', color='red') for i in range(len(trial_list))]
    plot_utils.plot_z_samples(fig, axs, z_p_test[:, trial_list, ..., :3].cpu(), color_map_list)
    _, y_upper_limit = axs[0].get_ylim()
    axs[0].annotate("prediction\nstarts",
                    xy=(bin_prd_start, y_upper_limit),
                    xytext=(bin_prd_start - (n_bins * 0.1), (y_upper_limit * 1.2)),
                    arrowprops=dict(facecolor='black', alpha=0.4, arrowstyle='->'),
                    fontsize=7, alpha=0.8, ha='center')
    plt.show()

# %% [markdown]
# ### Exercise (predict, then check)
#
# Before comparing the panels above: for a test trial, in the window **after**
# `bin_prd_start`, should the *forecasted* latent trace hug the *filtered* trace or
# drift away from it? Why?
#
# <details>
# <summary>Solution</summary>
#
# It drifts. After `bin_prd_start` the forecast uses **no data** - it only rolls the
# dynamics forward and injects process noise, so trials that share an initial state
# fan out. The filtered trace keeps ingesting spikes, so it stays pinned to the
# actual trial. They agree only up to `bin_prd_start`, where the forecast is seeded
# from the filtered state.
#
# </details>
#
# > **Stretch (optional):** change `bin_prd_start` (try 5 and 20), recompute
# > `n_bins_prd = n_bins - bin_prd_start`, and re-run the forecasting and latent-plot
# > cells. How does observing less of the trial change how quickly the forecast
# > diverges? (`07` quantifies this with a k-step R2 curve.)
#
# <details>
# <summary>Solution</summary>
#
# ```python
# bin_prd_start = 5
# n_bins_prd = n_bins - bin_prd_start
# # re-run the predict_forward cell and the forecasted-latent plot
# ```
# Smaller `bin_prd_start` means a longer open-loop rollout, so the forecast diverges
# from the data sooner and over more bins; larger `bin_prd_start` gives the filter
# more evidence and a shorter, tighter forecast.
#
# </details>

# %% [markdown]
# # Reconstruct firing rates from the latents
#
# The Poisson readout turns any latent sample into an expected spike count per bin:
#
# $$\hat r_t = \Delta \cdot \exp\big(\text{readout}(z_t)\big),$$
#
# averaged over the `n_samples` posterior samples. We build reconstructed rates from
# smoothed, filtered, and forecasted latents. See *Poisson observations* (`sec:expfam`).

# %%
rates_train_s = (cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z_s_train)).mean(dim=0)).cpu().detach().numpy()
rates_test_s = (cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z_s_test)).mean(dim=0)).cpu().detach().numpy()
rates_test_f = (cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z_f_test)).mean(dim=0)).cpu().detach().numpy()
rates_test_p = (cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z_p_test)).mean(dim=0)).cpu().detach().numpy()

# %% [markdown]
# ## Trial-averaged single-neuron rates: smoothing vs forecasting vs data
#
# For a grid of neurons, overlay the observed trial-averaged rate (black) with the
# smoothed reconstruction (green) and the forecast (coral). The forecast tracks the
# data before `bin_prd_start` and then generalizes from dynamics alone.

# %%
n_neurons_to_plot = 16
neuron_indcs = np.random.choice(range(0, y_test_obs.shape[2]), size=n_neurons_to_plot, replace=False)

fig, axes = plt.subplots(int(np.sqrt(n_neurons_to_plot)), int(np.sqrt(n_neurons_to_plot)), figsize=(14, 10))
fig.suptitle('Trial-averaged neuron activity\n\n\n\n')

for ax, neuron in zip(axes.flat, neuron_indcs):

    fr_data = torch.mean(y_test_obs[:, :, neuron], axis=0).cpu()
    fr_model_s = torch.mean(torch.from_numpy(rates_test_s[:, :, neuron]), axis=0)
    fr_model_p = torch.mean(torch.from_numpy(rates_test_p[:, :, neuron]), axis=0)

    ax.plot(np.arange(n_bins) * cfg.bin_sz_ms, fr_data, color='black', alpha=0.8, label='true' if neuron == neuron_indcs[-1] else '')
    ax.plot(np.arange(n_bins) * cfg.bin_sz_ms, fr_model_s, color='green', alpha=1.0, label='smoothed' if neuron == neuron_indcs[-1] else '')
    ax.plot(np.arange(n_bins) * cfg.bin_sz_ms, fr_model_p, color='coral', alpha=1.0, label='forecasted' if neuron == neuron_indcs[-1] else '')

    ax.axvline(bin_prd_start * cfg.bin_sz_ms, linestyle='--', color='coral')
    ax.axvline(move_onset_bin * cfg.bin_sz_ms, linestyle='--', color='gray')

    if neuron == neuron_indcs[0]:
        _, y_upper_limit = ax.get_ylim()
        ax.annotate("prediction\nstarts",
                    xy=(bin_prd_start * cfg.bin_sz_ms, y_upper_limit),
                    xytext=(bin_prd_start * cfg.bin_sz_ms - (n_bins * 0.3), (y_upper_limit * 1.2)),
                    arrowprops=dict(facecolor='black', alpha=0.2, arrowstyle='->'),
                    fontsize=7, alpha=0.8, ha='center')

    ax.set_title(f'\nneuron {neuron+1}\n', fontsize=8)
    ax.set_xlabel('time (ms)' if neuron == neuron_indcs[-int(np.sqrt(n_neurons_to_plot))] else '', fontsize=9)
    ax.set_ylabel('firing rate' if neuron == neuron_indcs[0] else '', fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=1, fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# # You can now...
#
# ...take a trained XFADS model and, from population spikes, infer a latent `z` under
# **smoothing**, **filtering**, and **forecasting**, then read it back out as
# single-neuron firing rates.
#
# **Transfer prompt:** point this at *your* recording. Swap in your binned spike
# tensor (trials x time x neurons), set `n_neurons_obs`, pick `bin_prd_start`, and ask:
# where does the forecast stop tracking the data, and which neurons does the model
# reconstruct well vs poorly?
#
# **Next:** open **`07_decoding_and_evaluation`** to score these latents and rates -
# ridge decoding of hand velocity, k-step forecast R2, a PCA-vs-R2 sweep, and
# predictive log-likelihood.
