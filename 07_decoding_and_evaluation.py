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
# # Decoding and evaluation: how good are the XFADS latents?
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/07_decoding_and_evaluation.ipynb)
#
# **Takeaway:** score the latents and rates from `06` four ways - ridge decoding of
# hand velocity, k-step forecast R2, a PCA-dimension sweep, and predictive
# log-likelihood - to see how filtering, smoothing, and forecasting compare.
#
# This is the **optional companion** to **`06_xfads_mc_maze`**. It reuses the same
# pretrained model. The first section repeats `06`'s setup so this notebook runs on
# its own; skim it and move to the evaluation sections.
#
# Background in the lecture notes: *Poisson observations* (`sec:expfam`),
# *RTS smoothing and forecasting* (`sec:smoothing`), and *XFADS* (`sec:xfads`).

# %% [markdown]
# ## Setup (repeats `06` - run and move on)
#
# Detect Colab, install XFADS, rebuild the model, load the checkpoint, and recompute
# the smoothed / filtered / forecasted latents and reconstructed rates.

# %%
try:
    import google.colab
    _in_colab = True
except ImportError:
    _in_colab = False

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

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import pytorch_lightning as lightning

import warnings
warnings.filterwarnings("ignore")

import xfads.utils as utils
import xfads.plot_utils as plot_utils

from xfads.smoothers.nonlinear_smoother_causal import LowRankNonlinearStateSpaceModel, NonlinearFilter
from xfads.ssm_modules.dynamics import DenseGaussianDynamics, DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.ssm_modules.likelihoods import PoissonLikelihood
from xfads.smoothers.lightning_trainers import LightningMonkeyReaching

# %%
"""config (must match the checkpoint from 06)"""
cfg = {
    'n_latents': 40, 'n_latents_read': 35,
    'rank_local': 15, 'rank_backward': 5, 'n_hidden_dynamics': 128,
    'n_samples': 25, 'n_hidden_local': 256, 'n_hidden_backward': 128,
    'use_cd': False, 'p_mask_a': 0.0, 'p_mask_b': 0.0, 'p_mask_apb': 0.0,
    'p_mask_y_in': 0.0, 'p_local_dropout': 0.4, 'p_backward_dropout': 0.0,
    'device': 'cpu', 'data_device': 'cpu',
    'lr': 1e-3, 'lr_gamma_decay': 0.997, 'n_epochs': 3, 'batch_sz': 128,
    'bin_sz': 20e-3, 'bin_sz_ms': 20, 'seed': 1234, 'default_dtype': torch.float32,
}

class Cfg(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'Cfg' object has no attribute '{attr}'")

cfg = Cfg(cfg)
if not torch.cuda.is_available():
    cfg.device = 'cpu'
    cfg.data_device = 'cpu'
lightning.seed_everything(cfg.seed, workers=True)
torch.set_default_dtype(torch.float32)

# %%
"""load data"""
data_splits_path = './external/xfads/examples/monkey_reaching/data' if not _in_colab else 'latent_dynamics_workshop/external/xfads/examples/monkey_reaching/data'
train_data = torch.load(data_splits_path + f'/data_train_{cfg.bin_sz_ms}ms.pt')
test_data = torch.load(data_splits_path + f'/data_test_{cfg.bin_sz_ms}ms.pt')

y_train_obs = train_data['y_obs'].type(torch.float32).to(cfg.data_device)
y_test_obs = test_data['y_obs'].type(torch.float32).to(cfg.data_device)
vel_train = train_data['velocity'].type(torch.float32).to(cfg.data_device)
vel_test = test_data['velocity'].type(torch.float32).to(cfg.data_device)

move_onset_bin = 12
bin_prd_start = 10
_, n_bins, n_neurons_obs = y_train_obs.shape
n_bins_prd = n_bins - bin_prd_start
n_bins_enc = train_data['n_time_bins_enc']

# %%
"""rebuild the SSM and load the checkpoint"""
H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))
likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
Q_0_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                        rank_local=cfg.rank_local, rank_backward=cfg.rank_backward, device=cfg.device)
local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_obs, cfg.n_hidden_local, cfg.n_latents,
                                  rank=cfg.rank_local, device=cfg.device, dropout=cfg.p_local_dropout)
nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)
ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                      local_encoder, nl_filter, device=cfg.device)

ckpts_path = 'latent_dynamics_workshop/ckpts/mc_maze' if _in_colab else './ckpts/mc_maze'
best_model_path = f'{ckpts_path}/epoch=827_valid_loss=1415.56_r2_valid_enc=0.89_r2_valid_bhv=0.00_valid_bps_enc=0.42.ckpt'
seq_vae = LightningMonkeyReaching.load_from_checkpoint(best_model_path, ssm=ssm, cfg=cfg,
                                                       n_time_bins_enc=n_bins_enc, n_time_bins_bhv=bin_prd_start,
                                                       strict=False)
seq_vae.ssm = seq_vae.ssm.to(cfg.device)
seq_vae.ssm.eval()

# %%
"""recompute latents (smoothed z_s, filtered z_f, forecasted z_p) and rates"""
with torch.no_grad():
    _, z_s_train, _ = seq_vae.ssm(y_train_obs, cfg.n_samples)
    _, z_s_test, _ = seq_vae.ssm(y_test_obs, cfg.n_samples)
    _, z_f_test, _ = seq_vae.ssm.forward_filter(y_test_obs, cfg.n_samples)

    z_p_test = torch.cat([z_f_test[:, :, :bin_prd_start],
                          seq_vae.ssm.predict_forward(z_f_test[:, :, bin_prd_start], n_bins_prd)], dim=2)

rate_of = lambda z: (cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z)).mean(dim=0)).cpu().detach().numpy()
rates_train_s = rate_of(z_s_train)
rates_test_s = rate_of(z_s_test)
rates_test_f = rate_of(z_f_test)
rates_test_p = rate_of(z_p_test)

# %% [markdown]
# # 1. Decode hand velocity with ridge regression
#
# A linear (ridge) decoder maps reconstructed rates to hand velocity. We fit on
# smoothed **training** rates, then score the smoothed / filtered / forecasted
# **test** rates.
#
# **Alignment note:** the neural window (45 bins, starting -240 ms) leads the velocity
# window (35 bins, starting -40 ms) by 200 ms. We deliberately keep that 200 ms neural
# lead: slicing rates to `[:, :n_bins_enc, :]` (the first 35 neural bins) pairs neural
# bin *i* with velocity bin *i*, so the decoder reads neural activity that *precedes*
# the movement it predicts. This is physiologically sensible (motor cortex leads the
# hand), not a bug.

# %%
with torch.no_grad():
    clf = Ridge(alpha=0.01)
    clf.fit(rates_train_s[:, :n_bins_enc, :].reshape(-1, n_neurons_obs), vel_train.cpu().reshape(-1, 2))

    pred_reshape = lambda rates, clf, original_shape: clf.predict(rates[:, :n_bins_enc, :].reshape(-1, n_neurons_obs)).reshape(list(original_shape)[:-1] + [2])
    calc_r2 = lambda rates, clf, true_velocity: clf.score(rates[:, :n_bins_enc, :].reshape(-1, n_neurons_obs), true_velocity.cpu().reshape(-1, 2))

    r2_test_s = calc_r2(rates_test_s, clf, vel_test)
    r2_test_f = calc_r2(rates_test_f, clf, vel_test)
    r2_test_p = calc_r2(rates_test_p, clf, vel_test)

    vel_hat_test_s = pred_reshape(rates_test_s, clf, vel_test.shape)
    vel_hat_test_f = pred_reshape(rates_test_f, clf, vel_test.shape)
    vel_hat_test_p = pred_reshape(rates_test_p, clf, vel_test.shape)

# %%
"""integrate decoded velocity into hand position and plot true vs decoded reaches"""
n_trials_test = vel_test.shape[0]
n_trials_plot = 35

vel_to_pos = lambda v: torch.cumsum(torch.tensor(v).clone().detach().to('cpu'), dim=1)

pos_test = vel_to_pos(vel_test.cpu())
trial_plt_dx = torch.randperm(n_trials_test)[:n_trials_plot]
reach_angle = torch.atan2(pos_test[:, -1, 0], pos_test[:, -1, 1])
reach_colors = plt.cm.hsv(reach_angle / (2 * np.pi) + 0.5)

pos_test_hat_s = vel_to_pos(vel_hat_test_s)
pos_test_hat_f = vel_to_pos(vel_hat_test_f)
pos_test_hat_p = vel_to_pos(vel_hat_test_p)

with torch.no_grad():
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    plot_utils.plot_reaching(axs[0], pos_test[trial_plt_dx], reach_colors[trial_plt_dx])
    plot_utils.plot_reaching(axs[1], pos_test_hat_s[trial_plt_dx], reach_colors[trial_plt_dx])
    plot_utils.plot_reaching(axs[2], pos_test_hat_f[trial_plt_dx], reach_colors[trial_plt_dx])
    plot_utils.plot_reaching(axs[3], pos_test_hat_p[trial_plt_dx], reach_colors[trial_plt_dx])
    axs[0].set_title('true')
    axs[1].set_title(f'smoothed, r2:{r2_test_s:.3f}')
    axs[2].set_title(f'filtered, r2:{r2_test_f:.3f}')
    axs[3].set_title(f'predicted, r2:{r2_test_p:.3f}')
    plt.show()

# %% [markdown]
# ### Exercise (tweak and observe)
#
# > **Stretch (optional):** sweep the ridge `alpha` over a few decades (e.g. `1e-3`,
# > `1e-1`, `1e1`) and/or fit the decoder on the **smoothed latents** `z_s` (mean over
# > samples) instead of the rates. Which representation and regularization decode hand
# > velocity best on the test set?
#
# <details>
# <summary>Solution</summary>
#
# ```python
# for a in [1e-3, 1e-1, 1e1]:
#     c = Ridge(alpha=a).fit(rates_train_s[:, :n_bins_enc, :].reshape(-1, n_neurons_obs),
#                            vel_train.cpu().reshape(-1, 2))
#     print(a, calc_r2(rates_test_s, c, vel_test))
#
# # decode from smoothed latents instead of rates
# z_tr = z_s_train.mean(0)[:, :n_bins_enc, :].reshape(-1, cfg.n_latents).cpu().numpy()
# z_te = z_s_test.mean(0)[:, :n_bins_enc, :].reshape(-1, cfg.n_latents).cpu().numpy()
# c = Ridge(alpha=0.01).fit(z_tr, vel_train.cpu().reshape(-1, 2))
# print('latents', c.score(z_te, vel_test.cpu().reshape(-1, 2)))
# ```
# Rates and latents are affine-related through the readout, so they decode
# comparably; very small `alpha` can overfit the 182-neuron rate features, very large
# `alpha` underfits. The best test R2 sits at a moderate `alpha`.
#
# </details>

# %%
def plot_spikes_and_decoded_behavior(spikes, velocity, velocity_hat, binsize, trials_inds, event_bin):
    """Plumbing: raster (top) and true vs decoded hand velocity (bottom)."""
    n_trials = len(trials_inds)
    fig, axes = plt.subplots(nrows=2, ncols=n_trials, figsize=(4 * n_trials, 6), sharex=False, sharey='row')
    if n_trials == 1:
        axes = axes.reshape(2, 1)

    for col, trial_idx in enumerate(trials_inds):
        trial = spikes[trial_idx]
        reach = velocity[trial_idx]
        decoded_reach = velocity_hat[trial_idx]
        ax_spikes = axes[0, col]
        ax_vel = axes[1, col]

        for neuron_idx in range(trial.shape[-1]):
            spike_times = np.where(trial[:, neuron_idx].cpu() == 1)[0]
            ax_spikes.scatter(spike_times, [neuron_idx] * len(spike_times), s=4, color='gray', marker='|')

        ax_spikes.axvline(x=event_bin, linestyle='--', color='purple', alpha=0.4)
        ax_spikes.set_ylabel('neurons')
        ax_spikes.set_title(f'\nTrial {trial_idx}\n# spikes: {int(torch.sum(trial))}', fontsize=10)
        ax_spikes.set_xlabel('time bins')

        time_axis = torch.arange(reach.shape[0]) * binsize
        ax_vel.plot(time_axis, reach[:, 0], color='navy', linewidth=1.0, label='true vel x' if col == 0 else '')
        ax_vel.plot(time_axis, reach[:, 1], color='coral', linewidth=1.0, label='true vel y' if col == 0 else '')
        ax_vel.plot(time_axis, decoded_reach[:, 0], linestyle='--', linewidth=1.0, color='navy', label='decoded vel x' if col == 0 else '')
        ax_vel.plot(time_axis, decoded_reach[:, 1], linestyle='--', linewidth=1.0, color='coral', label='decoded vel y' if col == 0 else '')
        ax_vel.axvline(x=event_bin * binsize, linestyle='--', linewidth=1.0, color='purple', alpha=0.4)
        ax_vel.set_xlabel('time (ms)')
        ax_vel.set_title('\nhand velocity', fontsize=10)

        if col == 0:
            _, y_top = ax_spikes.get_ylim()
            ax_spikes.annotate("movement\nonset", xy=(event_bin, y_top), xytext=(event_bin - 10, y_top + 3),
                               arrowprops=dict(facecolor='black', alpha=0.4, arrowstyle='->'),
                               fontsize=7, ha='center', alpha=0.8)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    plt.show()


plot_spikes_and_decoded_behavior(y_test_obs.cpu(), vel_test.cpu(), vel_hat_test_s, cfg.bin_sz_ms,
                                 torch.randperm(y_test_obs.size(0))[:4], event_bin=move_onset_bin)

# %% [markdown]
# # 2. k-step forecast: how far ahead can we decode?
#
# Start the forecast at each bin `k`, roll the latents forward to the end of the
# trial, reconstruct rates, and decode velocity. The resulting R2-vs-`k` curve shows
# how much the model needs to observe before its forecast of future hand velocity is
# useful. Filtering and smoothing R2 (which use data everywhere) are horizontal
# references.

# %%
with torch.no_grad():
    r2_k_step = []
    for k in range(n_bins):
        z_prd_test = utils.propagate_latent_k_steps(z_f_test[:, :, k], dynamics_mod, n_bins - (k + 1))
        z_prd_test = torch.concat([z_f_test[:, :, :k], z_prd_test], dim=2)

        rates_prd_test = cfg.bin_sz * torch.exp(seq_vae.ssm.likelihood_pdf.readout_fn(z_prd_test)).mean(dim=0).cpu().detach().numpy()
        r2_k_step.append(calc_r2(rates_prd_test, clf, vel_test))

fig, ax = plt.subplots()
ax.plot(r2_k_step)
ax.axhline(r2_test_s, color='green', label='smoothed')
ax.axhline(r2_test_f, color='orange', label='filtered')
ax.axvline(move_onset_bin, linestyle='--')
y_upper_limit = ax.get_ylim()[1]  # define locally (do not reuse a variable from another cell)
ax.annotate("movement\nonset",
            xy=(move_onset_bin, y_upper_limit),
            xytext=(move_onset_bin + (n_bins * 0.15), (y_upper_limit * 1.2)),
            arrowprops=dict(facecolor='black', alpha=0.4, arrowstyle='->'),
            fontsize=7, alpha=0.8, ha='center')
ax.set_xlabel('forecast start bin (k)')
ax.set_ylabel('r2')
ax.legend()
plt.show()

# %% [markdown]
# ### Exercise (tweak and observe)
#
# Read the k-step curve: at roughly which bin `k` does the forecast R2 climb to meet
# the filtered/smoothed references, and what does that say about how much of a reach
# the model must see before it can forecast the rest?
#
# <details>
# <summary>Solution</summary>
#
# The curve rises steeply around movement onset (`k` near `move_onset_bin = 12`): once
# the model has filtered through the onset of the reach, a forecast from there captures
# almost all the decodable velocity, so R2 approaches the filtered/smoothed lines. Very
# early forecasts (small `k`, before onset) have little evidence about which reach this
# is, so their R2 is low.
#
# </details>

# %% [markdown]
# # 3. PCA sweep: how many latent dimensions matter for decoding?
#
# Compress the smoothed encoding-window latents to `k` principal components, decode
# velocity from those `k` PCs, and plot test R2 vs `k`. The second panel ranks the
# latent dimensions by their decoding weight.

# %%
def pca_vs_r2(z_train, z_test, vel_train, vel_test, max_pcs=40, alpha=0.01):
    flatten = lambda x: x.reshape(-1, x.shape[2]).detach().cpu().numpy()
    flatten_vel = lambda v: v.reshape(-1, 2).detach().cpu().numpy()

    X_train, X_test = flatten(z_train), flatten(z_test)
    y_train, y_test = flatten_vel(vel_train), flatten_vel(vel_test)

    r2_scores = []
    clf = None
    for k in range(1, max_pcs + 1):
        pca = PCA(n_components=k)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        clf = Ridge(alpha=alpha).fit(X_train_pca, y_train)
        r2_scores.append(r2_score(y_test, clf.predict(X_test_pca), multioutput='uniform_average'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, max_pcs + 1), r2_scores, marker='o')
    axes[0].set_xlabel("num PCs used for decoding")
    axes[0].set_ylabel("R2")
    axes[0].set_title("velocity decoding from latents")
    axes[0].grid(True)

    importance = np.abs(clf.coef_).mean(axis=0)  # last fit uses all max_pcs PCs
    axes[1].bar(np.arange(len(importance)), importance[np.argsort(-importance)])
    axes[1].set_xlabel('latent PC (sorted)')
    axes[1].set_ylabel('mean |w| (vx & vy)')
    axes[1].set_title('PC importance for decoding')
    plt.tight_layout()
    plt.show()

    return np.array(r2_scores)


r2_scores = pca_vs_r2(z_s_train[:, :, :n_bins_enc, :].mean(dim=0),
                      z_s_test[:, :, :n_bins_enc, :].mean(dim=0),
                      vel_train, vel_test, max_pcs=40)

# %% [markdown]
# # 4. Predictive log-likelihood
#
# The Poisson log-likelihood of the held-out spikes under each reconstruction tells us,
# per time bin, how well the model *predicts* the data (not just the trial-averaged
# rate). Smoothing (all data) should beat filtering (past only), which should beat the
# forecast after `bin_prd_start`; all should beat a constant-mean-rate baseline.
#
# We drop the constant $-\log(y!)$ normalizer of the Poisson pmf. It does not depend on
# the model, so it cancels when comparing models on the same spikes - the y-axis is
# therefore the per-bin log-likelihood **up to that additive constant**.

# %%
def predictive_log_likelihood(spikes, rates):
    """Poisson log-likelihood up to the constant -log(y!) (drops the normalizer)."""
    eps = 1e-8
    spikes = torch.as_tensor(spikes, dtype=torch.float32)
    rates = torch.as_tensor(rates, dtype=torch.float32)
    log_likelihood = spikes * torch.log(rates + eps) - rates
    return log_likelihood.sum(dim=-1)


I, T, N = y_test_obs.shape

# NOTE: each curve is named for the rates it is computed from (filtered rates -> filtering).
ll_filter = predictive_log_likelihood(y_test_obs.cpu(), rates_test_f).numpy()
mean_filter = ll_filter.mean(axis=0)
sem_filter = ll_filter.std(axis=0) / np.sqrt(I)

ll_smooth = predictive_log_likelihood(y_test_obs.cpu(), rates_test_s).numpy()
mean_smooth = ll_smooth.mean(axis=0)
sem_smooth = ll_smooth.std(axis=0) / np.sqrt(I)

ll_forecast = predictive_log_likelihood(y_test_obs.cpu(), rates_test_p).numpy()
mean_forecast = ll_forecast.mean(axis=0)
sem_forecast = ll_forecast.std(axis=0) / np.sqrt(I)

# constant mean firing rate per neuron as a baseline
mean_rate_per_neuron = y_test_obs.mean(dim=(0, 1))
baseline_rates = mean_rate_per_neuron.unsqueeze(0).unsqueeze(0).expand(I, T, N)
ll_baseline = predictive_log_likelihood(y_test_obs, baseline_rates)
mean_base = ll_baseline.mean(dim=0).cpu().numpy()
sem_base = ll_baseline.std(dim=0).cpu().numpy() / np.sqrt(I)

# %% [markdown]
# ### Exercise (predict, then check)
#
# Before running the plot: in the pre-forecast window (bins before `bin_prd_start`),
# will **smoothing** or **filtering** give the higher predictive log-likelihood? Why?
#
# <details>
# <summary>Solution</summary>
#
# Smoothing. Smoothing conditions on the **entire** trial ($q(z_t\mid y_{1:T})$) while
# filtering conditions only on the past ($q(z_t\mid y_{1:t})$). More conditioning
# information cannot hurt the posterior, so the smoothed rates explain each bin's
# spikes at least as well - the gold (smoothing) curve sits at or above the navy
# (filtering) curve everywhere data is available.
#
# </details>

# %%
time = np.arange(T) * cfg.bin_sz_ms

plt.figure(figsize=(10, 6))
plt.plot(time, mean_filter, label='Filtering', color='navy')
plt.fill_between(time, mean_filter - sem_filter, mean_filter + sem_filter, color='navy', alpha=0.2)

plt.plot(time, mean_smooth, label='Smoothing', color='gold')
plt.fill_between(time, mean_smooth - sem_smooth, mean_smooth + sem_smooth, color='gold', alpha=0.2)

plt.plot(time, mean_forecast, label='Forecasting', color='coral')
plt.fill_between(time, mean_forecast - sem_forecast, mean_forecast + sem_forecast, color='coral', alpha=0.2)

plt.plot(time, mean_base, label='Baseline (neuron mean fr)', color='gray')
plt.fill_between(time, mean_base - sem_base, mean_base + sem_base, color='gray', alpha=0.2, label='± SEM')

plt.axvline(bin_prd_start * cfg.bin_sz_ms, linestyle='--', color='coral', label='prediction starts')
plt.xlabel('time (ms)')
plt.ylabel(r'$\log p(y \mid \hat{y})$ (up to const.)')
plt.title('Predictive log-likelihood\n\n\n\n\n')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize='medium', frameon=False)
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Optional: raster vs generated-rate heatmap
#
# A last sanity view - observed spikes (top) next to the smoothed generated rates
# (bottom) for a few trials.

# %%
def plot_trials(true_rates, generated_rates, n=4, spike_threshold=0.1):
    """Plumbing: observed spike raster (top) vs generated rate heatmap (bottom)."""
    true_rates = true_rates.cpu() if isinstance(true_rates, torch.Tensor) else true_rates
    generated_rates = generated_rates.cpu() if isinstance(generated_rates, torch.Tensor) else generated_rates

    trials = true_rates.shape[0]
    n = min(n, trials)
    random_indices = np.random.choice(trials, size=n, replace=False)

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 8), sharex=True, sharey='row')
    im = None
    for idx, trial_i in enumerate(random_indices):
        spikes = true_rates[trial_i]
        ax_raster = axes[0, idx]
        spike_times, neuron_ids = np.where(spikes > spike_threshold)
        ax_raster.scatter(spike_times, neuron_ids, s=2, color='black')
        if idx == 0:
            ax_raster.set_ylabel("neuron")

        ax_gen = axes[1, idx]
        im = ax_gen.imshow(generated_rates[trial_i].T, aspect='auto', cmap='viridis', origin='lower')
        if idx == 0:
            ax_gen.set_ylabel("neuron")
            ax_gen.set_xlabel("time bins")

    cbar_ax = fig.add_axes([1., 0.12, 0.015, 0.33])
    fig.colorbar(im, cax=cbar_ax, label="firing rate")
    axes[0, 0].set_title("Observed spikes (raster)", fontsize=12)
    axes[1, 0].set_title("Generated firing rates", fontsize=12)
    plt.tight_layout()
    plt.show()


plot_trials(y_test_obs, rates_test_s)

# %% [markdown]
# # You can now...
#
# ...evaluate a latent dynamical model four complementary ways: linear decoding of
# behavior, k-step forecasting quality, dimension-vs-decoding sweeps, and predictive
# log-likelihood against a baseline.
#
# **Transfer prompt:** on your own model, which metric moves first when you change the
# latent dimension or the amount of masking during training? A model can win on
# predictive log-likelihood yet decode behavior poorly (or vice versa) - decide which
# metric matches *your* scientific question before you optimize for it.
