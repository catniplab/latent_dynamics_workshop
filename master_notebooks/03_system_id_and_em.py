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

# %% [markdown]
# # Optional: learning the dynamics (Ho-Kalman subspace ID and EM)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/03_system_id_and_em.ipynb)
#
# In [`02_linear_lvms`](02_linear_lvms.ipynb) we *knew* the system
# $(\mathbf{A},\mathbf{C},\mathbf{Q},\mathbf{R})$; here we *estimate* the linear
# dynamics straight from the observations - non-iteratively with the Ho-Kalman
# subspace method, and probabilistically with one EM M-step.
#
# This is the optional companion to the core notebook 02. Full derivation of the
# Ho-Kalman algorithm is in the lecture notes; we only summarize it here.

# %% [markdown]
# ## Setup (Colab)
# On Colab this clones the repo and installs `xfads`. Locally it is a no-op.

# %%
try:
    import google.colab
    _in_colab = True
except ImportError:
    _in_colab = False

if _in_colab:
    # This notebook only needs the xfads submodule, so we init just that one
    # (not --recurse-submodules, which would also pull nlb_tools/neurofisherSNR).
    # !git clone https://github.com/catniplab/latent_dynamics_workshop.git
    # !cd latent_dynamics_workshop && git submodule update --init external/xfads
    # !pip install -e latent_dynamics_workshop/external/xfads/
    pass

import os
import sys

if _in_colab:
    cwd = os.getcwd()
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop"))
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop/external/xfads"))

import torch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

import xfads.utils as utils
import xfads.prob_utils as prob_utils
from xfads.linalg_utils import bmv
from xfads.prob_utils import (
    kalman_information_filter,
    rts_smoother,
    align_latent_variables,
)

from code_pack.plotting import plot_rotated_latents

# Minimal config: 2 latent dimensions, run on CPU unless a GPU is available.
n_latents = 2
seed = 20270714  # same seed as notebook 02, so the recreated spiral data matches
device = "cuda" if torch.cuda.is_available() else "cpu"

pl.seed_everything(seed, workers=True)
torch.set_default_dtype(torch.float32)

# %% [markdown]
# ## Recreate the spiral data and the true-parameter smoother
# Same generative model as notebook 02. We also run the RTS smoother
# with the *true* parameters, because the EM M-step below needs a posterior to start from.

# %%
n_neurons = 50
n_trials = 1000
n_time_bins = 50
n_samples = 5

omega, rho = 3.14 / 8.0, 0.97
mean_fn = utils.SpiralDynamics(omega, rho)

C = torch.nn.Linear(2, n_neurons, device="cpu").requires_grad_(False)
C.bias.data = torch.zeros_like(C.weight[:, 0])

Q_diag = 3e-2 * torch.ones(2)
Q_0_diag = 1.0 * torch.ones(2)
R_diag = 0.5 + 0.5 * torch.rand(n_neurons)
m_0 = torch.zeros(2)

z = utils.sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins)
y = C(z) + torch.sqrt(R_diag) * torch.randn_like(C(z))

n_valid = n_trials // 3
n_train = n_trials - n_valid
y_valid, z_valid = y[n_train:], z[n_train:]

# True-parameter Kalman filter + RTS smoother (E-step for EM later).
h_update = bmv(C.weight.T, (y_valid - C.bias) / R_diag)
J_update = (C.weight.T / R_diag) @ C.weight
J_update = J_update.expand(y_valid.shape[0], n_time_bins, n_latents, n_latents)

m_f, P_f, m_p, P_p = kalman_information_filter(h_update, J_update, mean_fn.A, Q_diag, m_0, Q_0_diag)
m_s, P_s, P_tp1_t_s, z_s = rts_smoother(m_p, P_p, m_f, P_f, mean_fn.A, n_samples=n_samples)

# %% [markdown]
# ## Ho-Kalman subspace identification
#
# The Ho-Kalman algorithm recovers the system non-iteratively from output statistics.
# In one paragraph:
#
# 1. Estimate output autocovariances $\Gamma_k=\mathbb{E}[\mathbf{y}_{t+k}\mathbf{y}_t^\top]$
#    and stack them into a block **Hankel** matrix $H$.
# 2. Take the rank-$d$ SVD $H\approx U_d\Sigma_d V_d^\top$ and factor into an
#    observability matrix $\mathcal{O}=U_d\Sigma_d^{1/2}$ (its first block is $\mathbf{C}$)
#    and a controllability factor $\mathcal{K}=\Sigma_d^{1/2}V_d^\top$ (its first block
#    column is $\mathbf{G}:=\mathbf{A}\mathbf{P}_\infty\mathbf{C}^\top$).
# 3. Recover $\mathbf{A}$ from **shift-invariance** of $\mathcal{O}$. With
#    $\mathcal{O}_{\text{top}}=\mathcal{O}$ minus its last block row and
#    $\mathcal{O}_{\text{bottom}}=\mathcal{O}$ minus its first block row, the relation is
#    $\mathcal{O}_{\text{bottom}}=\mathcal{O}_{\text{top}}\mathbf{A}$, hence
#
#    $$\mathbf{A}\approx \mathcal{O}_{\text{top}}^{\dagger}\,\mathcal{O}_{\text{bottom}}.$$
#
# 4. Recover $\mathbf{Q}$ and $\mathbf{R}$ from $\mathbf{G}$, the lag-0 covariance
#    $\Gamma_0=\mathbf{C}\mathbf{P}_\infty\mathbf{C}^\top+\mathbf{R}$, and the discrete
#    Lyapunov equation $\mathbf{P}_\infty=\mathbf{A}\mathbf{P}_\infty\mathbf{A}^\top+\mathbf{Q}$.
#
# The helper `get_kalman_ho_estimates` does steps 2-4.

# %%
# Build the block-Hankel matrix and lag-0 covariance, then read off the system.
H_hankel = prob_utils.construct_hankel(y_valid, 50, 50)
Gamma_0_hat = prob_utils.compute_gamma_0(y_valid.reshape(-1, n_neurons))

A_hat_kh, B_hat_kh, C_hat_kh, Q_hat_kh, R_hat_kh = prob_utils.get_kalman_ho_estimates(
    H_hankel, Gamma_0_hat, n_neurons, n_latents
)
R_diag_kh = torch.diag(R_hat_kh)
Q_diag_kh = torch.diag(Q_hat_kh)

# %% [markdown]
# ### Filtering with the *estimated* system
# Now we run the Kalman/RTS smoother using the Ho-Kalman **estimated** readout
# `C_hat_kh` and dynamics `A_hat_kh` (not the true ones). The information terms use
# the estimated observation precision $\mathbf{C}^\top\mathbf{R}^{-1}\mathbf{C}$.

# %%
# Information terms built from the ESTIMATED readout and noise.
hk_update = bmv(C_hat_kh.T, y_valid / R_diag_kh)  # estimated C bias is 0
Jk_update = (C_hat_kh.T / R_diag_kh) @ C_hat_kh
Jk_update = Jk_update.expand(y_valid.shape[0], n_time_bins, n_latents, n_latents)

m_f_hk, P_f_hk, m_p_hk, P_p_hk = kalman_information_filter(
    hk_update, Jk_update, A_hat_kh, Q_diag_kh, m_0, Q_0_diag
)
m_s_hk, P_s_hk, P_tp1_t_s_hk, z_s_hk = rts_smoother(
    m_p_hk, P_p_hk, m_f_hk, P_f_hk, A_hat_kh, n_samples=n_samples
)
rot_s_hk, m_rot_s_hk = align_latent_variables(z_valid, m_s_hk)
z_rot_s_hk = bmv(rot_s_hk, z_s_hk)

# %%
plot_rotated_latents(z_rot_s_hk, m_rot_s_hk, z_valid, label="ho-kalman (estimated system)", n_samples=n_samples)

# %% [markdown]
# ### Baseline: wrong (identity) dynamics
# To isolate the value of *learning the dynamics*, hold the readout at the **true** `C`
# and replace `A` with the identity - a random-walk latent. Everything else (the same
# `R_diag`, `Q_diag`) is held fixed, so any difference is due to the wrong dynamics.
#
# > **Micro-exercise (predict).** Before running: with $\mathbf{A}=\mathbf{I}$ the latent
# > has no rotation. Will the aligned estimate still track the spiral, or lag/oversmooth it?
# >
# > <details>
# > <summary>Solution</summary>
# >
# > Identity dynamics assume the state does not rotate, so the smoother's prior fights the
# > true spiral. The aligned mean still roughly follows the truth (the data term pulls it
# > there) but is biased toward slow drift - visibly worse than the true or Ho-Kalman `A`.
# >
# > </details>

# %%
A_eye = torch.eye(n_latents)
# True readout, identity dynamics, single consistent noise model.
heye_update = bmv(C.weight.T, (y_valid - C.bias) / R_diag)
Jeye_update = (C.weight.T / R_diag) @ C.weight
Jeye_update = Jeye_update.expand(y_valid.shape[0], n_time_bins, n_latents, n_latents)

m_f_eye, P_f_eye, m_p_eye, P_p_eye = kalman_information_filter(
    heye_update, Jeye_update, A_eye, Q_diag, m_0, Q_0_diag
)
m_s_eye, P_s_eye, P_tp1_t_s_eye, z_s_eye = rts_smoother(
    m_p_eye, P_p_eye, m_f_eye, P_f_eye, A_eye, n_samples=n_samples
)
rot_s_eye, m_rot_s_eye = align_latent_variables(z_valid, m_s_eye)
z_rot_s_eye = bmv(rot_s_eye, z_s_eye)

# %%
plot_rotated_latents(z_rot_s_eye, m_rot_s_eye, z_valid, label="identity dynamics (true readout)", n_samples=n_samples)

# %% [markdown]
# ## EM: one M-step from the true-parameter posterior
#
# EM alternates an **E-step** (infer the latent posterior with the current parameters -
# here RTS smoothing) and an **M-step** (update $\mathbf{A},\mathbf{C},\mathbf{Q},\mathbf{R}$
# to maximize the expected complete-data log-likelihood). Below we run a **single M-step**,
# fed the smoother posterior computed from the *true* parameters at the top of this notebook.
# So this near-recovers the true parameters by construction; it is not learning from scratch.
# A full EM loop would re-run the E-step with each updated parameter set and iterate.

# %%
A_hat_em, C_hat_em, Q_hat_em, R_hat_em = prob_utils.em_update_batch(m_s, P_s, P_tp1_t_s, y_valid)

# %% [markdown]
# ## Compare the learned dynamics by eigenvalues
# A linear system's behaviour is set by the eigenvalues of $\mathbf{A}$: inside the unit
# circle means stable/decaying, and the imaginary part sets rotation speed. We compare the
# true $\mathbf{A}$ against the Ho-Kalman and EM estimates.
#
# > **Stretch (optional).** Fill in the EM eigenvalues below and plot all three on the unit
# > circle. Which estimate lands closer to the true eigenvalues? Why can Ho-Kalman even
# > return an *unstable* estimate ($|\lambda|>1$)?
# >
# > <details>
# > <summary>Solution</summary>
# >
# > ```python
# > eig_kh = torch.linalg.eigvals(A_hat_kh)
# > eig_em = torch.linalg.eigvals(A_hat_em)
# > ```
# > EM (a maximum-likelihood update from the true-parameter posterior) lands essentially on
# > the true eigenvalues. Ho-Kalman estimates `A` from a finite-sample SVD of noisy
# > autocovariances with no stability constraint, so sampling error can push an eigenvalue
# > outside the unit circle even though the true system is stable.
# >
# > </details>

# %%
eig_true = torch.linalg.eigvals(mean_fn.A)
# BEGIN SOLUTION
eig_kh = torch.linalg.eigvals(A_hat_kh)
eig_em = torch.linalg.eigvals(A_hat_em)
# END SOLUTION
assert eig_kh.shape == eig_true.shape == eig_em.shape

fig, ax = plt.subplots(figsize=(5, 5))
theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), color="lightgray", linewidth=1)  # unit circle
ax.scatter(eig_true.real, eig_true.imag, marker="o", s=90, label="true", facecolors="none", edgecolors="C0")
ax.scatter(eig_kh.real, eig_kh.imag, marker="x", s=90, label="Ho-Kalman", color="C1")
ax.scatter(eig_em.real, eig_em.imag, marker="+", s=120, label="EM (1 M-step)", color="C2")
ax.set_aspect("equal")
ax.axhline(0, color="k", linewidth=0.3)
ax.axvline(0, color="k", linewidth=0.3)
ax.set_title("Eigenvalues of A: true vs. estimated")
ax.set_xlabel("real")
ax.set_ylabel("imag")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## You can now...
#
# ...estimate a linear latent dynamical system directly from data - Ho-Kalman for a fast
# non-iterative guess, EM for a maximum-likelihood refinement - and sanity-check the
# result by comparing eigenvalues of the learned transition matrix on the unit circle.
#
# **Transfer prompt.** On your own recording, use Ho-Kalman to initialize `A`, then run an
# actual EM *loop* (alternating E- and M-steps): does iterating past this single M-step
# tighten the eigenvalues toward a stable system?
#
# **Explore further.** Ho-Kalman is the classical entry point to *subspace identification*.
# Modern, more numerically robust variants estimate the state sequence directly from data
# projections rather than from Hankel-ed covariances:
# - **N4SID** (Numerical algorithms for Subspace State Space System IDentification, Van
#   Overschee & De Moor).
# - **MOESP** (Multivariable Output-Error State sPace, Verhaegen & Dewilde).
#
# Both are standard in control-systems texts and available in packages like `SIPPY` and
# MATLAB's `n4sid`; they are worth trying when Ho-Kalman's finite-sample estimates come out
# unstable.
#
# **Back to the core track:** the linear-Gaussian story ends here. When observations are
# Poisson spikes, the E-step posterior is no longer Gaussian in closed form, and we turn to
# variational inference, amortized inference / VAEs, and finally XFADS.