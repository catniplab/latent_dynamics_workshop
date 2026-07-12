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
# # Linear latent variable models: PCA -> Factor Analysis -> Kalman/RTS
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/02_linear_lvms.ipynb)
#
# **Takeaway:** on the same spiral latent dynamical system, each richer linear-Gaussian
# model (PCA -> Factor Analysis -> Kalman/RTS smoother) recovers the hidden state better,
# because it accounts for more structure: observation noise, then temporal dynamics.
#
# **Where this sits.**
# - *Optional companion:* [`03_system_id_and_em`](03_system_id_and_em.ipynb) learns the
#   dynamics from data (Ho-Kalman subspace ID and EM) instead of assuming them known.
# - *Next core notebook:* variational inference, once the observations become
#   non-Gaussian (Poisson spikes) and the tidy closed forms below stop applying.
#
# The math lives in the lecture notes; here we *do* it. Section links point to the notes:
# `sec:ppca` (Probabilistic PCA), `sec:fa` (Factor analysis), `sec:rotation` /
# `sec:scale` (the rotation and scale ambiguities), `sec:kalman` (the Kalman filter),
# and `sec:smoothing` (RTS smoothing and forecasting).

# %% [markdown]
# ## Setup (Colab)
# On Colab this clones the repo and installs `xfads`. Locally it is a no-op.

# %%
try:
    import google.colab
    _in_colab = True
except ImportError:
    _in_colab = False

# %%
if _in_colab:
    # !git clone --recurse-submodules https://github.com/catniplab/latent_dynamics_workshop.git
    # !pip install -e latent_dynamics_workshop/external/xfads/
    pass

# %%
import os
import sys

if _in_colab:
    cwd = os.getcwd()
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop"))
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop/external/xfads"))

# %%
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import xfads.utils as utils
import xfads.plot_utils as plot_utils
from sklearn.decomposition import PCA
from xfads.linalg_utils import bmv, chol_bmv_solve, triangular_inverse
from xfads.prob_utils import (
    kalman_information_filter,
    rts_smoother,
    align_latent_variables,
)

# %%
# Minimal config: 2 latent dimensions, run on CPU unless a GPU is available.
n_latents = 2
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"

pl.seed_everything(seed, workers=True)
torch.set_default_dtype(torch.float32)


# %% [markdown]
# ### One plotting helper (reused below)
# The four inference sections all draw the same figure: posterior samples (gray),
# the posterior mean, and the ground-truth latent, one panel per dimension. We
# define it once so the concept cells stay short.

# %%
from code_pack.plotting import plot_rotated_latents

# %% [markdown]
# ## Show the object: a spiral latent dynamical system
#
# We simulate a 2D latent that spirals inward, read out linearly into 50 noisy
# "neurons". The generative model (the one Kalman/RTS below will invert) is a
# linear-Gaussian state-space model:
#
# $$
# \begin{aligned}
# \mathbf{z}_1 &\sim \mathcal{N}(0, \mathbf{Q}_0), &
# \mathbf{z}_t &= \mathbf{A}\mathbf{z}_{t-1} + \mathbf{w}_t, &
# \mathbf{w}_t &\sim \mathcal{N}(0, \mathbf{Q}) \\
# \mathbf{y}_t &= \mathbf{C}\mathbf{z}_t + \mathbf{v}_t, &
# \mathbf{v}_t &\sim \mathcal{N}(0, \mathbf{R}).
# \end{aligned}
# $$
#
# We keep the latent named `z` throughout (matching the `xfads` code and the notes'
# `z`-convention). Full derivations of everything below are in the notes; see `sec:kalman`.

# %%
# Simulation parameters
n_neurons = 50
n_trials = 1000
n_time_bins = 50
n_samples = 5

omega, rho = 3.14 / 8.0, 0.97
mean_fn = utils.SpiralDynamics(omega, rho)  # linear spiral dynamics; mean_fn.A is A

C = torch.nn.Linear(2, n_neurons, device="cpu").requires_grad_(False)
C.bias.data = torch.zeros_like(C.weight[:, 0])

Q_diag = 3e-2 * torch.ones(2)     # process-noise variance
Q_0_diag = 1.0 * torch.ones(2)    # initial-state variance
R_diag = 0.5 + 0.5 * torch.rand(n_neurons)  # per-neuron observation-noise variance (non-constant!)
m_0 = torch.zeros(2)

z = utils.sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins)
y = C(z) + torch.sqrt(R_diag) * torch.randn_like(C(z))

# Train/validation split
n_valid = n_trials // 3
n_train = n_trials - n_valid
y_train, z_train = y[:n_train], z[:n_train]
y_valid, z_valid = y[n_train:], z[n_train:]

# %% [markdown]
# ### Single trial
# %%
plt.plot(z[0, :, 0], label="Latent dim 1")
plt.plot(z[0, :, 1], label="Latent dim 2")
plt.legend()
plt.title("Ground truth latent trajectory (trial 0)")
plt.xlabel("Time")
plt.ylabel("Latent value")
plt.grid(True)

# %% [markdown]
# ### Multiple trials in state space
# Because the latent is 2D we can overlay trajectories on the vector field.
# %%
fig, axs = plt.subplots()
plot_utils.plot_two_d_vector_field(mean_fn, axs)

for i in range(10):
    axs.plot(z[i, :, 0], z[i, :, 1], linewidth=0.5)
    if i == 0:
        axs.scatter(z[i, 0, 0], z[i, 0, 1], marker="x", label="start")
        axs.scatter(z[i, -1, 0], z[i, -1, 1], marker="o", label="end")
    else:
        axs.scatter(z[i, 0, 0], z[i, 0, 1], marker="x")
        axs.scatter(z[i, -1, 0], z[i, -1, 1], marker="o")

axs.legend()
axs.set_title("Sample trajectories")
axs.set_xlabel("dim 1")
axs.set_ylabel("dim 2")
plt.grid(True)

# %% [markdown]
# ## 1. Principal Component Analysis (PCA)
#
# PCA finds directions of maximum variance. It ignores both observation noise and
# time, and corresponds to the $\sigma^2 \to 0$ limit of a linear-Gaussian model
# $\mathbf{z}\sim\mathcal{N}(0,\mathbf{I})$, $\mathbf{y}\mid\mathbf{z}\sim\mathcal{N}(\mathbf{C}\mathbf{z}, \sigma^2\mathbf{I})$.
# Derivation and the exact PPCA connection: notes `sec:ppca`.

# %%
pca = PCA(n_components=2)
pca.fit(y_train.reshape(-1, n_neurons))
eig_vec = pca.components_
m_pca = pca.transform(y_valid.reshape(-1, n_neurons))
m_pca = torch.tensor(m_pca.reshape(n_valid, n_time_bins, -1), dtype=torch.float32)

# %% [markdown]
# Each column below is a principal component - a dominant instantaneous population
# pattern. The left panel already hints the raw PCA latent does not match the truth:
# a latent variable model is only identifiable up to rotation and scale (`sec:rotation`,
# `sec:scale`), so we must *align* before comparing.

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].set_title("latent trajectory (dim 1)")
axs[0].set_box_aspect(0.6)
axs[0].plot(m_pca[0, :, 0], label="pca")
axs[0].plot(z_valid[0, :, 0], label="true")
axs[0].legend()

axs[1].imshow(eig_vec.T, aspect=0.1)
plot_utils.remove_axs_fluff(axs[1])
axs[1].set_title("eigenvectors")
plt.tight_layout()
plt.show()

# %% [markdown]
# `align_latent_variables` regresses the estimate onto the truth to undo the
# rotation/scale ambiguity. After alignment PCA is a reasonable baseline - but not great.

# %%
rot_pca, m_rot_pca = align_latent_variables(z_valid, m_pca)

fig, axs = plt.subplots(1, 1)
axs.set_title("rotated latent trajectory (dim 1)")
axs.set_box_aspect(0.6)
axs.plot(m_rot_pca[0, :, 0], label="pca rotated")
axs.plot(z_valid[0, :, 0], label="true")
axs.legend()
axs.set_xlabel("time")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Factor Analysis (FA)
#
# FA is a probabilistic model that, unlike PCA, models a *separate* noise variance per
# observed dimension. It still treats each time point independently. With a standard
# latent prior $\mathbf{z}\sim\mathcal{N}(0,\mathbf{I})$ and
# $\mathbf{y}\mid\mathbf{z}\sim\mathcal{N}(\mathbf{C}\mathbf{z}+\mathbf{b},\mathbf{R})$,
# Gaussian calculus gives a Gaussian posterior $p(\mathbf{z}\mid\mathbf{y})=\mathcal{N}(\mathbf{m},\mathbf{P})$ with
#
# $$
# \mathbf{P}^{-1} = \mathbf{I} + \mathbf{C}^\top \mathbf{R}^{-1}\mathbf{C},
# \qquad
# \mathbf{m} = \mathbf{P}\,\mathbf{C}^\top \mathbf{R}^{-1}(\mathbf{y}-\mathbf{b}).
# $$
#
# Full derivation: notes `sec:fa`. The key term is $\mathbf{R}^{-1}$: because `R_diag`
# is non-constant here, each neuron is weighted by its *precision* (inverse noise).

# %% [markdown]
# > **Micro-exercise (fill one line).** Complete the precision-weighted readout below.
# > First **predict**: should a noisier neuron (larger `R_diag`) count *more* or *less*
# > toward the latent estimate? Then fill the `# YOUR CODE HERE`.
# >
# > <details>
# > <summary>Solution</summary>
# >
# > A noisier neuron should count *less*, so we divide by its variance (multiply by
# > precision $1/R$):
# >
# > ```python
# > readout = bmv(C.weight.mT, (y_valid - C.bias) / R_diag)
# > ```
# > Try deleting `/ R_diag` and re-running the alignment plot: the estimate degrades
# > because loud, noisy neurons then dominate the latent.
# >
# > </details>

# %%
# Posterior precision J = I + C^T R^{-1} C, then P = J^{-1}.
J_fa = (C.weight.mT / R_diag) @ C.weight + torch.eye(n_latents)
J_fa_chol = torch.linalg.cholesky(J_fa)
P_fa_chol = triangular_inverse(J_fa_chol).mT

# Posterior mean m = P C^T R^{-1} (y - b): the readout must be precision-weighted.
readout = bmv(C.weight.mT, (y_valid - C.bias) / R_diag)  # YOUR CODE HERE (the / R_diag)
m_fa = chol_bmv_solve(J_fa_chol, readout)
z_fa = m_fa.unsqueeze(0) + bmv(
    P_fa_chol, torch.randn((n_samples, n_valid, n_time_bins, n_latents))
)

# don't forget to align!
rot_fa, m_rot_fa = align_latent_variables(z_valid, m_fa)
z_rot_fa = bmv(rot_fa, z_fa)

# %%
plot_rotated_latents(z_rot_fa, m_rot_fa, z_valid, label="factor analysis", n_samples=n_samples)

# %% [markdown]
# Much better than PCA: modelling per-neuron observation noise sharpens the estimate.
# But the posterior samples are jagged - FA has no notion of time, so it cannot
# borrow strength from neighbouring bins. That is exactly what the Kalman filter adds.

# %% [markdown]
# ## 3. Kalman filter and RTS smoother
#
# Now we put the dynamics back in. The Kalman filter is the recursive exact posterior
# $p(\mathbf{z}_t\mid\mathbf{y}_{1:t})=\mathcal{N}(\mathbf{m}_t,\mathbf{P}_t)$ for the
# linear-Gaussian model above. Each step **predicts** through the dynamics, then
# **updates** with the new observation (information form):
#
# $$
# \begin{aligned}
# \text{predict:}\quad & \bar{\mathbf{m}}_t = \mathbf{A}\mathbf{m}_{t-1}, &
#   \bar{\mathbf{P}}_t &= \mathbf{A}\mathbf{P}_{t-1}\mathbf{A}^\top + \mathbf{Q}\\
# \text{update:}\quad & \mathbf{P}_t^{-1} = \bar{\mathbf{P}}_t^{-1} + \mathbf{C}^\top\mathbf{R}^{-1}\mathbf{C}, &
#   \mathbf{m}_t &= \mathbf{P}_t\big(\bar{\mathbf{P}}_t^{-1}\bar{\mathbf{m}}_t + \mathbf{C}^\top\mathbf{R}^{-1}\mathbf{y}_t\big).
# \end{aligned}
# $$
#
# Note the update precision adds the *predicted* precision $\bar{\mathbf{P}}_t^{-1}$
# (not $\mathbf{Q}^{-1}$) to the observation information $\mathbf{C}^\top\mathbf{R}^{-1}\mathbf{C}$.
# Full derivation: notes `sec:kalman`. The **RTS smoother** then passes backward to use
# *all* the data for each $\mathbf{z}_t$, giving $p(\mathbf{z}_t\mid\mathbf{y}_{1:T})$
# see notes `sec:smoothing` (RTS smoothing and forecasting).

# %% [markdown]
# > **Micro-exercise (predict, then tweak).** Before running: as the observation noise
# > `R_infer` shrinks, does the filter trust the data more or less, and do the smoothed
# > samples get smoother or jumpier? Then set `Q_infer = 10 * Q_diag` (or
# > `R_infer = 2 * R_diag`) in the cell below and re-run.
# >
# > <details>
# > <summary>Solution</summary>
# >
# > As $\mathbf{R}\to 0$ the Kalman gain $\to 1$: the filter trusts each observation and
# > tracks it tightly (jumpier). Inflating `Q` (more process noise) also loosens the
# > prior smoothing, so samples get jumpier; inflating `R` makes the smoother lean on
# > the dynamics and produce smoother, more prior-dominated trajectories. See `sec:kalman`.
# >
# > </details>

# %%
# Observation and process noise the filter *assumes* (tweak these in the exercise).
R_infer = R_diag
Q_infer = Q_diag

# Information-form observation terms: h = C^T R^{-1} (y - b), J = C^T R^{-1} C.
h_update = bmv(C.weight.T, (y_valid - C.bias) / R_infer)
J_update = (C.weight.T / R_infer) @ C.weight
J_update = J_update.expand(y_valid.shape[0], n_time_bins, n_latents, n_latents)

m_f, P_f, m_p, P_p = kalman_information_filter(h_update, J_update, mean_fn.A, Q_infer, m_0, Q_0_diag)
m_s, P_s, P_tp1_t_s, z_s = rts_smoother(m_p, P_p, m_f, P_f, mean_fn.A, n_samples=n_samples)
rot_s, m_rot_s = align_latent_variables(z_valid, m_s)
z_rot_s = bmv(rot_s, z_s)

# %%
plot_rotated_latents(z_rot_s, m_rot_s, z_valid, label="kalman (smoothed)", n_samples=n_samples)

# %% [markdown]
# The smoothed samples are far smoother than FA's: the dynamics tie the bins together.

# %% [markdown]
# ## Comparison
# Same trial, three models. Each added assumption (noise model, then dynamics) moves
# the estimate closer to the truth.

# %%
plt.figure(figsize=(10, 5))
plt.plot(z_valid[0, :, 0], label="Ground truth", linewidth=2)
plt.plot(m_rot_pca[0, :, 0], label="PCA", linestyle="--")
plt.plot(m_rot_fa[0, :, 0], label="Factor Analysis", linestyle="--")
plt.plot(m_rot_s[0, :, 0], label="Kalman (smoothed)", linestyle="--")
plt.title("Latent dimension 1: true vs. estimated")
plt.xlabel("Time")
plt.ylabel("Latent value")
plt.legend()
plt.grid(True)

# %% [markdown]
# ## You can now...
#
# ...take a high-dimensional noisy recording, fit PCA / Factor Analysis / a Kalman-RTS
# smoother, and read off latent trajectories - understanding *why* each step helps
# (observation noise, then temporal dynamics) and always aligning before you compare.
#
# **Transfer prompt.** Point this at your own data: bin your population activity into a
# `(trials, time, neurons)` array and run the FA and Kalman/RTS cells. Which matters more
# for your recording - modelling per-neuron noise (FA) or temporal smoothing (Kalman)?
#
# **Next.**
# - *Optional:* [`03_system_id_and_em`](03_system_id_and_em.ipynb) - here we *knew* `A`,
#   `C`, `Q`, `R`; there we learn the dynamics from data (Ho-Kalman ID, EM).
# - *Core:* variational inference, for when the observations are Poisson spikes rather
#   than Gaussian and these closed forms no longer exist (notes `sec:expfam`, `sec:vi`).
