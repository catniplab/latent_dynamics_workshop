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

# %% [markdown] id="d87286532e00cc67"
# # Latent Variable Models: PCA to Kalman Filtering
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/01_latent_variable_models.ipynb)
#
# In this notebook, we'll simulate a 2D latent dynamical system that generates high-dimensional observations, and then explore how different latent variable models (PCA, Factor Analysis, and Kalman Filtering) can be used to infer the hidden states.
#
# We'll see how progressively richer statistical models help us recover latent structure more accurately by incorporating better probabilistic modeling and temporal dynamics.
#

# %% id="c246630a-5e5d-47a1-ae3d-f6d1da94139a"
try:
    import google.colab
    _in_colab = True
except:
    _in_colab = False

# %% colab={"base_uri": "https://localhost:8080/"} id="7d92a9ef-a475-4dbc-9d3f-3c5315185d5f" outputId="2effd3a6-e465-40fd-ca8c-0fee1f2e5119"
if _in_colab:
    # !git clone --recurse-submodules https://github.com/catniplab/latent_dynamics_workshop.git

# %% id="a26c46d5-5329-4579-908d-b0ca7d80b2d2"
import sys
import os

cwd = os.getcwd()
if _in_colab:
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop"))
    sys.path.append(os.path.join(cwd, "latent_dynamics_workshop/external/xfads"))

# %% colab={"base_uri": "https://localhost:8080/"} id="5kwHsROHVszn" outputId="af25fff6-0ea7-456d-c280-ec273467a46c"
if _in_colab:
    # !pip install -e latent_dynamics_workshop/external/xfads/

# %% id="ce41247adf239f75"
import torch
import xfads.utils as utils
import xfads.plot_utils as plot_utils
import matplotlib.pyplot as plt
import xfads.prob_utils as prob_utils
import pytorch_lightning as pl

from hydra import compose, initialize
from sklearn.decomposition import PCA
from xfads.linalg_utils import bmv, chol_bmv_solve, triangular_inverse
from xfads.prob_utils import kalman_information_filter, rts_smoother, align_latent_variables, construct_hankel

# %% colab={"base_uri": "https://localhost:8080/"} id="ab5944ab8104180f" outputId="5c7701b0-6705-49b0-a73d-499f8a088bc2"
"""config"""

cfg_dict = {
    'n_latents': 2,
    'device': 'cuda',
    'default_dtype': torch.float32,
    'seed': 1234,
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

pl.seed_everything(cfg.seed, workers=True)
torch.set_default_dtype(cfg.default_dtype)

if cfg.device == 'cuda':
    torch.cuda.empty_cache()

# %% [markdown] id="6bedb7c7f3845cd1"
# ## Simulating Data from a Latent Dynamical System
# First, let's simulate data from a linear dynamical system; here, observations represent real valued 'neural activity' read out from a lower dimensional latent state.  This corresponds to a generative model formulated as,
#
# $$
# \begin{aligned}
# \mathbf{z}_1 &\sim \mathcal{N}(0, \mathbf{Q}_0) \\\\
# \mathbf{z}_t &= \mathbf{A} \mathbf{z}_{t-1} + \mathbf{w}_t \\\\
# \mathbf{y}_t &= \mathbf{C} \mathbf{z}_t + \mathbf{v}_t
# \end{aligned}
# $$
#
# where
#
# $$
# \begin{aligned}
# \mathbf{w}_t &\sim \mathcal{N}(0, \mathbf{Q}) \\\\
# \mathbf{v}_t &\sim \mathcal{N}(0, \mathbf{R})
# \end{aligned}
# $$
#
# This synthetic setup mimics common situations in neuroscience and time-series analysis, where observed data are noisy and high-dimensional, but governed by low-dimensional latent dynamics.
#

# %% id="d514c08ff941afa5"
# Simulation parameters
n_neurons = 50
n_trials = 1000
n_time_bins = 50
n_samples = 5

omega, rho = 3.14 / 8.0, 0.97
mean_fn = utils.SpiralDynamics(omega, rho)

C = torch.nn.Linear(2, n_neurons, device="cpu").requires_grad_(False)
# C = utils.FanInLinear(2, n_neurons, device="cpu").requires_grad_(False)
C.bias.data = torch.zeros_like(C.weight[:, 0])

Q_diag = 3e-2 * torch.ones(2)
Q_0_diag = 1.0 * torch.ones(2)
# R_diag = 0.8 * torch.ones(n_neurons)
R_diag = 0.5 + 0.5 * torch.rand(n_neurons)
m_0 = torch.zeros(2)

z = utils.sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins)
y = C(z) + torch.sqrt(R_diag) * torch.randn_like(C(z))

# Split data
n_valid = n_trials // 3
n_train = n_trials - n_valid
y_train, z_train = y[:n_train], z[:n_train]
y_valid, z_valid = y[n_train:], z[n_train:]

# %% [markdown] id="3e2438ba49821560"
# ## Visualize Simulated Data
# ### Single trial

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="33a54456e94c584a" outputId="6e4bef17-3d2d-4b49-d9d0-7e72cb18b4da"
plt.plot(z[0, :, 0], label="Latent dim 1")
plt.plot(z[0, :, 1], label="Latent dim 2")
plt.legend()
plt.title("Ground Truth Latent Trajectory (Trial 0)")
plt.xlabel("Time")
plt.ylabel("Latent Value")
plt.grid(True)

# %% [markdown] id="e865c88a3ea65eae"
# ### Multiple trials
# Since this is a 2D example, we can look at multiple trajectories overlayed one another in state-space.

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="11f526da826593c" outputId="2bf0a50f-bded-46f4-a288-6e7b6afcb360"
fig, axs = plt.subplots()
plot_utils.plot_two_d_vector_field(mean_fn, axs)

for i in range(10):
    axs.plot(z[i, :, 0], z[i, :, 1], linewidth=0.5)
    if i == 0:
        axs.scatter(z[i, 0, 0], z[i, 0, 1], marker='x', label='start')
        axs.scatter(z[i, -1, 0], z[i, -1, 1], marker='o', label='end')
    else:
        axs.scatter(z[i, 0, 0], z[i, 0, 1], marker='x')
        axs.scatter(z[i, -1, 0], z[i, -1, 1], marker='o')

axs.legend()
axs.set_title("Sample trajectories")
axs.set_xlabel("dim 1")
axs.set_ylabel("dim 2")
plt.grid(True)

# %% [markdown] id="2e018c7e6175e0d7"
# ## 1: Principal Component Analysis (PCA)
#
# PCA is a classical linear method that finds directions of maximum variance in the data. While simple and efficient, it doesn't account for time or observation noise, and assumes the entire dataset lies on a linear subspace.  Still, PCA often works surprisingly well as a baseline.
#
# PCA can be related to a limiting case of a linear and Gaussian model where data is generated according to,
#
# $$
# \begin{aligned}
# \mathbf{z} &\sim \mathcal{N}(0, \mathbf{I}) \\\\
# \mathbf{y} \mid \mathbf{z} &\sim \mathcal{N}(\mathbf{C} \mathbf{z}, \sigma^2 \mathbf{I})
# \end{aligned}
# $$
#
# where we take,
#
# $$
# \sigma^2 \rightarrow 0
# $$
#
#

# %% id="86637836ad60c5f9"
pca = PCA(n_components=2)
pca.fit(y_train.reshape(-1, n_neurons))
eig_vec = pca.components_
m_pca = pca.transform(y_valid.reshape(-1, n_neurons))
m_pca = torch.tensor(m_pca.reshape(n_valid, n_time_bins, -1), dtype=torch.float32)


# %% [markdown] id="b417a5baa5623752"
# Let's visualize the latent trajectories and the directions of maximum variance in the observed data

# %% colab={"base_uri": "https://localhost:8080/", "height": 507} id="393209692f73854c" outputId="3b78cdee-245f-4b89-ffb4-54361f341c43"
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Line plot
axs[0].set_title('latent trajectory (dim 1)')
axs[0].set_box_aspect(0.6)
axs[0].plot(m_pca[0, :, 0], label='pca')
axs[0].plot(z_valid[0, :, 0], label='true')

# Imshow - horizontally stretched
axs[1].imshow(eig_vec.T, aspect=0.1)
plot_utils.remove_axs_fluff(axs[1])
axs[1].set_title('eigenvectors')

plt.tight_layout()
plt.show()

# %% [markdown] id="437c623195f467f6"
# Each column on the right is a particular 'principal component' and can be considered a dominant mode of instantaneous neural population activity; the left most column is the pattern of neural population activity that explains the most variance in the observed data.
#
# But is PCA doing a good job here at recovering the low-dimensional structure underlying the observed data? From the plot it looks like it's not -- however, this is because we have the freedom to rotate and scale the latent space arbitrarily.  Lets align these pca inferred 'latent trajectories' to the ground truth data,

# %% colab={"base_uri": "https://localhost:8080/", "height": 442} id="4b0ac10bb042f1c1" outputId="34cb86b6-395c-4088-aaea-fb25d6ac2171"
rot_pca, m_rot_pca = align_latent_variables(z_valid, m_pca)

fig, axs = plt.subplots(1, 1)

# Line plot
axs.set_title('rotated latent trajectory (dim 1)')
axs.set_box_aspect(0.6)
axs.plot(m_rot_pca[0, :, 0], label='pca rotated')
axs.plot(z_valid[0, :, 0], label='true')

axs.legend()
axs.set_xlabel('time')
plt.tight_layout()
plt.show()


# %% [markdown] id="dc3f1f71ef67032f"
# looks better, but still not great -- let's examine a slightly more sophisticated statistical method.

# %% [markdown] id="d5245b376b3e70ec"
# ## 2: Factor Analysis
#
# Factor Analysis (FA) is another type of probabilistic generative model. It models the noise in each observation dimension and finds latent variables that explain shared structure across variables.  However, FA still treats each time point independently, ignoring dynamics entirely. The corresponding generative model for FA is,
#
# $$
# \begin{aligned}
# \mathbf{z} &\sim \mathcal{N}(0, \mathbf{Q}) \\\\
# \mathbf{y}\mid \mathbf{z} &\sim \mathcal{N}(\mathbf{C} \mathbf{z}, \mathbf{R})
# \end{aligned}
# $$
#
# and the posterior, by Bayes' rule is, $p(\mathbf{z} \mid \mathbf{y}) \propto p(\mathbf{y}\mid \mathbf{z}) p(\mathbf{z})$, which can be found analytically through some Gaussian calculus as,
#
# $$
# \begin{aligned}
# p(\mathbf{z}\mid \mathbf{y}) &= \mathcal{N}(\mathbf{m}, \mathbf{P})\\\\
# \mathbf{P}^{-1} &= \mathbf{Q}^{-1} + \mathbf{C}^\top \mathbf{R}^{-1} \mathbf{C}\\\\
# \mathbf{m} &= \mathbf{P} \mathbf{C}^\top \mathbf{R}^{-1} \mathbf{y}
# \end{aligned}
# $$
#
# Let's compute posterior statistics given data and then use them to draw samples from the posterior.
#

# %% id="df29d9cabe5f99ed"
# following the equations, find the precision and mean
J_fa = (C.weight.mT / R_diag) @ C.weight + torch.diag(1 / Q_diag)
J_fa_chol = torch.linalg.cholesky(J_fa)
P_fa_chol = triangular_inverse(J_fa_chol).mT
m_fa = chol_bmv_solve(J_fa_chol, bmv(C.weight.mT, (y_valid - C.bias)))
z_fa = m_fa.unsqueeze(0) + bmv(P_fa_chol, torch.randn((n_samples, n_valid, n_time_bins, cfg.n_latents)))

# don't forget to align!
rot_fa, m_rot_fa = align_latent_variables(z_valid, m_fa)
z_rot_fa = bmv(rot_fa, z_fa)


# %% colab={"base_uri": "https://localhost:8080/", "height": 507} id="c0a6e4de821358a0" outputId="4269c390-63cb-42b5-87f9-1ad431f34caa"
fig, axs = plt.subplots(2, 1, figsize=(12, 5))

for d in range(2):
    axs[d].set_title(f'rotated latent trajectory (dim {d})')
    axs[d].set_box_aspect(0.2)

    for s in range(n_samples):
        axs[d].plot(z_rot_fa[s, 0, :, d], linewidth=0.5, color='gray')

    axs[d].plot(m_rot_fa[0, :, d], label='pca rotated')

    axs[d].plot(z_valid[0, :, d], label='true')
    axs[d].legend()
    axs[d].set_xlabel('time')

plt.tight_layout()
plt.show()


# %% [markdown] id="4171f018b34938f5"
# looks much better than pca! factor analysis is better able to handle data with a higher SNR because the additional observation uncertainty is accounted for in the generative model. however, the ground truth trajectory is fairly smooth but samples from our posterior aren't -- factor analysis cannot account for the temporal structure underlying the data.

# %% [markdown] id="b3134efc9ded8d7d"
# ## 3: Kalman Filtering and Smoothing
#
# Now, we'll account for temporal structure in the data by explicitly accounting for dynamics in the generative model -- specifically, we consider a probabilistic generative model where the latent state $\mathbf{z}_t$ evolves according to a linear stochastic difference equation and each observation $\mathbf{y}_t$ is linear and noisy readout of the latent state so that much like the data was generated we have,
#
# $$
# \begin{aligned}
# \mathbf{z}_1 &\sim \mathcal{N}(0, \mathbf{Q}_0) \\\\
# \mathbf{z}_t &= \mathbf{A} \mathbf{z}_{t-1} + \mathbf{w}_t \\\\
# \mathbf{y}_t &= \mathbf{C} \mathbf{z}_t + \mathbf{v}_t
# \end{aligned}
# $$
#
# where
#
# $$
# \begin{aligned}
# \mathbf{w}_t &\sim \mathcal{N}(0, \mathbf{Q}) \\\\
# \mathbf{v}_t &\sim \mathcal{N}(0, \mathbf{R})
# \end{aligned}
# $$
#
# The Kalman filter is a recursive algorithm for calculating the statistics of the posterior 'filtering' distribution, which by linearity and Gaussianity of the system will also be Gaussian, which we specify by,
#
# $$
# \begin{aligned}
# p(\mathbf{z}_t\mid \mathbf{y}_{1:t}) &= \mathcal{N}(\breve{\mathbf{m}}_t, \breve{\mathbf{P}}_t)
# \end{aligned}
# $$
#
# From these statistics, and a new observation, $\mathbf{y}_{t+1}$, we want to update our posterior belief about $\mathbf{z}_{t+1}$.  The great thing is that Baye's rule tells us exactly how to do this, since
#
# $$
# \begin{aligned}
#     p(\mathbf{z}_{t+1} \mid \mathbf{y}_{1:t+1}) &\propto p(\mathbf{y}_{t+1} \mid \mathbf{z}_{t+1}) p(\mathbf{z}_{t+1} \mid \mathbf{y}_{1:t})
# \end{aligned}
# $$
#
# We know how to do this Gaussian calculus, but, we don't know $p(\mathbf{z}_{t+1}\mid \mathbf{y}_{1:t})$, so lets find that first using quantities we already know,
#
# $$
# \begin{aligned}
# p(\mathbf{z}_{t+1}\mid \mathbf{y}_{1:t}) &= \int p(\mathbf{z}_{t+1}, \mathbf{z}_t\mid \mathbf{y}_{1:t}) \, d \mathbf{z}_t \\\\
#  \text{} &= \int p(\mathbf{z}_{t+1} \mid  \mathbf{z}_t) p(\mathbf{z}_t \mid \mathbf{y}_{1:t}) \, d\mathbf{z}_t \\\\
# \text{} &= \mathcal{N}(\bar{\mathbf{m}}_{t+1}, \bar{\mathbf{P}}_{t+1})
# \end{aligned}
# $$
#
# where
# $$
# \begin{aligned}
# \bar{\mathbf{m}}_{t+1} &= \mathbf{A} \breve{\mathbf{m}}_t\\\\
# \bar{\mathbf{P}}_{t+1} &= \mathbf{A} \breve{\mathbf{P}}_t \mathbf{A}^{\top} + \mathbf{Q}
# \end{aligned}
# $$
#
# Notice that the predictive distribution mean and covariance are an affine combination of the filtered mean and covariance respectively.  Finally, we return to our posterior update equation and some Gaussian calculus again,
#
# $$
# \begin{aligned}
#     p(\mathbf{z}_{t+1} \mid \mathbf{y}_{1:t+1}) &= \mathcal{N}(\mathbf{m}_{t+1}, \mathbf{P}_{t+1})\\\\
#     \mathbf{P}_{t+1}^{-1} &= \mathbf{Q}^{-1} + \mathbf{C}^\top \mathbf{R}^{-1} \mathbf{C}\\\\
#     \mathbf{m}_{t+1} &= \mathbf{P}_{t+1} (\bar{\mathbf{P}}_{t+1}^{-1} \bar{\mathbf{m}}_{t+1} + \mathbf{C}^\top \mathbf{R}^{-1} \mathbf{y}_{t+1})
# \end{aligned}
# $$

# %% id="8fe6c46cc2353bdf"
h_update = bmv(C.weight.T, (y_valid - C.bias) / R_diag)
J_update = (C.weight.T / R_diag) @ C.weight
J_update = J_update.expand(y_valid.shape[0], n_time_bins, cfg.n_latents, cfg.n_latents)

m_f, P_f, m_p, P_p = kalman_information_filter(h_update, J_update, mean_fn.A, Q_diag, m_0, Q_0_diag)
m_s, P_s, P_tp1_t_s, z_s = rts_smoother(m_p, P_p, m_f, P_f, mean_fn.A, n_samples=n_samples)
rot_s, m_rot_s = align_latent_variables(z_valid, m_s)
z_rot_s = bmv(rot_s, z_s)

# %% colab={"base_uri": "https://localhost:8080/", "height": 507} id="722ca8ae8fa31a34" outputId="46f3e7b8-da8f-482f-e509-37ac8ebe1e6c"
fig, axs = plt.subplots(2, 1, figsize=(12, 5))

for d in range(2):
    axs[d].set_title(f'rotated latent trajectory (dim {d})')
    axs[d].set_box_aspect(0.2)

    for s in range(n_samples):
        axs[d].plot(z_rot_s[s, 0, :, d], linewidth=0.5, color='gray')

    axs[d].plot(m_rot_s[0, :, d], label='pca rotated')

    axs[d].plot(z_valid[0, :, d], label='true')
    axs[d].legend()
    axs[d].set_xlabel('time')

plt.tight_layout()
plt.show()

# %% [markdown] id="f3658a0051d3cd33"
# look how much smoother the posterior samples are!

# %% [markdown] id="99fcf7c14edb5b0b"
# ## Comparisons

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} id="7d7a9062db51be50" outputId="64d7f14c-d4e2-4499-8e7b-e1122e05b9da"
plt.figure(figsize=(10, 5))
plt.plot(z_valid[0, :, 0], label='Ground Truth', linewidth=2)
plt.plot(m_rot_pca[0, :, 0], label='PCA', linestyle='--')
plt.plot(m_rot_fa[0, :, 0], label='Factor Analysis', linestyle='--')
plt.plot(m_rot_s[0, :, 0], label='Kalman (Smoothed)', linestyle='--')
plt.title("Latent Dimension 1: True vs. Estimated")
plt.xlabel("Time")
plt.ylabel("Latent Value")
plt.legend()
plt.grid(True)

# %% [markdown] id="41a5ffb05ba0af39"
# # Learning Latent Dynamics Parameters
#
# So far, we assumed access to the true dynamics and observation parameters. But in practice, these must be estimated from data.
#
# We'll now explore two approaches for learning the parameters of a Linear Dynamical System (LDS):
#
# 1. **System Identification** using the **Kalman-Ho algorithm** (a subspace method).
# 2. **Expectation-Maximization (EM)** for LDS parameter learning (a probabilistic approach).
#
# We'll compare their learned state transition matrices via their eigenvalues.
#

# %% [markdown] id="7cee93c263de19dd"
# ## Kalman-Ho System Identification
#
# The Kalman-Ho algorithm is a classic subspace identification method. It works by constructing a Hankel matrix from the observed outputs and applying an SVD to extract latent dynamics.
#
# This method is fast and often used in control and system ID applications.
#

# %% [markdown] id="89f6b9e75a62d92d"
# # Kalman–Ho Algorithm: Derivation and Parameter Estimation
#
# We consider the linear–Gaussian state-space model:
#
# $$
# \begin{aligned}
# z_{t+1} &= A z_t + w_t, \quad w_t \sim \mathcal{N}(0, Q) \\
# y_t &= C z_t + v_t, \quad v_t \sim \mathcal{N}(0, R)
# \end{aligned}
# $$
#
# We assume stationarity: $z_t \sim \mathcal{N}(0, P_\infty)$, where $P_\infty$ satisfies the discrete Lyapunov equation:
#
# $$
# P_\infty = A P_\infty A^\top + Q
# $$
#
# ---
#
# ## Step 1: Estimate Output Covariances
#
# Given output sequences $\{ y_t \}$, compute empirical autocovariances:
#
# $$
# \Gamma_k := \mathbb{E}[y_{t+k} y_t^\top] \approx \frac{1}{T - k} \sum_{t=1}^{T-k} y_{t+k} y_t^\top, \quad k = 0, 1, \dots, K
# $$
#
# ---
#
# ## Step 2: Build the Hankel Matrix
#
# Construct the block Hankel matrix:
#
# $$
# H = \begin{bmatrix}
# \Gamma_1 & \Gamma_2 & \cdots & \Gamma_k \\
# \Gamma_2 & \Gamma_3 & \cdots & \Gamma_{k+1} \\
# \vdots & \vdots & \ddots & \vdots \\
# \Gamma_j & \Gamma_{j+1} & \cdots & \Gamma_{j+k-1}
# \end{bmatrix}
# $$
#
# ---
#
# ## Step 3: Low-Rank Factorization via SVD
#
# Perform singular value decomposition on the Hankel matrix:
#
# $$
# H \approx U \Sigma V^\top
# $$
#
# Extract the rank-$d$ approximation, and define:
#
# $$
# \mathcal{O} := U_d \Sigma_d^{1/2}, \quad \mathcal{C} := \Sigma_d^{1/2} V_d^\top
# $$
#
# Then:
#
# - $C \approx \mathcal{O}_{\text{first block}}$
# - $B \approx \mathcal{C}_{\text{first block}}$
#
# ---
#
# ## Step 4: Estimate the State Transition Matrix $A$
#
# Using shift-invariance of the observability matrix:
#
# - Let $\mathcal{O}_{\text{top}}$ be all but the last block row
# - Let $\mathcal{O}_{\text{bottom}}$ be all but the first block row
#
# Then:
#
# $$
# A \approx \mathcal{O}_{\text{bottom}}^\dagger \mathcal{O}_{\text{top}}
# $$
#
# ---
#
# ## Step 5: Estimate Process Noise Covariance $Q$
#
# Using the fact that $B \approx$ Cholesky-like factor of $Q$, estimate:
#
# $$
# Q \approx B B^\top
# $$
#
# ---
#
# ## Step 6: Solve for Stationary Covariance $P_\infty$
#
# Solve the discrete Lyapunov equation:
#
# $$
# P_\infty = A P_\infty A^\top + Q
# $$
#
# ---
#
# ## Step 7: Estimate Observation Noise Covariance $R$
#
# Use the identity from autocovariance at lag 0:
#
# $$
# \Gamma_0 = C P_\infty C^\top + R
# $$
#
# Solve for:
#
# $$
# R \approx \Gamma_0 - C P_\infty C^\top
# $$
#
# ---
#
# ## Summary
#
# The Kalman–Ho algorithm enables non-iterative identification of the state-space model:
#
# - Recover $A$, $B$, $C$ from SVD of Hankel matrix
# - Estimate $Q$ from $B B^\top$
# - Solve for $P_\infty$ via Lyapunov equation
# - Estimate $R$ from the empirical autocovariance
#
# This procedure is fully data-driven and avoids iterative inference or EM.
#

# %% id="2e13713ba37ada2d"
# Construct Hankel matrix
H_hankel = prob_utils.construct_hankel(y_valid, 50, 50)
Gamma_0_hat = prob_utils.compute_gamma_0(y_valid.reshape(-1, n_neurons))

# Estimate system matrices using Kalman-Ho
A_hat_kh, B_hat_kh, C_hat_kh, Q_hat_kh, R_hat_kh = prob_utils.get_kalman_ho_estimates(
    H_hankel, Gamma_0_hat, n_neurons, cfg.n_latents
)
R_diag_kh = torch.diag(R_hat_kh)
Q_diag_kh = torch.diag(Q_hat_kh)

# Eigenvalues of learned A
eig_vals_kh_hat = torch.linalg.eigvals(A_hat_kh)

# %% [markdown] id="a622155f52f4964e"
# Lets compare Kalman filtering with identity dynamics versus those inferred by the Ho-Kalman algorithm

# %% id="4a95dd579706c931"
hk_update = bmv(C_hat_kh.T, y_valid / R_diag_kh) # C bias is 0
Jk_update = (C_hat_kh.T @ R_hat_kh) @ C_hat_kh
Jk_update = Jk_update.expand(y_valid.shape[0], n_time_bins, cfg.n_latents, cfg.n_latents)

m_f_hk, P_f_hk, m_p_hk, P_p_hk = kalman_information_filter(h_update, J_update, A_hat_kh, Q_diag_kh, m_0, Q_0_diag)
m_s_hk, P_s_hk, P_tp1_t_s_hk, z_s_hk = rts_smoother(m_p_hk, P_p_hk, m_f_hk, P_f_hk, A_hat_kh, n_samples=n_samples)
rot_s_hk, m_rot_s_hk = align_latent_variables(z_valid, m_s_hk)
z_rot_s_hk = bmv(rot_s_hk, z_s_hk)

# %% colab={"base_uri": "https://localhost:8080/", "height": 507} id="ee0380e3eb8ba267" outputId="99661177-b1ce-4dbf-99d7-126bc356044f"
fig, axs = plt.subplots(2, 1, figsize=(12, 5))

for d in range(2):
    axs[d].set_title(f'rotated latent trajectory (dim {d})')
    axs[d].set_box_aspect(0.2)

    for s in range(n_samples):
        axs[d].plot(z_rot_s_hk[s, 0, :, d], linewidth=0.5, color='gray')

    axs[d].plot(m_rot_s_hk[0, :, d], label='ho-kalman rotated')

    axs[d].plot(z_valid[0, :, d], label='true')
    axs[d].legend()
    axs[d].set_xlabel('time')

plt.tight_layout()
plt.show()

# %% id="f667f8560bbc051d"
R_hat_eye = torch.ones(n_neurons)
A_hat_eye = torch.eye(cfg.n_latents)
C_hat_eye = torch.nn.Linear(cfg.n_latents, n_neurons, bias=False, device=hk_update.device).requires_grad_(False)
heye_update = bmv(C_hat_eye.weight.T, y_valid / R_diag) # C bias is 0
Jeye_update = (C_hat_eye.weight.T / R_hat_eye) @ C_hat_eye.weight
Jeye_update = Jeye_update.expand(y_valid.shape[0], n_time_bins, cfg.n_latents, cfg.n_latents)

m_f_eye, P_f_eye, m_p_eye, P_p_eye = kalman_information_filter(heye_update, Jeye_update, A_hat_eye, Q_diag, m_0, Q_0_diag)
m_s_eye, P_s_eye, P_tp1_t_s_eye, z_s_eye = rts_smoother(m_p_eye, P_p_eye, m_f_eye, P_f_eye, A_hat_eye, n_samples=n_samples)
rot_s_eye, m_rot_s_eye = align_latent_variables(z_valid, m_s_eye)
z_rot_s_eye = bmv(rot_s_eye, z_s_eye)


# %% colab={"base_uri": "https://localhost:8080/", "height": 507} id="a057b3b0b0d4b418" outputId="84befa9b-7473-4f8b-e560-e43745998257"
fig, axs = plt.subplots(2, 1, figsize=(12, 5))

for d in range(2):
    axs[d].set_title(f'rotated latent trajectory (dim {d})')
    axs[d].set_box_aspect(0.2)

    for s in range(n_samples):
        axs[d].plot(z_rot_s_eye[s, 0, :, d], linewidth=0.5, color='gray')

    axs[d].plot(m_rot_s_eye[0, :, d], label='identity kalman rotated')

    axs[d].plot(z_valid[0, :, d], label='true')
    axs[d].legend()
    axs[d].set_xlabel('time')

plt.tight_layout()
plt.show()

# %% [markdown] id="49b0d0044ef5d47a"
# ## EM for LDS Parameter Estimation
#
# The EM algorithm is a probabilistic approach to estimating LDS parameters. It alternates between:
#
# - **E-step**: Inferring latent trajectories (here, using RTS smoothing).
# - **M-step**: Updating parameters to maximize the expected complete-data log-likelihood.
#
# This method can be more accurate and flexible, especially with noise or missing data.
#

# %% id="b1a4cdcb2eb9c13e"
# Estimate parameters using EM
A_hat_em, C_hat_em, Q_hat_em, R_hat_em = prob_utils.em_update_batch(m_s, P_s, P_tp1_t_s, y_valid)

# Eigenvalues of learned A
eig_vals_em_hat = torch.linalg.eigvals(A_hat_em)


# %% id="1b7d6d1e-34f0-4bbb-b889-9a252e1cff8a"
