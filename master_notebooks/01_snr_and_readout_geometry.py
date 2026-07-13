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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SNR and readout geometry (optional companion)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/01_snr_and_readout_geometry.ipynb)
#
# **Takeaway:** a Fisher-information SNR bound says how much a Poisson population
# tells you about its latent $z$, and *how* neurons read out $z$ - all dimensions
# at once (random projection) or one axis each (axis-aligned) - reshapes the
# raster without changing that information budget much.
#
# This is an optional detour off `00_state_space_intuition.ipynb`; it reuses the
# same low-D-latent -> Poisson-spikes generative model built there and needs the
# `neurofisherSNR` external submodule.

# %% [markdown]
# ## Setup (Colab)
# On Colab this clones the repo (with submodules) and moves into it, then installs
# `neurofisherSNR`, so it and `code_pack` import exactly as they do locally.

# %%
import os

try:
    import google.colab
    _in_colab = True
except ImportError:
    _in_colab = False

if _in_colab:
    # !git clone --recurse-submodules https://github.com/catniplab/latent_dynamics_workshop.git
    os.chdir("latent_dynamics_workshop")
    # !pip install -e external/neurofisherSNR/

import numpy as np
import matplotlib.pyplot as plt

# fixed seed so the SNR numbers and rasters are reproducible run to run
rng = np.random.default_rng(20260707)

from neurofisherSNR.optimize import optimize_C
from neurofisherSNR.snr import SNR_bound_instantaneous
from neurofisherSNR.utils import power_to_dB, power_from_dB
from code_pack.plotting import plot_raster, sort_by_dominant_loading, sort_by_loading_dim

# %% [markdown]
# ## Rebuild the generative model from notebook 00
#
# Same 1-D sinusoidal latent, same population of Poisson neurons with an
# exponential rate link $\lambda(t) = \exp(z(t)\,C^\top + b)$.

# %%
nT = 1000
T = 10
frq = 0.3
tr = np.linspace(0, T, nT)
dt = tr[1] - tr[0]
z = np.sin(2 * np.pi * frq * tr)[:, None]  # shape (nT, 1)

nNeuron = 100
target_rate = 5.0
max_rate = 100.0

# %% [markdown]
# ## Signal-to-noise ratio of population spike trains
#
# We use the `neurofisherSNR` package to estimate the Fisher-information upper
# bound on SNR: how much the spike trains carry about the latent per time bin.
# The Fisher-information view of Poisson observations (and this SNR bound) is
# developed in the lecture notes' Poisson observation section, based on:
#
# - Jeon, H., & Park, I. M. (2024). Quantifying Signal-to-Noise Ratio in Neural
#   Latent Trajectories via Fisher Information. 32nd European Signal Processing
#   Conference (EUSIPCO). arXiv:2408.08752.
#   - https://arxiv.org/abs/2408.08752
#   - https://github.com/catniplab/neurofisherSNR
#
# Decibels are a logarithmic unit.
#
# Below, we visualize how the sorted spike raster changes as the population
# SNR spans from extremely noisy (-20 dB) to highly structured (+20 dB).

# %% [markdown]
# **Predict before running:** which SNR target below will be the first where the
# travelling band is visually clear: `-20`, `-10`, `0`, `10`, or `20` dB?
#
# Your prediction:

# %%
C_base = rng.standard_normal((nNeuron, 1))
b0 = np.zeros((1, nNeuron))

# Spanning SNRs from -20 dB to +20 dB
snr_targets = [-20.0, -10.0, 0.0, 10.0, 20.0]
fig, axs = plt.subplots(1, len(snr_targets), figsize=(15, 3), sharex=True, sharey=True)

for i, target in enumerate(snr_targets):
    C_scaled, b_scaled, achieved_snr = optimize_C(
        x=z,
        C=C_base,
        b=b0,
        tgt_rate_per_bin=target_rate,
        max_rate_per_bin=max_rate,
        tgt_snr=target,
        snr_fn=SNR_bound_instantaneous,
        priority="mean",
    )
    lam = np.exp(z @ C_scaled.T + b_scaled)
    y_scaled = rng.poisson(lam * dt)
    plot_raster(
        axs[i],
        y_scaled,
        f"SNR = {achieved_snr:.0f} dB",
        dt=dt,
        order=sort_by_loading_dim(C_scaled, 0),
        ylabel="sorted neurons" if i == 0 else "",
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## A 2-D latent for the readout-geometry question
#
# Add a centered sawtooth as a second latent (as in notebook 00),
# $z_2(t) = 1.5\,((t \bmod 1) - 0.5)$, so we have two dimensions to read out.

# %%
z2 = 1.5 * ((tr % 1) - 0.5)[:, np.newaxis]
Z = np.hstack([z, z2])  # shape [nT, dLatent]
dLatent = Z.shape[1]

# %%
plt.subplots(2, 1, figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(tr, z)
plt.ylabel('first latent dim')
plt.subplot(2, 1, 2)
plt.plot(tr, z2)
plt.ylabel('second latent dim')
plt.xlabel('time (s)')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Readout geometry at matched 2-D SNR
#
# Now keep the information budget fixed and change the readout geometry. A
# random-projection population lets each neuron mix both latent dimensions. An
# axis-aligned population makes each neuron read out only one latent dimension.
#
# - Gao, P., & Ganguli, S. (2015). On Simplicity and Complexity in the Brave New
#   World of Large-Scale Neuroscience. Current Opinion in Neurobiology, 32,
#   148-155.
# - Whittington, J. C. R., Dorrell, W., Ganguli, S., & Behrens, T. E. J. (2022).
#   Disentangling with Biological Constraints: A Theory of Functional Cell Types.
#   arXiv:2210.01768.
#
# **Predict before running:** both populations below are scaled to the same
# 2-D SNR target. Which sorted raster should show two clearer cell groups, and
# why?
#
# Your prediction:

# %%
target_snr_2d = 8.0  # After predicting, also try: -5.0, 5.0, 15.0
b0 = np.zeros((1, nNeuron))

C_random_base = rng.standard_normal((nNeuron, dLatent))
C_random, b_random, SNRdb_random = optimize_C(
    x=Z,
    C=C_random_base,
    b=b0,
    tgt_rate_per_bin=target_rate,
    max_rate_per_bin=max_rate,
    tgt_snr=target_snr_2d,
    snr_fn=SNR_bound_instantaneous,
    priority="mean",
)
lam_random = np.exp(Z @ C_random.T + b_random)
y_random = rng.poisson(lam_random * dt)

axis_mask = rng.random(nNeuron) < 0.5
C_axis_base = rng.standard_normal((nNeuron, dLatent))
C_axis_base[axis_mask, 0] = 0
C_axis_base[~axis_mask, 1] = 0
C_axis, b_axis, SNRdb_axis = optimize_C(
    x=Z,
    C=C_axis_base,
    b=b0,
    tgt_rate_per_bin=target_rate,
    max_rate_per_bin=max_rate,
    tgt_snr=target_snr_2d,
    snr_fn=SNR_bound_instantaneous,
    priority="mean",
)
lam_axis = np.exp(Z @ C_axis.T + b_axis)
y_axis = rng.poisson(lam_axis * dt)

# per-axis SNR bounds, using only the neurons/latent for that axis
print(f"random projection SNR: {SNRdb_random:.2f} dB")
print(f"axis-aligned SNR:     {SNRdb_axis:.2f} dB")
SNRdb1 = SNR_bound_instantaneous(Z[:, [0]], C_axis[:, [0]].T, b_axis)
SNRdb2 = SNR_bound_instantaneous(Z[:, [1]], C_axis[:, [1]].T, b_axis)
print(f"axis 1: {SNRdb1:.2f} dB,  axis 2: {SNRdb2:.2f} dB")
# For axis-aligned loadings the Fisher information is block-diagonal, so the
# information about the two independent axes adds. The full-population SNR bound
# therefore sits between the two per-axis bounds - it is NOT their average
# (that would require equal latent power and equal per-axis noise, which does not
# hold here: the sine and the sawtooth carry different power).

# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
plot_raster(axs[0, 0], y_random, "random projection", dt=dt, ylabel="neurons")
plot_raster(
    axs[0, 1],
    y_random,
    "random, sorted by z1 loading",
    dt=dt,
    order=sort_by_loading_dim(C_random, 0),
    ylabel="sorted neurons",
)
plot_raster(axs[1, 0], y_axis, "axis-aligned", dt=dt, ylabel="neurons")
plot_raster(
    axs[1, 1],
    y_axis,
    "axis-aligned, sorted by cell type",
    dt=dt,
    order=sort_by_dominant_loading(C_axis),
    ylabel="sorted neurons",
)
plt.tight_layout()
plt.show()

# %% [markdown]
# > **Stretch (optional):** make the two latents carry equal power - rescale
# > `z2` so `mean(z2**2) == mean(z**2)` - and re-run this cell. Verify the full
# > SNR now lands much closer to the mean of the two per-axis SNRs, and explain
# > why unequal latent power breaks the naive averaging.
#
# <details>
# <summary>Solution</summary>
#
# ```python
# z2_eq = z2 * np.sqrt(np.mean(z**2) / np.mean(z2**2))  # equal power
# Z_eq = np.hstack([z, z2_eq])
# SNRdb1_eq = SNR_bound_instantaneous(Z_eq[:, [0]], C_axis[:, [0]].T, b_axis)
# SNRdb2_eq = SNR_bound_instantaneous(Z_eq[:, [1]], C_axis[:, [1]].T, b_axis)
# SNRdb_eq = SNR_bound_instantaneous(Z_eq, C_axis.T, b_axis)
# mean_axis_snr = power_to_dB((power_from_dB(SNRdb1_eq) + power_from_dB(SNRdb2_eq)) / 2)
# print(f"full: {SNRdb_eq:.2f} dB, mean per-axis: {mean_axis_snr:.2f} dB")
# ```
#
# SNR is (signal power) / (noise power). With axis-aligned loadings the Fisher
# information adds, so the full linear SNR is `(p1 + p2) / (i1 + i2)`, a
# sum-of-signals over sum-of-noises. That equals the average of `p1/i1` and
# `p2/i2` only when `p1 == p2` and `i1 == i2`. Unequal latent power (or unequal
# per-axis noise) tilts the weighted combination toward the stronger axis, so the
# simple average is wrong. Equalizing the power removes the `p1 != p2` tilt and
# the full SNR moves toward the mean.
#
# </details>

# %% [markdown]
# ## You can now...
#
# ...quantify how informative a Poisson population is about its latent (the
# Fisher-information SNR bound) and read a raster for its readout geometry -
# telling an oblique, mixed-selective random projection from a clean
# axis-aligned code.
#
# On your own recordings, is the population code closer to
# random-projection (every cell mixes several variables) or axis-aligned (each
# cell tuned to one)? Sort your raster by a candidate loading and see which
# picture it resembles.
#
# **Clustering the loading matrix:** One could try clustering the loading matrix
# $C$ to see if the population exhibits some block structure (representing
# functional cell types). However, keep in mind the **latent factor rotation issue**
# from the lecture notes. Because any orthogonal rotation of the
# latents can be compensated by counter-rotating the loading matrix without changing
# the model's likelihood, the coordinate orientation of $C$ is arbitrary. A random
# projection could be rotated to look axis-aligned or block-like, or vice versa,
# meaning any apparent block structure depends heavily on the coordinate basis.
#
# ## Mini project idea
#
# given a loading matrix $C$, find a factor rotation that maximizes the blockiness of $C$.
