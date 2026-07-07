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

# %%
import numpy as np
import matplotlib.pyplot as plt

from neurofisherSNR.snr import SNR_bound_instantaneous
from neurofisherSNR.utils import power_to_dB, power_from_dB
from code_pack.plotting import raster_to_events

# fixed seed so the SNR numbers and rasters are reproducible run to run
rng = np.random.default_rng(20260707)

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

nNeuron = 200

# %% [markdown]
# ## Signal-to-noise ratio of population spike trains
#
# We use the `neurofisherSNR` package to estimate the Fisher-information upper
# bound on SNR: how much the spike trains carry about the latent per time bin.
# The Fisher-information view of Poisson observations (and this SNR bound) is
# developed in the lecture notes' Poisson / exponential-family section
# (`sec:expfam`). Decibels are a logarithmic unit.

# %%
C = 2 * rng.standard_normal((nNeuron, 1))
b = -2.0 + rng.random((1, nNeuron))

SNRdb = SNR_bound_instantaneous(z, C.T, b)
print(f"{SNRdb:.2f} dB")  # decibel is a logarithmic unit

# %% [markdown]
# ## A 2-D latent for the readout-geometry question
#
# Add a centered sawtooth as a second latent (as in notebook 00),
# $z_2(t) = 1.5\,((t \bmod 1) - 0.5)$, so we have two dimensions to read out.

# %%
z2 = 1.5 * ((tr % 1) - 0.5)[:, np.newaxis]
Z = np.hstack([z, z2])  # shape [nT, dLatent]
dLatent = Z.shape[1]

# %% [markdown]
# ### Random projection observation
#
# Random projection assumes each neuron is driven by *all* latent dimensions by a
# random amount. The neural manifold is then oblique to the axes: a neuron
# responds to changes in any direction of the latent state space. [Gao & Ganguli
# 2015] showed that under random projections few neurons need be sampled to
# recover the manifold, and *mixed selectivity* appears as a byproduct.
#
# - Gao, P., & Ganguli, S. (2015). On Simplicity and Complexity in the Brave New
#   World of Large-Scale Neuroscience. Current Opinion in Neurobiology, 32,
#   148-155.

# %%
C = 0.8 * rng.standard_normal((nNeuron, dLatent))  # random projection
b = 0.1 * rng.standard_normal(nNeuron) + np.log(5)
lam = np.exp(Z @ C.T + b)
y = rng.poisson(lam * dt)

SNRdb = SNR_bound_instantaneous(Z, C.T, b)
print(f"{SNRdb:.2f} dB")

# %%
cidx1 = np.lexsort((C[:, 0], C[:, 1]), axis=0)
cidx2 = np.lexsort((C[:, 1], C[:, 0]), axis=0)

plt.subplots(1, 3, figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.eventplot(raster_to_events(y), lw=0.5, color='k')
plt.xlabel('time bin'); plt.yticks([]); plt.title('raster plot'); plt.ylabel('neurons');
plt.subplot(1, 3, 2)
plt.eventplot(raster_to_events(y[:, cidx1]), lw=0.5, color='k')
plt.xlabel('time bin'); plt.yticks([]); plt.title('sorted by dim 1'); plt.ylabel('sorted neurons');
plt.subplot(1, 3, 3)
plt.eventplot(raster_to_events(y[:, cidx2]), lw=0.5, color='k')
plt.xlabel('time bin'); plt.yticks([]); plt.title('sorted by dim 2'); plt.ylabel('sorted neurons');

# %% [markdown]
# ### Axis-aligned observation
#
# Biologists have long favored neurons tuned to one feature and unmodulated by
# others. Here each neuron is driven by *either* the first or the second latent
# dimension. A recent paper argues this is optimal [Whittington et al. 2022].
#
# - Whittington, J. C. R., Dorrell, W., Ganguli, S., & Behrens, T. E. J. (2022).
#   Disentangling with Biological Constraints: A Theory of Functional Cell Types.
#   arXiv:2210.01768.

# %%
bidx = rng.random(nNeuron) < 0.5
C = 0.8 * rng.standard_normal((nNeuron, dLatent))
C[bidx, 0] = 0
C[~bidx, 1] = 0
b = 0.1 * rng.standard_normal(nNeuron) + np.log(5)
b[bidx] += 1.5  # boost firing rate for the neurons reading the 2nd latent dim
lam = np.exp(Z @ C.T + b)
y = rng.poisson(lam * dt)

# per-axis SNR bounds, using only the neurons/latent for that axis
SNRdb1 = SNR_bound_instantaneous(Z[:, [0]], C[:, [0]].T, b)
SNRdb2 = SNR_bound_instantaneous(Z[:, [1]], C[:, [1]].T, b)
SNRdb = SNR_bound_instantaneous(Z, C.T, b)
print(f"axis 1: {SNRdb1:.2f} dB,  axis 2: {SNRdb2:.2f} dB")
print(f"full:   {SNRdb:.2f} dB")
# For axis-aligned loadings the Fisher information is block-diagonal, so the
# information about the two independent axes adds. The full-population SNR bound
# therefore sits between the two per-axis bounds - it is NOT their average
# (that would require equal latent power and equal per-axis noise, which does not
# hold here: the sine and the sawtooth carry different power).

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
# print(power_to_dB((power_from_dB(SNRdb1) + power_from_dB(SNRdb2)) / 2))
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

# %%
# sort by which C row entry is dominant, then by loading strength (from C only)
dominant_dim = np.argmax(np.abs(C), axis=1)
active_loading = np.abs(C).max(axis=1)
cidx = np.lexsort((active_loading, dominant_dim))

plt.subplots(1, 2, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.eventplot(raster_to_events(y), lw=0.5, color='k')
plt.xlabel('time bin'); plt.yticks([]); plt.title('raster plot'); plt.ylabel('neurons')
plt.subplot(1, 2, 2)
plt.eventplot(raster_to_events(y[:, cidx]), lw=0.5, color='k')
plt.xlabel('time bin'); plt.yticks([]); plt.title('raster plot (sorted by cell type)'); plt.ylabel('sorted neurons')

# %% [markdown]
# ## You can now...
#
# ...quantify how informative a Poisson population is about its latent (the
# Fisher-information SNR bound) and read a raster for its readout geometry -
# telling an oblique, mixed-selective random projection from a clean
# axis-aligned code.
#
# **Transfer prompt:** on your own recordings, is the population code closer to
# random-projection (every cell mixes several variables) or axis-aligned (each
# cell tuned to one)? Sort your raster by a candidate loading and see which
# picture it resembles. Then return to the core sequence: the *Latent Variable
# Models* notebook infers $z$ and $C$ from spikes like these.
