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
# # Intuitions on the State Space Model
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/00_state_space_intuition.ipynb)
#
# **Takeaway:** a low-dimensional latent process $z(t)$, passed through an
# exponential rate link, generates the spike trains of a whole population of
# Poisson neurons - and the latent can itself carry dynamics.
#
# Where to go from here:
#
# - **Optional companion:** `01_snr_and_readout_geometry.ipynb` - how much the
#   population tells you about $z$ (Fisher-information SNR), and how random vs
#   axis-aligned readouts shape the raster.
# - **Next core notebook:** *Latent Variable Models* - the inverse problem, i.e.
#   inferring the latent and the model from the spikes alone.

# %%
import numpy as np
import matplotlib.pyplot as plt

# fixed seed so the rasters below are reproducible run to run
rng = np.random.default_rng(20260707)

# %% [markdown]
# ## A simple 1-D latent process
#
# For illustration we use a sinusoid as the 1-D latent process,
# $$ z(t) = \sin(2\pi f\cdot t). $$
# Here $z(t)$ is the instantaneous state of the neural population of interest.
#
# Note: these latents are *hand-chosen deterministic* signals, picked so the
# generated spikes are easy to eyeball. In the lecture notes the latent is
# instead a random variable, $z\sim\mathcal{N}(0, I_d)$ (see the notation and
# PPCA sections, `sec:ppca`); here we prescribe $z(t)$ by hand.

# %%
# simulate a simple latent process
nT = 1000
T = 10
frq = 0.3
tr = np.linspace(0, T, nT)
dt = tr[1] - tr[0]
z = np.sin(2 * np.pi * frq * tr) # generate a sinusoid over time

# %%
fig = plt.figure(figsize=(10, 3))
plt.plot(tr, z); plt.title('1-D latent process'); plt.xlabel('time');

# %% [markdown]
# ## One Poisson neuron driven by the latent process
#
# We generate spikes from an inhomogeneous Poisson process with a time-varying
# firing rate $\lambda(t)$. The spike count $y(t)$ in a bin of size $\Delta$ is
# $$ y(t) \sim \text{Poisson}(\Delta\lambda(t)). $$
#
# The rate is a function of $z(t)$ alone (not of past $z$ nor past $y$),
# $\lambda(t) = g(z(t))$, and $g(\cdot)$ only needs to stay non-negative. The
# exponential inverse-link is the convenient choice,
# $$ \lambda(t) = \exp(a\, z(t) + b), $$
# so $\exp(b)$ is the baseline rate (at $z=0$) and $a$ is the gain. This is the
# Poisson / exponential-family observation model derived in the lecture notes
# (`sec:expfam`).
#
# > **Exercise (fill one line):** set the baseline firing rate $\exp(b)$ to
# > 2 Hz. Replace the placeholder `b` below.
#
# <details>
# <summary>Solution</summary>
#
# ```python
# b = np.log(2)  # baseline rate exp(b) = 2 Hz
# ```
#
# </details>

# %%
a = 5
b = 0.0  # YOUR CODE HERE: choose b so the baseline rate exp(b) equals 2 Hz
lam = np.exp(a * z + b)
y = rng.poisson(lam * dt)

plt.figure(figsize=(10, 2))
plt.plot(tr, lam, label='firing rate');
plt.eventplot(np.nonzero(y)[0]/nT*T, lw=0.5, color='k', label='spikes')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.legend();

# %% [markdown]
# ## A population of Poisson neurons driven by a common 1-D latent
#
# More than one neuron can be driven by the same latent process, giving us more
# observation dimensions than latent dimensions. Each neuron gets a random amount
# of "drive" through its loading $C$,
# $$ \lambda(t) = \exp(z(t)\, C^\top + b). $$

# %%
z = z[:, None]  # shape: (nT, 1) for matrix-vector multiplications

# %%
nNeuron = 200
C = 2 * rng.standard_normal((nNeuron, 1))
b = -2.0 + rng.random((1, nNeuron))
lam = np.exp(z @ C.T + b)
y = rng.poisson(lam * dt)

# %% [markdown]
# We can make a spike raster. Since we know each neuron's drive (its value in
# $C$), we can also sort the neurons by it - which reveals a travelling band.

# %%
cidx = np.argsort(C[:, 0])

# raster_to_events(y) turns a (time x neuron) count matrix into a list of
# spike-time-bin arrays, one per neuron (see code_pack/plotting.py). We reuse it
# everywhere instead of hand-rolling the nonzero loop.
from code_pack.plotting import raster_to_events

plt.subplots(1, 2, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.eventplot(raster_to_events(y), lw=0.5, color='k')
plt.xlabel('time bin'); plt.yticks([]); plt.title('raster plot'); plt.ylabel('neurons')
plt.subplot(1, 2, 2)
plt.eventplot(raster_to_events(y[:, cidx]), lw=0.5, color='k')
plt.xlabel('time bin'); plt.yticks([]); plt.title('raster plot (sorted by drive)'); plt.ylabel('sorted neurons')

# %% [markdown]
# ## A 2-D latent space
#
# Nothing forces the latent to be one-dimensional. Let us add a second,
# independent latent: a centered sawtooth
# $$ z_2(t) = 1.5\,\bigl((t \bmod 1) - 0.5\bigr). $$
# The centering makes $z_2$ zero-mean, matching the zero-mean latent convention
# in the notes (and keeping any later power/SNR normalization meaningful).

# %%
z2 = 1.5 * ((tr % 1) - 0.5)[:, np.newaxis]
Z = np.hstack([z, z2])  # shape [nT, dLatent]
dLatent = Z.shape[1]

# %%
plt.subplots(2, 1, figsize=(10, 4))
plt.subplot(2, 1, 1);
plt.plot(tr, z ); plt.ylabel('first latent dim'); plt.xlabel('time')
plt.subplot(2, 1, 2);
plt.plot(tr, z2); plt.ylabel('second latent dim'); plt.xlabel('time')

# %% [markdown]
# With two latent dimensions we now face a choice of *how* each neuron reads out
# the latent space (all dimensions at once, or one axis each). That readout
# geometry - and how much information the population carries about $z$ - is the
# subject of the optional companion `01_snr_and_readout_geometry.ipynb`.

# %% [markdown]
# ## A dynamical law governing the latent states
#
# So far the latents were prescribed, not *generated*: given $z(t)$ the future
# did not depend on the past. In general a latent can obey its own dynamical law.
# In continuous time,
# $$ \dot{z} = f(z(t)), $$
# where $f$ is a smooth vector field. This Markovian latent is exactly what the
# smoothing / forecasting machinery in the notes exploits (`sec:smoothing`).

# %%
#generate some data (heavy; run once, then just load below)
# !python code_pack/generate_vdp_data.py

# %% [markdown]
# ### van der Pol oscillator
#
# The van der Pol oscillator is a 2-D first-order system with state
# $(z_1, z_2)$:
# $$ \dot{z}_1 = z_2, \qquad \dot{z}_2 = \mu\,(1 - z_1^2)\, z_2 - z_1. $$
# (We rename the two state coordinates $z_1, z_2$ to avoid clashing with $y$, the
# spike counts.)
#
# For simulation we Euler-integrate a *noisy* version on a discrete time grid
# ($\mu=1.5$). The exact discrete update - including a coordinate rescaling and
# the transition-noise scaling - lives in `code_pack/generate_vdp_data.py`, which
# is the source of truth; here we only load the saved data.

# %% jupyter={"outputs_hidden": false}
import h5py

from code_pack.plotting import plot_two_d_vector_field_from_data
from code_pack.generate_vdp_data import generate_noisy_van_der_pol

# loading data from ./vanderpol/data/poisson_obs.h5
file_name = "vanderpol/data/poisson_obs.h5"

# dynamics parameters
data = h5py.File(file_name, 'r')
system_parameters = {}
system_parameters['mu'] = data['mu']
system_parameters['tau_1'] = data['tau_1']
system_parameters['tau_2'] = data['tau_2']
system_parameters['sigma'] = data['sigma']
system_parameters['scale'] = np.array(data['scale'])

Y = np.array(data['Y'])
X = np.array(data['X'])
C = np.array(data['C'])
b = np.array(data['bias'])

n_trials = Y.shape[0]
n_latents = X.shape[2]
n_neurons = Y.shape[2]
n_time_bins = Y.shape[1]

# %% [markdown]
# ### Visualizing trajectories

# %% jupyter={"outputs_hidden": false}
# plotting trajectories of the dataset
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
_ = ax.plot(X[0, :, 0], X[0, :, 1])
ax.scatter(X[0, 0, 0], X[0, 0, 1], marker='o', color='red', zorder=10, s=100, label='start')
ax.scatter(X[0, -1, 0], X[0, -1, 1], marker='x', color='red', zorder=10, s=100, label='end')

# overlay the (noise-free) vector field so the trajectory sits on its flow
system_parameters['sigma'] = 0.0
dynamic_func = lambda inp : generate_noisy_van_der_pol(inp, np.array([0.0, 5e-3]), system_parameters)
axs_range = {'x_min':-1.5, 'x_max':1.5, 'y_min':-1.5, 'y_max':1.5}
plot_two_d_vector_field_from_data(dynamic_func, ax, axs_range)

ax.legend()
ax.set_title('sample trajectory (true state)');

# %% [markdown]
# ### Effect of the tuning (inverse-link) function
#
# The same latent trajectory produces different-looking rasters depending on the
# inverse-link that turns state into rate: `exp` vs `softplus`, and dense
# (random-projection) vs axis-aligned loadings. Compare the three below.

# %% jupyter={"outputs_hidden": false}
C_tilde = np.array(data['C_tilde'])
idx = np.lexsort((C_tilde[:, 0], C_tilde[:, 1]), axis=0)  # sort the loading

# spike raster generated from the noisy van der Pol latent
fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharex=True, sharey=True)
events = raster_to_events(np.array(data['Y'])[0, :, :])
events_softplus = raster_to_events(np.array(data['Y_softplus'])[0, :, :])
events_axis_aligned = raster_to_events(np.array(data['Y_axis'])[0, :, idx].transpose())
axs[0].eventplot(events, linewidths=0.5, color='k');
axs[1].eventplot(events_softplus, linewidths=0.5, color='k');
axs[2].eventplot(events_axis_aligned, linewidths=0.5, color='k');
axs[0].set_title(r'$\exp()$');
axs[1].set_title(r'softplus$()$');
axs[2].set_title('axis aligned');
axs[0].set_xlabel("Time");
axs[0].set_ylabel("Neuron");

# %% [markdown]
# ## You can now...
#
# ...build a state-space generative model from scratch: pick a low-D latent
# $z(t)$ (prescribed or dynamical), map it through an exponential rate link with
# a loading $C$, and sample Poisson spike trains for a whole population.
#
# **Transfer prompt:** take a latent that matters for *your* system - a phase, a
# heading, a decision variable - and generate a population raster from it. Does
# the raster you get resemble the recordings you actually see?
#
# What we did here is the *forward* (generative) direction. The rest of the
# workshop tackles the inverse problem: given only the spikes, infer the latent
# and the model. That is statistical inference over latent variables
# (`sec:vi`, `sec:amortized`) and, for dynamical latents, XFADS (`sec:xfads`).
# Continue with the *Latent Variable Models* notebook, or detour through the
# optional `01_snr_and_readout_geometry.ipynb` first.
