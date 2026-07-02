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
# # Intuitions on State Space Model
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/00_state_space_intuition.ipynb)
#
# In this notebook, we will play around with the parameters of a state space model and generate various spike trains.
# These population activity patterns will be fun to look at. (hopefully)

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## A simple 1-D latent process
#
# For illustration, we will use a sinusoid as the 1-D latent process.
# $$ x(t) = sin(2\pi f\cdot t) $$
# In this example, $x(t)$ represents the instantaneous state of the neural population of interest.

# %%
# simulate a simple latent process
nT = 1000
T = 10
frq = 0.3
tr = np.linspace(0, T, nT)
dt = tr[1] - tr[0]
x = np.sin(2 * np.pi * frq * tr) # generate a sinusoid over time

# %%
fig = plt.figure(figsize=(10, 3))
plt.plot(tr, x); plt.title('1-D latent process'); plt.xlabel('time');

# %% [markdown]
# ## One Poisson neuron driven by the latent process
#
# We will generate spike trains from an inhomogeneous Poisson process with a time varying firing rate function $\lambda(t)$.
# The spike count $y(t)$ in a small time bin of size $\Delta$ is distributed as a Poisson distribution:
# $$ y(t) \sim \text{Poisson}(\Delta\lambda(t)) $$
#
# Importantly, the firing rate will be a function of $x(t)$, but not of past $x$ nor past $y$.
# $$ \lambda(t) = g(x(t)) $$
# The only constraint for $g(\cdot)$ is that the resulting firing rate has to be non-negative.
# A mathematically convenient function is the exponential function.
#
# $$ \lambda(t) = \exp(a x(t) + b) = \exp(b)\exp(a x(t)) $$

# %%
a = 5
b = -3
lam = np.exp(a * x + b)
y = np.random.poisson(lam*dt)

plt.figure(figsize=(10, 2))
plt.plot(tr, lam, label='firing rate');
plt.eventplot(np.nonzero(y)[0]/nT*T, lw=0.5, color='k', label='spikes')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.legend();

# %% [markdown]
# ## A population of Poisson neurons driven by a common 1-D latent process
#
# We can have more than one neuron that's driven by the same latent process.
# This way, we an have more observation dimensions than the latent state space dimension.
# Let's given them a random amount of "drive".

# %%
x = x[:, None]  # shape: (nT, 1) for matrix vector multiplications

# %%
nNeuron = 200
C = 2 * np.random.randn(nNeuron, 1)
b = -2.0 + np.random.rand(1, nNeuron)
lam = np.exp(x @ C.T + b)
y = np.random.poisson(lam * dt)


# %% [markdown]
# We can make spike raster plot. But since we know the amount of drive (each value in $C$), we can sort the neurons accordingly as well.

# %%
cidx = np.argsort(C[:, 0])

raster = []
rasterSorted = []
for k in range(nNeuron):
    raster.append(np.nonzero(y[:, k])[0] / nT * T)
    rasterSorted.append(np.nonzero(y[:, cidx[k]])[0] / nT * T)

plt.subplots(1, 2, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.eventplot(raster, lw=0.5, color='k', label='spikes')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.title('raster plot'); plt.ylabel('neurons')
plt.subplot(1, 2, 2)
plt.eventplot(rasterSorted, lw=0.5, color='k', label='spikes')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.title('raster plot (again)'); plt.ylabel('sorted neurons')


# %% [markdown]
# ## Signal-to-Noise Ratio of population spike trains
#
# We use the `neurofisherSNR` package (submodule at `external/neurofisherSNR/`) to estimate the Fisher-information upper bound on SNR — how much information the spike trains carry about the latent variable per time bin.
#

# %%
from neurofisherSNR.snr import SNR_bound_instantaneous
from neurofisherSNR.utils import power_to_dB, power_from_dB


# %%
SNRdb = SNR_bound_instantaneous(x, C.T, b)
print(f"{SNRdb:.2f} dB")  # decibel is a logarithmic unit


# %% [markdown]
# ## 2D latent space example
#
# Here we will build a 2D manifold with two independent processes.
# The first latent dimension will be same as above, but we will add $x_2(t)$ as a sawtooth function:
# $$ x_2(t) = t \,\, \text{mod} \, 1 $$

# %%
x2 = 1.5 * ((tr % 1) - 0.5)[:, np.newaxis]
X = np.hstack([x, x2])  # shape [nT, dLatent]
dLatent = X.shape[1]


# %%
plt.subplots(2,1,figsize=(10,4))
plt.subplot(2,1,1);
plt.plot(tr, x ); plt.ylabel('first latent dim'); plt.xlabel('time')
plt.subplot(2,1,2);
plt.plot(tr, x2); plt.ylabel('second latent dim'); plt.xlabel('time')

# %% [markdown]
# Now that we have more than 1 latent dimension, we face a choice of how neurons relate to each of the latent dimes.

# %% [markdown]
# ### Random projection observation
#
# Random projection assumes that each neuron is deriven by all the latent dimensions with a random amount.
# Under this assumption, the neural manifold is likely oblique to the axes, i.e., the neuron will be modulated by changes in any direction in the latent state space.
# Theoretical analysis of [Gao & Ganguli 2015] assumes random projections and showed that not many neurons need to be sampled (observed) to recover the manifold structure.
# In addition, the so-called *mixed-selectivity* appears as a result.
#
# - Gao, P., & Ganguli, S. (2015). On Simplicity and Complexity in the Brave New World of Large-Scale Neuroscience. Current Opinion in Neurobiology, 32, 148–155.

# %%
C = 0.8 * np.random.randn(nNeuron, dLatent)  # random projection
b = 0.1 * np.random.randn(nNeuron) + np.log(5)
lam = np.exp(X @ C.T + b)
y = np.random.poisson(lam * dt)

SNRdb = SNR_bound_instantaneous(X, C.T, b)
print(f"{SNRdb:.2f} dB")


# %%
cidx1 = np.lexsort((C[:,0], C[:,1]), axis=0)
cidx2 = np.lexsort((C[:,1], C[:,0]), axis=0)

# %%
raster = []; rasterSorted1 = []; rasterSorted2 = []
for k in range(nNeuron):
    raster.append(np.nonzero(y[:, k])[0] / nT * T)
    rasterSorted1.append(np.nonzero(y[:, cidx1[k]])[0] / nT * T)
    rasterSorted2.append(np.nonzero(y[:, cidx2[k]])[0] / nT * T)

plt.subplots(1,3, figsize=(10, 3))
plt.subplot(1,3,1)
plt.eventplot(raster, lw=0.5, color='k', label='spikes')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.title('raster plot'); plt.ylabel('neurons');
plt.subplot(1,3,2)
plt.eventplot(rasterSorted1, lw=0.5, color='k', label='spikes')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.title('raster plot (1)'); plt.ylabel('sorted neurons');
plt.subplot(1,3,3)
plt.eventplot(rasterSorted2, lw=0.5, color='k', label='spikes')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.title('raster plot (2)'); plt.ylabel('sorted neurons');

# %% [markdown]
# ### Axis aligned observation
#
# Biologists have long loved neurons that are tuned specifically for a particular feature but not modulated by others.
# In our context, the neurons will be either driven by the first dimension or the second dimension of the latent process.
# Recent paper argues that this is optimal [Whittington et al. 2022].
#
#  - Whittington, J. C. R., Dorrell, W., Ganguli, S., & Behrens, T. E. J. (2022). Disentangling with Biological Constraints: A Theory of Functional Cell Types. In arXiv [q-bio.NC]. arXiv. http://arxiv.org/abs/2210.01768

# %%
bidx = np.random.rand(nNeuron) < 0.5
C = 0.8 * np.random.randn(nNeuron, dLatent)
C[bidx, 0] = 0
C[~bidx, 1] = 0
b = 0.1 * np.random.randn(nNeuron) + np.log(5)
b[bidx] += 1.5  # boost firing rate for the 2nd latent dim
lam = np.exp(X @ C.T + b)
y = np.random.poisson(lam * dt)

SNRdb1 = SNR_bound_instantaneous(X[:, [0]], C[:, [0]].T, b)
SNRdb2 = SNR_bound_instantaneous(X[:, [1]], C[:, [1]].T, b)
print(f"{SNRdb1:.2f} dB, {SNRdb2:.2f} dB")
SNRdb = SNR_bound_instantaneous(X, C.T, b)
print(f"{SNRdb:.2f} dB = ({SNRdb1:.2f} dB + {SNRdb2:.2f} dB)/2 = {power_to_dB((power_from_dB(SNRdb1) + power_from_dB(SNRdb2)) / 2):.2f} dB")


# %%
cidx = np.lexsort((C[:, 1], C[:, 0]), axis=0)

raster = []
rasterSorted = []
for k in range(nNeuron):
    raster.append(np.nonzero(y[:, k])[0] / nT * T)
    rasterSorted.append(np.nonzero(y[:, cidx[k]])[0] / nT * T)

plt.subplots(1, 2, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.eventplot(raster, lw=0.5, color='k')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.title('raster plot'); plt.ylabel('neurons')
plt.subplot(1, 2, 2)
plt.eventplot(rasterSorted, lw=0.5, color='k')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.title('raster plot (sorted by C entries)'); plt.ylabel('neurons')


# %%
# sort by which C row entry is dominant, then loading strength (from C only)
dominant_dim = np.argmax(np.abs(C), axis=1)
active_loading = np.abs(C).max(axis=1)
cidx = np.lexsort((active_loading, dominant_dim))

raster = []
rasterSorted = []
for k in range(nNeuron):
    raster.append(np.nonzero(y[:, k])[0] / nT * T)
    rasterSorted.append(np.nonzero(y[:, cidx[k]])[0] / nT * T)

plt.subplots(1, 2, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.eventplot(raster, lw=0.5, color='k', label='spikes')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.title('raster plot'); plt.ylabel('neurons')
plt.subplot(1, 2, 2)
plt.eventplot(rasterSorted, lw=0.5, color='k', label='spikes')
plt.xlim(0, T); plt.xlabel('time'); plt.yticks([]); plt.title('raster plot (sorted by cell type)'); plt.ylabel('sorted neurons')

# %% [markdown]
# ## Some dynamical law governing the latent states
#
# So far, the latent states were given and not generated in a Markovian manner, that is, given the current state $x(t)$, the future states do not depend on $x(t-1)$ (or further past).
# In general, in discrete time, a dynamical law can be represented as a dynamical system:
# $$ \frac{dx}{dt} = \dot{x} = f(x(t)) $$
# where $f$ is a smooth function that represents the vector field.

# %%
#generate some data
# !python code_pack/generate_vdp_data.py

# %% [markdown]
# ### van der Pol oscillator
# Van der pol oscillator is defined as 2D dimensional first order differential equations:
#     $$ \dot{x} = y$$
#     $$ \dot{y} = \mu(1-x^2)y -x $$
#
# For our simulations we take a discrete time grid for convenience.
# We use an Euler integration of a Van der Pol oscillator with noisy transitions with $\mu=1.5$, $\tau_1=0.1$, $\tau_2=0.1$, and $\sigma=0.1$:
#
# $$ x_{t+1,1} = x_{t,1} + \tau_1^{-1} \Delta x_{t,2} + \sigma \epsilon$$
# $$ x_{t+1,2} = x_{t,2} + \tau_2^{-1} \Delta(\mu (1-x_{t,1})^2 x_{t,2} - x_{t,1}) + \sigma \epsilon$$
#
# For the sake of brevity in the notebook, the generation code is provided in `code_pack/generate_vdp_data.py` and we only load the saved data here.

# %% jupyter={"outputs_hidden": false}
import h5py

from code_pack.plotting import plot_two_d_vector_field_from_data, raster_to_events
from code_pack.generate_vdp_data import generate_noisy_van_der_pol

# loading data from ./data/vdp_noisy.h5
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
# ### Visualizationing trajectories

# %% jupyter={"outputs_hidden": false}
# plotting trajectories of the dataset
fig, ax = plt.subplots(1, 1, figsize=(5,5))
_ = ax.plot(X[0,:,0], X[0,:,1])
ax.scatter(X[0, 0, 0], X[0, 0, 1], marker='o', color='red', zorder=10, s=100, label='start')
ax.scatter(X[0, -1, 0], X[0, -1, 1], marker='x', color='red', zorder=10, s=100, label='end')

# system_parameters_copy = copy.deepcopy(system_parameters)
system_parameters['sigma'] = 0.0
dynamic_func = lambda inp : generate_noisy_van_der_pol(inp, np.array([0.0, 5e-3]), system_parameters)
axs_range = {'x_min':-1.5, 'x_max':1.5, 'y_min':-1.5, 'y_max':1.5}
plot_two_d_vector_field_from_data(dynamic_func, ax, axs_range)

ax.legend()
ax.set_title('sample trajectory (true state)');

# %% [markdown]
# ### Effect of tuning function
#
# TODO: explain different inverse link functions (soft plus)

# %% jupyter={"outputs_hidden": false}
C_tilde = np.array(data['C_tilde'])
idx = np.lexsort((C_tilde[:, 0], C_tilde[:, 1]), axis=0)  # sort the loading

# showing the spike raster generated from noisy Vdp
fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharex=True, sharey=True)
events = raster_to_events(np.array(data['Y'])[0, :, :])
events_softplus = raster_to_events(np.array(data['Y_softplus'])[0, :, :])
events_axis_aligned = raster_to_events(np.array(data['Y_axis'])[0, :, idx].transpose())
axs[0].eventplot(events, linewidths=0.5, color='k');
axs[1].eventplot(events_softplus, linewidths=0.5, color='k');
axs[2].eventplot(events_axis_aligned, linewidths=0.5, color='k');
axs[0].set_title(f'$\exp()$');
axs[1].set_title(f'softplus$()$');
axs[2].set_title(f'axis aligned');
axs[0].set_xlabel("Time");
axs[0].set_ylabel("Neuron");

# %% [markdown]
# ## What's next?
#
# Now we understand better the generative process of the model. But what we are interested is the opposite direction, that is, how do we infer the model parameters given just the observations (neural data)? This is the statistical inference problem of interest.
