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
# # Variational inference
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/04_variational_inference.ipynb)
#
# Variational inference turns an intractable posterior into an
# optimization problem - pick a family $q(z;\phi)$, then maximize the ELBO to make
# $q$ hug the true posterior $p(z\mid y)$.
#
# This notebook is a single sitting with no optional companion. It is the bridge
# from exact Bayesian inference to the amortized inference that powers XFADS.
# **Next core notebook:** XFADS on a simulated ring attractor, where this same
# ELBO is optimized over a neural network instead of two scalars.
#
# The math here (KL, ELBO, the reparameterization trick) is derived in full in the
# **variational inference** section of the lecture notes; here we build the
# intuition by doing it on a 1-D toy problem.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import trange  # env-agnostic: notebook widget on Colab, console bar otherwise

# fixed seed so the ELBO trace and fitted q are reproducible run to run
torch.manual_seed(20260714)

# %% [markdown]
# ## The object: an exact posterior on a grid
#
# Given a joint $p(y,z) = p(y\mid z)\,p(z)$ over a latent $z$ and observation $y$,
# Bayes' rule gives the posterior
# $$ p(z\mid y) = \frac{p(y\mid z)\,p(z)}{p(y)} \propto p(y\mid z)\,p(z). $$
# In 1-D we can just evaluate everything on a grid and normalize. (This grid trick
# does not scale to high dimensions - that is the whole reason we need VI.)

# %%
zr = np.linspace(-10., 10., 201)
dz = zr[1] - zr[0]

likelihood = np.exp(-(zr - 2)**2 / 6)         # Normal(mean=2, var=3) as a function of z
likelihood /= np.sum(likelihood) * dz

prior = np.exp(-np.abs(zr) / 3)               # Laplace(loc=0, scale=3)
prior /= np.sum(prior) * dz

# Bayes' rule on the grid (does not scale to high-dim; VI is the scalable answer)
posterior = prior * likelihood
posterior /= np.sum(posterior) * dz

plt.plot(zr, likelihood, label="$p(y\\mid z)$ likelihood")
plt.plot(zr, prior, label="$p(z)$ prior")
plt.plot(zr, posterior, '--', label="$p(z\\mid y)$ posterior")
plt.xlabel("z"); plt.ylabel("probability density"); plt.legend(); plt.grid()

# %% [markdown]
# ## Reminder: why we need approximate inference
#
# The grid works in 1-D, but for most likelihoods and priors the normalizer
# $p(y) = \int p(y\mid z)\,p(z)\,dz$ is intractable unless the two are *conjugate*
# (a rare convenience). In the neural setting the likelihood is Poisson spike
# counts and the prior is a temporal dynamics model, so exact inference is off the
# table. We instead *approximate* the posterior.
#
# We only assume we can evaluate and sample $p(z)$ and $p(y\mid z)$ efficiently.
# We use PyTorch distributions so everything is autodifferentiable.

# %%
lik = torch.distributions.normal.Normal(torch.tensor([2.0]), torch.tensor([np.sqrt(3)]))
pri = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([3.0]))

# %% [markdown]
# ## Step 1: choose a parametric family $q(z;\phi)$
#
# Pick a family of distributions to approximate the posterior,
# $$ q(z;\phi) \approx p(z\mid y), $$
# and let inference find the best parameter $\phi$. We need $q$ to be easy to
# sample from and to have a differentiable, easy-to-compute entropy. A Gaussian
# with a free mean and standard deviation is the simplest useful choice.

# %%
mu = torch.tensor([0.0], requires_grad=True)
sigma = torch.tensor([5.0], requires_grad=True)
q = torch.distributions.normal.Normal(mu, sigma)

# %% [markdown]
# ## Step 2: the loss is the (negative) ELBO
#
# VI measures approximation quality with the KL divergence $d_{\mathrm{KL}}(q\|p)$,
# which is non-negative and zero iff $q=p$. We want to minimize
# $d_{\mathrm{KL}}\big(q(z;\phi)\,\|\,p(z\mid y)\big)$, but $p(z\mid y)$ contains the
# intractable $p(y)$. The fix is the **Evidence Lower BOund (ELBO)**:
# $$ \mathrm{ELBO}(\phi) = \mathbb{E}_q[\log p(y\mid z)] + \mathbb{E}_q[\log p(z)] + H(q(z;\phi))
#    = \log p(y) - d_{\mathrm{KL}}\big(q(z;\phi)\,\|\,p(z\mid y)\big), $$
# where $H(\cdot)$ is the entropy. Because $\log p(y)$ is a constant in $\phi$,
# **maximizing the ELBO is the same as minimizing the KL** - the two differ only by
# that constant. The lecture notes give the full derivation, including the sign
# bookkeeping and the KL definition.
#
# We estimate the two expectations by Monte Carlo with $n$ samples $z_i \sim q(z;\phi)$:
# $$ \widehat{\mathrm{ELBO}}(\phi) = \frac{1}{n}\sum_i \big[\log p(y\mid z_i) + \log p(z_i)\big] + H(q(z;\phi)). $$

# %%
nMC = 100
Z = q.sample(torch.Size([nMC]))  # plain sample - does NOT carry gradients to mu/sigma (see Step 3)
ELBO = torch.mean(lik.log_prob(Z) + pri.log_prob(Z)) + q.entropy()  # the ELBO itself; loss = -ELBO

# %% [markdown]
# ## Step 3: the reparameterization trick
#
# There is a catch. A plain `q.sample()` returns numbers that are *constants* with
# respect to $\phi$ - the sampler blocks the gradient - so the data-term
# expectation cannot be differentiated through $\mu,\sigma$. The reparameterization
# trick rewrites a sample as a deterministic, differentiable function of $\phi$:
# draw $\epsilon \sim \mathcal{N}(0,1)$ once, then set $z = \mu + \sigma\,\epsilon$.
# Now gradients flow into $\mu$ and $\sigma$. (PyTorch exposes this as `.rsample()`
# we spell it out by hand to see the mechanism.)
#
# **Predict (before running):** if you kept the plain `q.sample()` from Step 2
# inside the training loop instead of the reparameterized draw, what happens to the
# gradients of $\mu$ and $\sigma$?
#
# <details>
# <summary>Solution</summary>
#
# The data-term gradient w.r.t. $\mu$ vanishes entirely (the samples are constants,
# not functions of $\phi$), and only the entropy term still updates $\sigma$. The
# fit fails to move the mean toward the posterior - which is exactly why the
# reparameterization trick is needed.
#
# </details>

# %% [markdown]
# **Fill one line:** turn the standard-normal noise `eps` into a differentiable
# sample of $q(z;\phi)$ by scaling by $\sigma$ and shifting by $\mu$.

# %%
sn = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
eps = sn.sample(torch.Size([nMC]))  # standard-normal noise, independent of phi
# BEGIN SOLUTION
Z = eps * sigma + mu  # reparameterization: differentiable in mu, sigma
# END SOLUTION
assert Z.requires_grad  # a reparameterized sample must carry gradients to mu, sigma

# %% [markdown]
# ## Step 4: maximize the ELBO with SGD
#
# Each step: draw reparameterized samples, form the negative ELBO, backprop, update.
# The loop below is the entire inference procedure.

# %%
optimizer = torch.optim.SGD([mu, sigma], lr=1e-3)  # try Adam instead (see the stretch below)

# %% [markdown]
# > **Stretch (optional):** `sigma` is optimized unconstrained here. At this lr / step
# > count it stays positive, but nothing stops a step from driving it $\le 0$, which
# > gives NaNs in `Normal.entropy()`/`log_prob`. Fix this by optimizing an unconstrained
# > parameter and mapping it through a strictly positive transform - an **exponential**,
# > $\sigma = e^{\rho}$, so the raw parameter $\rho$ can roam all of $\mathbb{R}$ while
# > $\sigma$ stays positive. Re-parameterize, re-run, and confirm the fit is unchanged.
# >
# > <details>
# > <summary>Solution</summary>
# >
# > ```python
# > rho = torch.tensor([np.log(5.0)], requires_grad=True)  # sigma = exp(rho); start at sigma=5
# > optimizer = torch.optim.SGD([mu, rho], lr=1e-3)
# > for k in trange(10000):
# >     sigma = torch.exp(rho)                       # always > 0, no NaN guard needed
# >     q = torch.distributions.normal.Normal(mu, sigma)
# >     Z = sn.sample(torch.Size([nMC])) * sigma + mu
# >     nELBO = -torch.mean(lik.log_prob(Z) + pri.log_prob(Z)) - q.entropy()
# >     optimizer.zero_grad(); nELBO.backward(); optimizer.step()
# > ```
# > The exponential map is a bijection from $\mathbb{R}$ to $(0,\infty)$, so gradient
# > descent on $\rho$ can never produce an invalid $\sigma$. The fitted posterior is the
# > same; you have only changed the coordinates you optimize in. This is exactly how
# > libraries parameterize scale parameters (and what XFADS does for its covariances).
# >
# > </details>

# %%
# Note: sigma is optimized unconstrained here (see the stretch above for the exp fix).
ELBO_trace = []
for k in trange(10000):
    Z = sn.sample(torch.Size([nMC])) * sigma + mu  # reparameterization trick, fresh noise each step
    nELBO = -torch.mean(lik.log_prob(Z) + pri.log_prob(Z)) - q.entropy()  # negative ELBO (minimize)
    ELBO_trace.append(-nELBO.item())
    optimizer.zero_grad()
    nELBO.backward()
    optimizer.step()

# %%
plt.plot(ELBO_trace)
plt.title("convergence"); plt.ylabel("ELBO"); plt.xlabel("gradient steps"); plt.grid()

# %%
q_plot = torch.distributions.normal.Normal(mu.detach(), sigma.detach())
zrt = torch.tensor(zr)
plt.plot(zr, np.exp(lik.log_prob(zrt).numpy()), label="likelihood")
plt.plot(zr, np.exp(pri.log_prob(zrt).numpy()), label="prior")
plt.plot(zr, posterior, '--', label="true posterior")
plt.plot(zr, np.exp(q_plot.log_prob(zrt).numpy()), label="variational posterior $q$")
plt.xlabel("z"); plt.ylabel("probability density"); plt.legend(); plt.grid()

# %% [markdown]
# > **Stretch (optional):** swap the Laplace prior (the `pri` in Step 1) for a
# > `Normal(0, 3)`. Now the prior and the Gaussian likelihood are conjugate, so the
# > posterior is itself exactly Gaussian. Re-run and overlay: does VI recover the
# > analytic posterior exactly? As a second tweak, switch the optimizer from SGD to
# > Adam and compare the convergence traces.
#
# <details>
# <summary>Solution</summary>
#
# ```python
# pri = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([3.0]))
# # ... rebuild the grid `prior`/`posterior` with the Gaussian prior, rerun Step 4 ...
# optimizer = torch.optim.Adam([mu, sigma], lr=1e-2)
# ```
#
# With a conjugate Gaussian prior the true posterior is Gaussian, so a Gaussian $q$
# can match it exactly (up to Monte Carlo noise): the variational posterior lands
# right on the analytic one. Adam typically converges in far fewer steps than SGD.
#
# </details>

# %% [markdown]
# ## You can now...
#
# ...take any model where you can evaluate $\log p(y\mid z)$ and $\log p(z)$, pick a
# reparameterizable $q(z;\phi)$, and fit an approximate posterior by maximizing the
# ELBO with gradient descent. You turned Bayesian inference into optimization.
#
# **The road to VAEs and XFADS.** As written, every new observation $y$ needs its
# own optimization over $\phi$. But that optimization is itself a function -
# observation in, optimal $\phi$ out - so we can *amortize* it by training a neural
# network $q_\phi(z\mid y)$ to output the variational parameters directly. That is
# the recognition model / encoder of a **variational autoencoder**. XFADS applies
# exactly this amortized ELBO to state-space models, where $p(z)$ is a temporal
# dynamics prior and $p(y\mid z)$ is Poisson.