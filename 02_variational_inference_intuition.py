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
# # Introduction to variational inference
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/catniplab/latent_dynamics_workshop/blob/main/02_variational_inference_intuition.ipynb)
#
# Given a probabilistic model where the joint distribution of latent states $x$ and observation $y$ is given by,
# $ p(y,x) = p(y|x)p(x) $, we would like to infer the posterior distribution,
# $$ p(x|y) = \frac{p(x,y)}{p(y)} \propto p(y|x)p(x) $$
#
# Let's use a 1-D grid to evaluate these distributions.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

xr = np.linspace(-10., 10., 201)
dx = xr[1] - xr[0]

likelihood = np.exp(-(xr - 2)**2/6)
likelihood /= sum(likelihood) * dx

#prior = np.array(xr < 0).astype(float) + np.array(xr >= 0).astype(float) * 0.1
prior = np.exp(-np.abs(xr)/3)
prior /= sum(prior) * dx

# Bayes Rule - note that this approximate inference on a grid doesn't scale to high-dim
posterior = prior * likelihood
posterior /= sum(posterior) * dx

plt.plot(xr, likelihood, label="p(y|x) likelihood")
plt.plot(xr, prior, label="p(x) prior dist.")
plt.plot(xr, posterior, '--', label="p(x|y) posterior dist.")
plt.xlabel("x"); plt.ylabel("probability density"); plt.legend(); plt.grid();

# %% [markdown]
# This problem is intractible for most likelihood $p(y|x)$ and prior $p(x)$ unless they are *conjugate*, a rare mathematically convienient coincidence.
# Therefore, we seek an approximate inference method.
#
# We assume that we can evaluate $p(x)$ and $p(y|x)$ and also sample from both distributions, computationally efficiently.
# We will use PyTorch to represent these distributions, so that we can autodifferentiate.

# %%
lik = torch.distributions.normal.Normal(torch.tensor([2.0]), torch.tensor([np.sqrt(3)]))
pri = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([3.0]))

# %% [markdown]
# ### Step 1: choose a parametric family of distributions
#
# Let's choose a parametric family of distributions to approximate the desired posterior distribution $p(x|y)$.
# We denote the approximate distribution with
# $$ q(x;\phi) \approx p(x|y) $$
# where the goal of the inference is to find the "best" parameter $\phi \in \Phi$.
# We assume that it is easy to evaluate and sample from $q$, entropy of $q$ is differentiable with respect to $\phi$ and is easy to compute.

# %%
mu = torch.tensor([0.0],requires_grad=True); sigma = torch.tensor([5.0],requires_grad=True)
q = torch.distributions.normal.Normal(mu, sigma)

# %% [markdown]
# ## Step 2: define the loss function
# Variational inference turns the inference problem into an optimization problem by defining the loss function that measures the quality of approximation based on a divergence measure.
# A divergence measure $d(p,q)$ is non-negative, and returns 0 if and only if the two distributions are identical.
# A typical choice for variational inference is the Kullback-Leibler (KL) divergence:
# $$ d_{\text{KL}}(p || q) = \int \log\left(\frac{dp}{dq}\right) dp $$
# KL is a central quantity in Shannon's information theory in measuring the amount of bits wasted in compression by using $q$ where the true distribution is $p$.
#
# KL works particularly nicely with exponential family distributions and hence plays a key role in Amari's information geometry.
#
# KL is not symmetric, and variational inference uses the following loss function:
# $$
# ELBO(\phi) = -d_{\text{KL}}(q(x;\phi) || p(x|y)) \\
# = -E_q[ \log(p(x|y)) ] + H(q(x;\phi)) \\
# = -E_q[ \log(p(y|x)) ] - E_q[ \log(p(x)) ] + E_q[ \log(p(y)) ]+ H(q(x;\phi))
# $$
# where $H(\cdot)$ denotes the entropy.
#
# Since $E_q[ \log(p(y)) ]$ is constant, we can drop it for the optimization.
# $$
# ELBO'(\phi) \approx \frac{1}{n} \sum_i [ - \log(p( x_i | y )) - \log(p(x_i))] + H(q(x;\phi))
# $$
# where $x_i$ are $n$ independent samples from $q(x;\phi)$.

# %%
nMC = 100
X = q.sample(torch.Size([nMC])) # <-- this doesn't propagate gradients; each realization is a constant, not a function of phi
ELBO = -torch.mean(lik.log_prob(X) + pri.log_prob(X)) + q.entropy()

# %% [markdown]
# ## Step 3: use the reparameterization trick
#
# Once a realization is drawn from a distribution, they no longer depend on the parameters.
# Therefore, the above Monte Carlo approximation of the expectation doesn't work.
# However, if the parameters are location and/or scale parameters, we could simply shift and scale the samples, and making them differentiable.
# This is the reparameterization trick. Fortunately, PyTorch has implemented samplers for many common distributions, so that differentiable samples may be drawn from a `torch.Distribution` object using `.rsample()`.
# But for demonstration purposes, we implement the reparameterization trick for Gaussian distribution.
#
# There are a number of variational distributions that allow the reparameterization trick.
#
# - Rezende, D. J., Mohamed, S., & Wierstra, D. (2014, May 30). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. International Conference on Machine Learning. http://jmlr.org/proceedings/papers/v32/rezende14.html
# - Kingma, D. P., & Welling, M. (2014, May 1). Auto-Encoding Variational Bayes. International Conference on Learning Representation. http://arxiv.org/abs/1312.6114
# - http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/

# %%
sn = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
X = (sn.sample(torch.Size([nMC])) ) * sigma + mu  #<-- reparametrization trick (only works for certain distributions)

# %% [markdown]
# ## Step 4: maximize the ELBO

# %%
optimizer = torch.optim.SGD([mu, sigma], lr=1e-3) # you could try Adam, if you want

# %%
from tqdm.notebook import trange, tqdm

# %%
ELBO_trace = []
for k in trange(10000):
    X = (sn.sample(torch.Size([nMC])) ) * sigma + mu # <-- reparametrization trick (only works for certain distributions)
    nELBO = -torch.mean(lik.log_prob(X) + pri.log_prob(X)) - q.entropy() # negative of the ELBO to be minimized
    ELBO_trace.append(-nELBO.item())
    optimizer.zero_grad()
    nELBO.backward()
    optimizer.step()

# %%
plt.plot(ELBO_trace);
plt.title("convergence"); plt.ylabel("ELBO"); plt.xlabel("gradient steps"); plt.grid();

# %%
q_plot = torch.distributions.normal.Normal(mu.detach(), sigma.detach())
xrt = torch.tensor(xr)
plt.plot(xr, np.exp(lik.log_prob(xrt).numpy()), label="likelihood")
plt.plot(xr, np.exp(pri.log_prob(xrt).numpy()), label="prior")
plt.plot(xr, posterior, '--', label="true posterior")
plt.plot(xr, np.exp(q_plot.log_prob(xrt).numpy()), label="variational posterior")
plt.legend(); plt.grid();

# %% [markdown]
# ## Road continues to VAE: recognition model and amortization
# There you have it! You inferred an approximate posterior through variational inference.
# We have turned Bayesian inference into optimization. As you can see, for every new observation, this approach requires an optimization with respect to the parameters of $q(\cdot)$.
#
# However, if since the optimization itself can be considered a function: You input the observation, and it outputs the optimal parameters.
# Therefore, we can fit a universal function approximator such as a neural network to the per observation inference optimization.
# This results in an architecture such that the parameters of $q(\cdot)$ to depend on the observation.
# We now write,
#
# $$ q(x) = q_\phi(x | y) $$
# where $\phi$ are the parameters of the function approximator.
# Once again, we can train $\phi$ for the training set using an optimization procedure.
# This is the so-called *amortized* inference network, or, *recognition model*, or *variational encoder*.
#
# Notice that that we have an autoencoder. Observation $y$ is "encoded" into (a variational posterior distribution over) $x$, and reconstructed to $y$.
# In other words, we have a **variational autoencoder (VAE)**.
