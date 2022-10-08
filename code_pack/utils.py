import torch
import numpy as np
from einops import rearrange

def expected_ll_poisson(Y, m, P, C, delta):
    log_rate = C(m) + 0.5 * torch.einsum('nl, btlk, kn -> btn', C.weight, P, C.weight)
    likelihood_pdf = torch.distributions.Poisson(delta * torch.exp(log_rate))
    likelihood_pdf = torch.distributions.Independent(likelihood_pdf, 2)
    log_prob = likelihood_pdf.log_prob(Y)

    return log_prob
def best_fit_transformation(X, X_lat, n_trials, n_time_bins, n_latents):
    # regress to account for invariance
    S = np.linalg.pinv(X_lat) @ X.reshape(n_trials * n_time_bins, n_latents)
    X_hat_tilde = X_lat @ S
    X_hat_tilde = X_hat_tilde.reshape(n_trials, n_time_bins, n_latents)

    return X_hat_tilde
