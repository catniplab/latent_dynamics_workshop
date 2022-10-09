import torch
import numpy as np

def expected_ll_poisson(Y, m, P, C, delta, dtype=torch.float32):
    m_t = torch.tensor(m, dtype=dtype)
    P_t = torch.tensor(P, dtype=dtype)
    Y_t = torch.tensor(Y, dtype=dtype)

    log_rate = C(m_t) + 0.5 * torch.einsum('nl, btl, nl -> btn', C.weight, P_t, C.weight)
    likelihood_pdf = torch.distributions.Poisson(delta * torch.exp(log_rate))
    likelihood_pdf = torch.distributions.Independent(likelihood_pdf, 2)
    log_prob = likelihood_pdf.log_prob(Y_t)

    return torch.mean(log_prob)


def best_fit_transformation(X, X_lat, n_trials, n_time_bins, n_latents):
    # regress to account for invariance
    S = np.linalg.pinv(X_lat) @ X.reshape(n_trials * n_time_bins, n_latents)
    X_hat_tilde = X_lat @ S
    X_hat_tilde = X_hat_tilde.reshape(n_trials, n_time_bins, n_latents)

    return X_hat_tilde


def estimate_readout_matrix(Y, m, P, delta, n_iter=2500):
    n_trials = Y.shape[0]
    n_neuron = Y.shape[2]
    n_latent = m.shape[2]
    n_time_bins = Y.shape[1]
    M = torch.zeros((n_trials, n_time_bins, n_latent))
    C_hat = torch.nn.Linear(n_latent, n_neuron, bias=True)

    for n in range(n_trials):
        M[n] = torch.tensor(m[n])

    opt = torch.optim.Adam(C_hat.parameters(), lr=1e-2)
    loss_log = []

    for i in range(n_iter):
        log_r = C_hat(M)
        ell = torch.tensor(Y) * log_r - delta * torch.exp(log_r)
        loss = -1 * torch.sum(ell)

        loss.backward()
        opt.step()
        opt.zero_grad()
        loss_log.append(loss.item())

    return C_hat
