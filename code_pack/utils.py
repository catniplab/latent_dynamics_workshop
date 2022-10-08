import torch

def expected_ll_poisson(Y, m, P, C, delta):
    log_rate = C(m) + 0.5 * torch.einsum('nl, btlk, kn -> btn', C.weight, P, C.weight)
    likelihood_pdf = torch.distributions.Poisson(delta * torch.exp(log_rate))
    likelihood_pdf = torch.distributions.Independent(likelihood_pdf, 2)
    log_prob = likelihood_pdf.log_prob(Y)

    return log_prob


