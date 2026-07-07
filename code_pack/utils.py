import torch
import numpy as np
from einops import rearrange


# Filename of the pretrained XFADS checkpoint shipped for the MC_Maze notebooks
# (06 and 07). Both notebooks must load exactly this checkpoint.
_MC_MAZE_CKPT_NAME = 'epoch=827_valid_loss=1415.56_r2_valid_enc=0.89_r2_valid_bhv=0.00_valid_bps_enc=0.42.ckpt'


def load_mc_maze_data(cfg, in_colab):
    """Load the MC_Maze train/valid/test splits as saved tensors (no randomness)."""
    data_splits_path = ('latent_dynamics_workshop/external/xfads/examples/monkey_reaching/data'
                        if in_colab else './external/xfads/examples/monkey_reaching/data')
    train_data = torch.load(data_splits_path + f'/data_train_{cfg.bin_sz_ms}ms.pt')
    valid_data = torch.load(data_splits_path + f'/data_valid_{cfg.bin_sz_ms}ms.pt')
    test_data = torch.load(data_splits_path + f'/data_test_{cfg.bin_sz_ms}ms.pt')
    return train_data, valid_data, test_data


def build_mc_maze_ssm(cfg, n_neurons_obs):
    """Wire the XFADS state-space model (dynamics, Poisson likelihood, encoders, filter).

    This is the module construction shared verbatim by notebooks 06 and 07. The random
    parameter initialization here is overwritten when a checkpoint is loaded; the model
    is returned untrained. Returns the assembled ``ssm`` and its ``dynamics_mod`` (needed
    for k-step forecasting in 07).
    """
    import torch.nn as nn
    import xfads.utils as xfads_utils
    from xfads.smoothers.nonlinear_smoother_causal import LowRankNonlinearStateSpaceModel, NonlinearFilter
    from xfads.ssm_modules.dynamics import DenseGaussianDynamics, DenseGaussianInitialCondition
    from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
    from xfads.ssm_modules.likelihoods import PoissonLikelihood

    if cfg.device == 'cuda':
        torch.cuda.empty_cache()

    """likelihood module (Poisson spikes; see sec:expfam)"""
    H = xfads_utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))
    likelihood_pdf = PoissonLikelihood(readout_fn, n_neurons_obs, cfg.bin_sz, device=cfg.device)

    """dynamics module"""
    Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = xfads_utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

    """initial condition"""
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    Q_0_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                            rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                            device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_obs, cfg.n_hidden_local, cfg.n_latents, rank=cfg.rank_local,
                                      device=cfg.device, dropout=cfg.p_local_dropout)

    """nonlinear filter"""
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

    """sequential vae"""
    ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                          local_encoder, nl_filter, device=cfg.device)
    return ssm, dynamics_mod


def load_mc_maze_model(cfg, n_neurons_obs, n_bins_enc, bin_prd_start, in_colab):
    """Build the XFADS SSM and load the pretrained MC_Maze checkpoint into it.

    Returns ``(seq_vae, ssm, dynamics_mod)`` with the SSM in eval mode. This is the
    load-pretrained path shared by notebooks 06 and 07; 06 uses ``build_mc_maze_ssm``
    directly for its optional train-from-scratch branch.
    """
    from xfads.smoothers.lightning_trainers import LightningMonkeyReaching

    ssm, dynamics_mod = build_mc_maze_ssm(cfg, n_neurons_obs)

    ckpts_path = 'latent_dynamics_workshop/ckpts/mc_maze' if in_colab else './ckpts/mc_maze'
    best_model_path = f'{ckpts_path}/{_MC_MAZE_CKPT_NAME}'
    seq_vae = LightningMonkeyReaching.load_from_checkpoint(best_model_path, ssm=ssm, cfg=cfg,
                                                           n_time_bins_enc=n_bins_enc, n_time_bins_bhv=bin_prd_start,
                                                           strict=False)
    seq_vae.ssm = seq_vae.ssm.to(cfg.device)
    seq_vae.ssm.eval()
    return seq_vae, ssm, dynamics_mod


def expected_ll_poisson(Y, m, P, C, delta, dtype=torch.float32):
    m_t = torch.tensor(m, dtype=dtype)
    P_t = torch.tensor(P, dtype=dtype)
    Y_t = torch.tensor(Y, dtype=dtype)
    spk_count_per_trial = Y_t.sum(dim=1)

    log_rate = C(m_t) + 0.5 * torch.einsum('nl, btl, nl -> btn', C.weight, P_t, C.weight)
    likelihood_pdf = torch.distributions.Poisson(delta * torch.exp(log_rate))
    log_prob = likelihood_pdf.log_prob(Y_t)
    log_prob = log_prob.sum(dim=1)

    null_likelihood_pdf = torch.distributions.Poisson(delta * torch.exp(C.bias) * torch.ones_like(log_rate))
    null_likelihood_log_prob = null_likelihood_pdf.log_prob(Y_t)
    null_likelihood_log_prob = null_likelihood_log_prob.sum(dim=1)

    bidx = spk_count_per_trial != 0 # exclude (neurons x trials) with no spikes
    nats_array = torch.mean((log_prob[bidx] - null_likelihood_log_prob[bidx]) * (1 / spk_count_per_trial[bidx]))

    return nats_array / np.log(2.0) # bits/spike/neuron


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
        if (torch.is_tensor(m)):
            M[n] = torch.tensor(m[n].detach().clone())
        else:
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
#
#
# def main():
#     n_trials = 5
#     n_latents = 2
#     n_neurons = 150
#     n_time_bins = 500
#
#     Y = torch.randint(3, (n_trials, n_time_bins, n_neurons))**2
#     m = torch.randn((n_trials, n_time_bins, n_latents))
#     P = torch.randn((n_trials, n_time_bins, n_latents))**2
#     C = torch.nn.Linear(n_latents, n_neurons)
#
#     expected_ll_poisson(Y, m, P, C, 5e-3)
#
#
# if __name__ == '__main__':
#     main()
