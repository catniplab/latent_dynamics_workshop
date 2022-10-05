import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, reduce, repeat


class SeqDataLoader:
    def __init__(self, data_tuple, batch_size, shuffle=False):
        """
        Constructor for fast data loader
        :param data_tuple: a tuple of matrices, where element i is an (trial x time x features) vector
        :param batch_size: batch size
        """
        self.shuffle = shuffle
        self.data_tuple = data_tuple
        self.batch_size = batch_size
        self.dataset_len = self.data_tuple[0].shape[0]

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
        else:
            r = torch.arange(self.dataset_len)

        self.indices = [r[j * self.batch_size: (j * self.batch_size) + self.batch_size] for j in range(self.n_batches)]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_batches:
            raise StopIteration
        idx = self.indices[self.i]
        batch = tuple([x[idx, :, :] for x in self.data_tuple])
        self.i += 1
        return batch

    def __len__(self):
        return self.n_batches


class NeuralVAE(nn.Module):
    def __init__(self, cfg, time_delta, dim_in_features, dim_latents, dim_time_bins):
        super(NeuralVAE, self).__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.SYSTEM.DEVICE)
        self.d_type = torch.float32

        self.time_delta = torch.tensor(time_delta, dtype=self.d_type)
        self.dim_neurons = None
        self.dim_latents = dim_latents
        self.dim_time_bins = dim_time_bins
        self.dim_in_features = dim_in_features

        self.encoder = None
        self.decoder = None

        # variational approximation formulation
        self.q_rnn_dim_hidden = cfg.ENCODER.RNN.HIDDEN_SZ          # list of dim
        self.q_rnn_num_layers = cfg.ENCODER.RNN.HIDDEN_NUM_LAYERS

        self.q_mlp_dim_hidden = cfg.ENCODER.MLP.HIDDEN_SZ          # list of dim
        self.q_mlp_num_layers = len(cfg.ENCODER.MLP.HIDDEN_SZ)

        # generative model formulation
        self.p_mlp_dim_hidden = cfg.DECODER.MLP.HIDDEN_SZ          # list of dim
        self.p_mlp_num_layers = len(cfg.DECODER.MLP.HIDDEN_SZ)

        # build module
        self.build_module()

    def build_module(self):
        self.encoder = NeuralVAEEncoder(self.cfg, self.dim_latents, self.dim_in_features)
        self.decoder = NeuralVAEDecoder(self.cfg, self.time_delta, self.dim_latents, self.dim_in_features)

    def forward(self, y_aux, y_spikes, beta):
        y_aux = y_aux.to(self.device)
        y_spikes = y_spikes.to(self.device)

        mu_0, log_var_0, rnn_output = self.encoder.encode_initial_state(y_aux)
        z, mu_t, log_var_t, kld_loss, reconstruction_loss = self.sample_latents_forward(mu_0, log_var_0, rnn_output,
                                                                                        y_aux, y_spikes)

        if beta == 0.0:
            loss = -1 * torch.mean(reconstruction_loss)
        else:
            loss = -1 * torch.mean(reconstruction_loss - beta * kld_loss)

        return loss, z, mu_t, log_var_t

    def sample_latents_forward(self, mu_0, log_var_0, rnn_output, y_aux, y_spikes):
        """
        function for sampling from encoder and computing the loss
        :param mu_0: mean for first time step of q
        :param log_var_0:  log variance for first time step of q
        :param rnn_output: hidden states from rnn where dimensions are Time by Batch by Dimension
        :param y_aux: embedded spikes where dimensions are Batch by Time by Neurons
        :param y_spikes: raw spikes where dimensions are Batch by Time by Neurons
        :return:
        """
        y_aux = y_aux.to(self.device)
        y_spikes = y_spikes.to(self.device)

        y_spikes_rnn = rearrange(y_spikes, 'batch time neurons -> time batch neurons')  # T by B by D
        y_aux_rnn = rearrange(y_aux, 'batch time neurons -> time batch neurons')  # # T by B by D
        n_time_bins, n_trials, _ = y_aux_rnn.shape

        z_t = torch.zeros((n_time_bins, n_trials, self.dim_latents), dtype=self.d_type).to(self.device)
        mu_t = torch.zeros((n_time_bins, n_trials, self.dim_latents), dtype=self.d_type).to(self.device)
        log_var_t = torch.zeros((n_time_bins, n_trials, self.dim_latents), dtype=self.d_type).to(self.device)

        mu_t[0, :, :] = mu_0
        log_var_t[0, :, :] = log_var_0
        z_t[0, :, :] = self.reparameterize(mu_0, log_var_0)

        "generate latent trajectory samples from encoder"
        for t in range(1, n_time_bins):
            # sample latents forward through recognition model
            cat_state = torch.cat((z_t[t - 1], rnn_output[t]), dim=-1)  # concatenate previous sample with hidden states
            encoded_mean = self.encoder.q_fc_mu(cat_state)
            encoded_log_var = self.encoder.q_fc_log_var(cat_state)
            mu_t[t, :, :] = encoded_mean
            log_var_t[t, :, :] = encoded_log_var

            z_t_sample = self.reparameterize(encoded_mean, encoded_log_var)   # sample
            z_t[t, :, :] = z_t_sample  # store

        "compute KL and likelihood per time bin and sum across time. Should be size B"
        reconstruction_loss = torch.mean(self.decoder.compute_reconstruction_loss_t(y_spikes_rnn, mu_t, log_var_t), 0)
        kld = torch.mean(self.decoder.compute_kld_mc_loss(z_t, mu_t, log_var_t), 0)
        return z_t, mu_t, log_var_t, kld, reconstruction_loss

    def build_decoder(self, dim_neurons):
        self.decoder.build_module(dim_neurons)

    def manually_set_readout_params(self, C, b):
        # TODO:
        with torch.no_grad():
            self.decoder.C.bias.data = b.to(self.device)
            self.decoder.C.weight.data = C.to(self.device)

    def reparameterize(self, mu, log_var):
        mu = mu.to(self.device)
        log_var = log_var.to(self.device)

        eps = torch.randn(mu.shape, dtype=self.d_type).to(self.device)
        sigma = torch.exp(0.5 * log_var)
        z = torch.addcmul(mu, eps, sigma)

        return z

    def _calc_grad_readout(self, y_spk, mu_t, log_var_t):
        with torch.no_grad():
            device = mu_t.device
            y_spk = rearrange(y_spk, 'trial time neuron -> trial time neuron').to(device)

            mu_t = rearrange(mu_t, 'time trial latent -> trial time latent')
            log_var_t = rearrange(log_var_t, 'time trial latent -> trial time latent')
            var_t = torch.exp(log_var_t)

            A_bnl = torch.einsum('btn, btl -> bnl', y_spk, mu_t)
            A_nl = torch.sum(A_bnl, dim=0)

            B_nbtl = mu_t + torch.einsum('btl, nl -> nbtl', var_t, self.decoder.C.weight)
            B_lnbt = rearrange(B_nbtl, 'neuron batch time latent -> latent neuron batch time')
            L_nbt = torch.einsum('nl, btl -> nbt', self.decoder.C.weight, mu_t)
            Q_nbt = torch.einsum('nl, btl, nl -> nbt', self.decoder.C.weight, var_t, self.decoder.C.weight)

            # batch gradient
            exp_quadratic = torch.exp(L_nbt + 0.5 * Q_nbt + self.decoder.C.bias.unsqueeze(1).unsqueeze(2))
            exp_factor = torch.sum(B_lnbt * exp_quadratic, dim=[2,3])
            exp_factor_nl = rearrange(exp_factor, 'latent neuron -> neuron latent')
            grad_nl = A_nl - self.time_delta * exp_factor_nl

            # batch hessian
            H_nbtlk = torch.einsum('lnbt, knbt -> nbtlk', B_lnbt, B_lnbt)
            var_t_diag = torch.zeros_like(H_nbtlk).to(device)
            var_t_diag[..., np.arange(0, self.dim_latents), np.arange(0, self.dim_latents)] = var_t

            hess_nbtlk = -self.time_delta * (H_nbtlk + var_t_diag) * exp_quadratic.unsqueeze(3).unsqueeze(4)
            hess_nlk = torch.sum(hess_nbtlk, dim=[1,2])
            hess_nlk_inv = torch.inverse(hess_nlk)
            weight_update = torch.bmm(hess_nlk_inv, grad_nl.unsqueeze(2)).squeeze(2)

            # batch update for bias term
            grad_bias = torch.sum(y_spk, dim=[0,1]) - self.time_delta * torch.sum(exp_quadratic, dim=[1,2])
            hess_bias = - self.time_delta * torch.sum(exp_quadratic, dim=[1,2])
            bias_update = (1 / hess_bias) * grad_bias

        return weight_update, bias_update

    def update_readout_matrix(self, y_spk, mu_t, log_var_t, lr=1.0):
        with torch.no_grad():
            weight_update, bias_update = self._calc_grad_readout(y_spk, mu_t, log_var_t)
            C_hat = self.decoder.C.weight - lr * weight_update
            bias_hat = self.decoder.C.bias - lr * bias_update
            self.manually_set_readout_params(C_hat, bias_hat)

            C_hat = C_hat / torch.norm(C_hat, dim=0)

        return C_hat, bias_hat

    def get_rates(self, y_aux):
        y_aux = y_aux.to(self.device)
        mu_0, log_var_0, rnn_output = self.encoder.encode_initial_state(y_aux)
        dim_neurons = self.decoder.C.weight.shape[0]

        y_aux_rnn = rearrange(y_aux, 'batch time neurons -> time batch neurons')  # T by B by D
        n_time_bins, n_trials, _ = y_aux_rnn.shape
        rates = torch.zeros((n_trials, n_time_bins, dim_neurons), dtype=self.d_type).to(self.device)

        z_t = torch.zeros((n_time_bins, n_trials, self.dim_latents), dtype=self.d_type).to(self.device)
        mu_t = torch.zeros((n_time_bins, n_trials, self.dim_latents), dtype=self.d_type).to(self.device)
        log_var_t = torch.zeros((n_time_bins, n_trials, self.dim_latents), dtype=self.d_type).to(self.device)

        mu_t[0, :, :] = mu_0
        log_var_t[0, :, :] = log_var_0
        z_t[0, :, :] = self.reparameterize(mu_0, log_var_0)

        "generate latent trajectory samples from encoder"
        for t in range(1, n_time_bins):
            # sample latents forward through recognition model
            cat_state = torch.cat((z_t[t - 1], rnn_output[t]), dim=-1)  # concatenate previous sample with hidden states
            encoded_mean = self.encoder.q_fc_mu(cat_state)
            encoded_log_var = self.encoder.q_fc_log_var(cat_state)
            mu_t[t, :, :] = encoded_mean
            log_var_t[t, :, :] = encoded_log_var

            z_t_sample = self.reparameterize(encoded_mean, encoded_log_var)  # sample
            z_t[t, :, :] = z_t_sample

            P_t = torch.exp(encoded_log_var)
            quad_exp_term = torch.einsum('nl, bl, nl-> bn', self.decoder.C.weight, P_t, self.decoder.C.weight)
            rates[:, t] = torch.exp(self.decoder.C(mu_t[t]) + 0.5 * quad_exp_term)

        return rates


class NeuralVAEDecoder(nn.Module):
    def __init__(self, cfg, time_delta, dim_latents, dim_neurons):
        super(NeuralVAEDecoder, self).__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.SYSTEM.DEVICE)
        self.d_type = torch.float32

        self.time_delta = time_delta
        self.dim_neurons = dim_neurons
        self.dim_latents = dim_latents

        # prior
        self.p_mlp = None
        self.p_mlp_dim_hidden = cfg.DECODER.MLP.HIDDEN_SZ
        self.p_nlp_num_layers = len(cfg.DECODER.MLP.HIDDEN_SZ)

        self.p_fc_mu = None
        self.p_fc_log_var = None

        # likelihood
        self.C = None
        self.build_module(self.dim_neurons)

    def build_module(self, dim_neurons):
        self.dim_neurons = dim_neurons
        self._build_p_architecture()

    def compute_reconstruction_loss_t(self, y_spikes, m_neural_latents, log_var_neural_latents):
        """
        function for computing expected log likelihood analytically
        :param y_spikes: raw spike counts where dimensions are Time by Batch by Neurons
        :param m_neural_latents: mean of variational distribution where dimensions are Time by Batch by Latent Dimension
        :param log_var_neural_latents: log variance of variational distribution where dimensions are Time by Batch by Latent Dimension
        :return:
        """
        y_spikes = y_spikes.to(self.device)

        # poisson likelihood with linear readout
        var_sq_neural_latents = torch.exp(log_var_neural_latents)  # TODO: replace with softplus

        latent_projection = self.C(m_neural_latents)
        quadratic_term = torch.einsum('ni, tbi, ni -> tbn', self.C.weight, var_sq_neural_latents, self.C.weight)  # T by B by Neurons
        exp_term = torch.exp(latent_projection + 0.5 * quadratic_term)

        # reconstruction_loss = torch.sum(y_spikes * latent_projection - self.time_delta * exp_term, -1)  # T by B
        reconstruction_loss = torch.sum(y_spikes * latent_projection - self.time_delta * exp_term, dim=[0, 2])  # T by B
        return reconstruction_loss

    def compute_kld_mc_loss(self, z_t, posterior_m_t, posterior_log_var_t):
        """
        function for computing conditional KLD analytically
        :param z_t: samples from variational distribution with dimensions T by B by L
        :param posterior_m_t: mean of variational distribution with dimensions T by B by L
        :param posterior_log_var_t: log variance of variational distribution with dimensions T by B by L
        :return:
        """
        posterior_sigma_t = torch.exp(0.5 * posterior_log_var_t)  # posterior standard deviation

        "compute KL for initial conditions. we will assume a N(0, 1) prior"
        # marginal_kl = posterior_m_t[0] ** 2 + posterior_sigma_t[0] ** 2 - 1 - 2 * torch.log(posterior_sigma_t[0])
        marginal_kl = 0.0

        # using previous z_t sample, compute parameter distribution
        prior_hidden_output = self.p_mlp(z_t[:-1])
        prior_m_t = self.p_fc_mu(prior_hidden_output)
        prior_log_var_t = self.p_fc_log_var(prior_hidden_output)
        prior_sigma_t = torch.exp(0.5 * prior_log_var_t)

        sigma_ratio = posterior_sigma_t[1:] / prior_sigma_t
        marginal_kl = marginal_kl + ((posterior_m_t[1:] - prior_m_t) / prior_sigma_t) ** 2 + (sigma_ratio) ** 2 - 1 - 2 * torch.log(sigma_ratio)

        # kl = 0.5 * torch.sum(marginal_kl, dim=-1)
        kl = 0.5 * torch.sum(marginal_kl, dim=[0, 2])
        return kl

    def _build_p_architecture(self):
        # readout matrix
        self.C = torch.nn.Linear(self.dim_latents, self.dim_neurons, bias=True, dtype=self.d_type).to(self.device)

        # mlp markov transitions
        self.p_mlp = self._build_nn_function(self.dim_latents, self.p_mlp_dim_hidden, torch.nn.SiLU).to(self.device)

        # linear transformation of hidden state to parameters of dynamics distribution
        self.p_log_var = torch.nn.Parameter(torch.log(torch.ones(self.dim_latents)))
        self.p_fc_mu = torch.nn.Linear(self.p_mlp_dim_hidden[-1], self.dim_latents, dtype=self.d_type).to(self.device)
        self.p_fc_log_var = torch.nn.Linear(self.p_mlp_dim_hidden[-1], self.dim_latents, dtype=self.d_type).to(self.device)

    def p_fc_log_var(self, arg):
        return torch.ones((arg.shape[0], arg.shape[1], self.dim_latents)) * self.p_log_var

    def _build_nn_function(self, dim_input, dim_hidden_layers, nonlinearity_fn):
        nn_modules = []

        for dx, hidden_layer_dim in enumerate(dim_hidden_layers):
            if dx == 0:
                dim_in = dim_input
                dim_out = hidden_layer_dim
            else:
                dim_in = dim_hidden_layers[dx - 1]
                dim_out = hidden_layer_dim

            nn_modules.append(
                torch.nn.Sequential(
                    torch.nn.Linear(dim_in, dim_out, dtype=self.d_type).to(self.device),
                    nonlinearity_fn()
                )
            )

        neural_net = torch.nn.Sequential(*nn_modules)

        return neural_net


class NeuralVAEEncoder(nn.Module):
    def __init__(self, cfg, dim_latents, dim_in_features):
        super(NeuralVAEEncoder, self).__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.SYSTEM.DEVICE)
        self.d_type = torch.float32
        self.dim_latents = dim_latents
        self.dim_in_features = dim_in_features

        self.q_rnn = None
        self.q_rnn_dim_hidden = cfg.ENCODER.RNN.HIDDEN_SZ
        self.q_rnn_num_layers = cfg.ENCODER.RNN.HIDDEN_NUM_LAYERS

        self.q_mlp = None
        self.q_mlp_dim_hidden = cfg.ENCODER.MLP.HIDDEN_SZ
        self.q_mlp_num_layers = len(cfg.ENCODER.MLP.HIDDEN_SZ)

        self.q_fc_mu = None
        self.q_fc_log_var = None

        self.build_module()

    def encode_initial_state(self, y_aux):
        y_aux = y_aux.to(self.device)
        y_aux_rnn = rearrange(y_aux, 'batch time neurons -> time batch neurons')

        with torch.backends.cudnn.flags(enabled=False):
            rnn_output, _ = self.q_rnn(y_aux_rnn)

        all_hidden_states = rnn_output.view(y_aux_rnn.shape[0], y_aux_rnn.shape[1], 2, self.q_rnn_dim_hidden)

        # concatenate the forward and backward
        hn = torch.cat((all_hidden_states[:, :, 0], all_hidden_states[:, :, 1]), 2)

        temp = torch.cat((torch.zeros(y_aux_rnn.shape[1], self.dim_latents,
                                      device=self.device, dtype=self.d_type), hn[0]), 1)
        # mu_0 = self.q_fc_mu_0(temp)
        # log_var_0 = self.q_fc_log_var_0(temp)
        mu_0 = self.q_fc_mu(temp)
        log_var_0 = self.q_fc_log_var(temp)

        return mu_0, log_var_0, rnn_output

    def build_module(self):
        self._build_q_architecture()
        self.init_weights()

    def _build_q_architecture(self):
        # parameterize variational posterior
        self.q_rnn = nn.GRU(self.dim_in_features, self.q_rnn_dim_hidden, self.q_rnn_num_layers,
                            bidirectional=True, dtype=self.d_type).to(self.device)

        self.q_fc_mu = nn.Linear(2 * self.q_rnn_dim_hidden + self.dim_latents,
                                 self.dim_latents, dtype=self.d_type).to(self.device)
        self.q_fc_log_var = nn.Linear(2 * self.q_rnn_dim_hidden + self.dim_latents,
                                      self.dim_latents, dtype=self.d_type).to(self.device)

    def _build_nn_function(self, dim_input, dim_hidden_layers, nonlinearity_fn):
        nn_modules = []

        for dx, hidden_layer_dim in enumerate(dim_hidden_layers):
            if dx == 0:
                dim_in = dim_input
                dim_out = hidden_layer_dim
            else:
                dim_in = dim_hidden_layers[dx - 1]
                dim_out = hidden_layer_dim

            nn_modules.append(
                torch.nn.Sequential(
                    torch.nn.Linear(dim_in, dim_out, dtype=self.d_type).to(self.device),
                    nonlinearity_fn()
                )
            )

        neural_net = torch.nn.Sequential(*nn_modules)

        return neural_net

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        for idx in range(4):
                            mul = param.shape[0] // 4
                            if 'weight_ih' in name:
                                torch.nn.init.xavier_uniform_(param.data[idx*mul:(idx+1)*mul])
                            else:
                                torch.nn.init.orthogonal_(param.data[idx * mul:(idx + 1) * mul])
                    elif 'bias' in name:
                        param.data.fill_(0)



def main():
    import h5py
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    from config import get_cfg_defaults
    from SequentialVAE import NeuralVAE

    cfg = get_cfg_defaults()
    data = h5py.File('data/poisson_obs.h5')
    Y = torch.tensor(np.array(data['Y']), dtype=torch.float32)
    X = torch.tensor(np.array(data['X']), dtype=torch.float32)
    C = torch.tensor(np.array(data['C']), dtype=torch.float32)
    b = torch.tensor(np.array(data['bias']), dtype=torch.float32)

    n_epochs = 100
    batch_size = 10
    time_delta = 5e-3
    n_latents = X.shape[2]
    n_neurons = Y.shape[2]
    n_time_bins = Y.shape[1]
    vae = NeuralVAE(cfg, time_delta, n_neurons, n_latents, n_time_bins)
    vae.manually_set_readout_params(C, b)

    vae.decoder.C.bias.requires_grad_(False)
    vae.decoder.C.weight.requires_grad_(False)
    train_data_loader = SeqDataLoader((Y, X), batch_size)


    opt = torch.optim.Adam(vae.parameters(), lr=1e-2)
    for epoch in range(n_epochs):
        for batch_idx, (y, x) in enumerate(train_data_loader):
            loss, z, mu_t, log_var_t = vae(y, y, 1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1e-1, norm_type=2)
            opt.step()
            opt.zero_grad()
            # print(batch_idx)

        print(epoch)
        if epoch % 5 == 0:
            plt.plot(z[:, 0, 0].detach().numpy())
            plt.plot(x[0, :, 0].detach().numpy())
            plt.show()


if __name__ == '__main__':
    main()
