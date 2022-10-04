import h5py
import torch
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint


def generate_van_der_pol(state0, t, system_parameters):
    mu = system_parameters['mu']
    tau_1 = system_parameters['tau_1']
    tau_2 = system_parameters['tau_2']

    def f(state, t):
        x, y = state
        return y / tau_1, (mu * (1 - x**2) * y - x) / tau_2

    states = odeint(f, state0, t)

    return states


def generate_noisy_van_der_pol(state0, t, system_parameters):
    T = t.shape[0]
    delta = t[1] - t[0]
    mu = system_parameters['mu']
    tau_1 = system_parameters['tau_1']
    tau_2 = system_parameters['tau_2']
    sigma = system_parameters['sigma']

    states = np.zeros((T, 2))
    states[0, 0] = state0[0]
    states[0, 1] = state0[1]

    for dx in range(1, T):
        x_next = states[dx - 1, 0] + (delta / tau_1) * states[dx - 1, 1]
        y_next = states[dx - 1, 1] + (delta / tau_2) * (mu * (1 - states[dx - 1, 0]**2) * states[dx - 1, 1] - states[dx - 1, 0])

        states[dx, 0] = x_next + sigma * np.random.randn()
        states[dx, 1] = y_next + sigma * np.random.randn()

    return states


def main():
    data_path = pathlib.Path(f'../vanderpol/data/vdp_noisy.h5')
    # data_path = pathlib.Path(f'../van_der_pol_poisson_obs_hairplot/data/observations/noisy_van_der_pol_dynamics_poisson_obs.h5')
    data_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    torch.set_default_dtype(torch.float64)

    delta = 5e-3  # time bin size
    n_trials = 30  # total trials [1/3 train | 1/3 val | 1/3 test]
    n_latents = 2
    n_neurons = 150
    n_time_bins = 1000
    n_cutoff_bins = 500

    system_parameters = {}
    system_parameters['mu'] = 1.5
    system_parameters['tau_1'] = 0.1
    system_parameters['tau_2'] = 0.1
    system_parameters['sigma'] = 0.1  # noise add into euler integration
    Q = np.diag(system_parameters['sigma'] * np.ones(n_latents))

    t = delta * torch.arange(n_time_bins)
    C = torch.randn((n_neurons, n_latents), dtype=torch.float64)
    C = (1 / np.sqrt(3)) * (C / torch.norm(C, dim=1).unsqueeze(1))
    b = torch.log(25 + 25 * torch.rand(n_neurons, dtype=torch.float64))  # 10 to 60 hz baseline

    X = torch.zeros(n_trials, n_time_bins, n_latents)
    Y = torch.zeros(n_trials, n_time_bins, n_neurons)
    r = torch.zeros(n_trials, n_time_bins, n_neurons)

    for trial in range(n_trials):
        if trial < n_trials//2:
            state00 = np.random.uniform(-0.5, 0.5)
            state01 = np.random.uniform(-0.5, 0.5)
        else:
            state00 = np.random.uniform(-4.0, 4.0)
            state01 = np.random.uniform(-4.0, 4.0)

        state0 = (state00, state01)

        states = generate_noisy_van_der_pol(state0, t, system_parameters)
        states_torch = torch.tensor(0.4 * states)
        r[trial] = delta * torch.exp(states_torch @ C.T + b)
        X[trial] = states_torch
        Y[trial] = torch.poisson(r[trial])

        plt.plot(states_torch[:, 0], states_torch[:, 1])

    f = h5py.File(data_path, 'w')
    perm_trial_dx = torch.randperm(n_trials)
    train_trial_dx = perm_trial_dx[:n_trials // 3]
    test_trial_dx = perm_trial_dx[-n_trials // 3:]
    val_trial_dx = perm_trial_dx[n_trials // 3:-n_trials // 3]

    f.create_dataset('C', data=C)
    f.create_dataset('Q', data=Q)
    f.create_dataset('bias', data=b)
    f.create_dataset('delta', data=delta)

    ## adding dataset params
    f.create_dataset('n_trials', data=n_trials)
    f.create_dataset('n_latents', data=n_latents)
    f.create_dataset('n_neurons', data=n_neurons)
    f.create_dataset('n_time_bins', data=n_time_bins)
    f.create_dataset('mu', data=system_parameters['mu'])
    f.create_dataset('tau_1', data=system_parameters['tau_1'])
    f.create_dataset('tau_2', data=system_parameters['tau_2'])
    f.create_dataset('sigma', data=system_parameters['sigma'])

    f.create_dataset('r', data=r[train_trial_dx, n_cutoff_bins:, :])
    f.create_dataset('X', data=X[train_trial_dx, n_cutoff_bins:, :])
    f.create_dataset('Y', data=Y[train_trial_dx, n_cutoff_bins:, :])

    f.create_dataset('r_val', data=r[val_trial_dx, n_cutoff_bins:, :])
    f.create_dataset('X_val', data=X[val_trial_dx, n_cutoff_bins:, :])
    f.create_dataset('Y_val', data=Y[val_trial_dx, n_cutoff_bins:, :])

    f.create_dataset('r_test', data=r[test_trial_dx, n_cutoff_bins:, :])
    f.create_dataset('X_test', data=X[test_trial_dx, n_cutoff_bins:, :])
    f.create_dataset('Y_test', data=Y[test_trial_dx, n_cutoff_bins:, :])
    f.close()


if __name__ == '__main__':
    main()