import h5py
import torch
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from process_generators.rotational_ar_process_generator import AutoRegressiveProcess


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
    data_path = pathlib.Path(f'../linear_gaussian_filtering_complexity/data/observations/rotational_dynamics_poisson_obs.h5')
    data_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    torch.set_default_dtype(torch.float64)

    delta = 5e-3  # time bin size
    n_trials = 6  # total trials [1/3 train | 1/3 val | 1/3 test]
    n_cutoff = 300  # cutoff bins at trial start (let transient settle)
    n_latents = 2
    n_neurons = 150
    theta = np.pi / 16  # rotational velocity for gen data
    damping_factor = 0.99  # pre-multiplier reduce eigenvalues of dynamics matrix
    dynamics_noise_std = 0.05
    n_time_bins = 25000 + n_cutoff

    system_parameters = {}
    system_parameters['mu'] = 1.5
    system_parameters['tau_1'] = 0.1
    system_parameters['tau_2'] = 0.1
    system_parameters['sigma'] = 0.1
    system_parameters['obs_noise'] = 0.5

    R = system_parameters['obs_noise']**2 * torch.eye(n_neurons)
    C = torch.randn((n_neurons, n_latents), dtype=torch.float64)
    C = (1 / np.sqrt(3)) * (C / torch.norm(C, dim=1).unsqueeze(1))
    b = torch.log(25 + 25 * torch.rand(n_neurons, dtype=torch.float64))  # 10 to 60 hz baseline

    ar_proc = AutoRegressiveProcess(theta, C, b, delta, dynamics_noise_std, damping_factor)
    X, Y, r = ar_proc.sample_trajectory(n_time_bins, n_trials)
    print(Y.max())

    fig, axs = plt.subplots(2, 1, figsize=(10, 3))
    axs[0].plot(X[0, :, 0])
    axs[1].plot(X[0, :, 1])
    axs[0].set_title('example trajectory')
    plt.show()

    f = h5py.File(data_path, 'w')
    perm_trial_dx = torch.randperm(n_trials)
    train_trial_dx = perm_trial_dx[:n_trials // 3]
    test_trial_dx = perm_trial_dx[-n_trials // 3:]
    val_trial_dx = perm_trial_dx[n_trials // 3:-n_trials // 3]

    A = ar_proc.A
    Q = ar_proc.Q

    f.create_dataset('R', data=R)
    f.create_dataset('Q', data=Q)
    f.create_dataset('A', data=A)
    f.create_dataset('C', data=C)
    f.create_dataset('bias', data=b)
    f.create_dataset('delta', data=delta)

    f.create_dataset('X', data=X[train_trial_dx, :, :])
    f.create_dataset('Y', data=Y[train_trial_dx, :, :])

    f.create_dataset('X_val', data=X[val_trial_dx, :, :])
    f.create_dataset('Y_val', data=Y[val_trial_dx, :, :])

    f.create_dataset('X_test', data=X[test_trial_dx, :, :])
    f.create_dataset('Y_test', data=Y[test_trial_dx, :, :])
    f.close()


if __name__ == '__main__':
    main()