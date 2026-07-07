import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def plot_two_d_vector_field_from_data(dynamics_func, axs, axs_range, P=None):
    x = np.linspace(min(axs_range['x_min'], -2), max(axs_range['x_max'], 2), 25)
    y = np.linspace(min(axs_range['y_min'], -2), max(axs_range['y_max'], 2), 25)

    X, Y = np.meshgrid(x, y)
    u, v = np.zeros(X.shape), np.zeros(Y.shape)
    speed = np.zeros(X.shape)
    NI, NJ = Y.shape

    for i in range(NI):
        for j in range(NJ):
            x = X[i, j]
            y = Y[i, j]

            vec_in = np.array([x, y])

            if('torch.nn.modules' in str(type(dynamics_func))):
                vec_out = np.asarray(dynamics_func(torch.tensor(vec_in, dtype=torch.float32)))
            else:
                # ode always needs 0th time point, so we take the first mapping which is not 0
                vec_out = dynamics_func(vec_in)[1]


            if P is None:
                s = (vec_out - vec_in)
            else:
                s = (vec_out - vec_in) @ np.transpose(P)

            u[i, j] = np.array(s[0])
            v[i, j] = s[1]
            speed[i, j] = torch.norm(torch.tensor(s)).cpu().data.numpy()

    # speed = speed / speed.max()
    axs.streamplot(X, Y, u, v, color=speed, linewidth=0.5, arrowsize=0.3)


def raster_to_events(raster):
    events = []
    for i in range(raster.shape[1]):
        row = raster[:, i]
        rowidx = np.nonzero(row)[0]
        events.append(rowidx)
    return events


def plot_rotated_latents(z_rot, m_rot, z_true, label, n_samples):
    """Overlay posterior samples (gray), posterior mean, and true latent per dim."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    for d in range(2):
        axs[d].set_title(f"rotated latent trajectory (dim {d})")
        axs[d].set_box_aspect(0.2)
        for s in range(n_samples):
            axs[d].plot(z_rot[s, 0, :, d], linewidth=0.5, color="gray")
        axs[d].plot(m_rot[0, :, d], label=label)
        axs[d].plot(z_true[0, :, d], label="true")
        axs[d].legend()
        axs[d].set_xlabel("time")
    plt.tight_layout()
    plt.show()


def plot_single_reaches(reaches, n_trials_to_plot):
    """Plumbing: integrate velocity to hand position and color by reach angle."""
    trial_plt_dx = torch.randperm(reaches.shape[0])[:n_trials_to_plot]

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle('hand reaches')
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis('off')

    for n in trial_plt_dx:
        traj = torch.cumsum(reaches[n], dim=0)
        reach_angle = torch.atan2(traj[-1, 0], traj[-1, 1])
        reach_color = plt.cm.hsv(reach_angle / (2 * np.pi) + 0.5)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, alpha=0.8, color=reach_color)


def plot_spikes_and_behavior(spikes, velocity, binsize, trials_inds, event_bin):
    """Plumbing: raster (top) and hand velocity (bottom) for a few trials."""
    n_trials = len(trials_inds)
    fig, axes = plt.subplots(nrows=2, ncols=n_trials, figsize=(4 * n_trials, 6), sharex=False, sharey='row')
    if n_trials == 1:
        axes = axes.reshape(2, 1)

    for col, trial_idx in enumerate(trials_inds):
        trial = spikes[trial_idx]
        reach = velocity[trial_idx]
        ax_spikes = axes[0, col]
        ax_vel = axes[1, col]

        for neuron_idx in range(trial.shape[-1]):
            spike_times = np.where(trial[:, neuron_idx].cpu() == 1)[0]
            ax_spikes.scatter(spike_times, [neuron_idx] * len(spike_times), s=4, color='gray', marker='|')

        ax_spikes.axvline(x=event_bin, linestyle='--', color='purple', alpha=0.4)
        ax_spikes.set_ylabel('neurons')
        ax_spikes.set_title(f'Trial {trial_idx}\n# spikes: {int(torch.sum(trial))}', fontsize=10)
        ax_spikes.set_xlabel('time bins')

        time_axis = torch.arange(reach.shape[0]) * binsize
        ax_vel.plot(time_axis, reach[:, 0], color='navy', label='vel x')
        ax_vel.plot(time_axis, reach[:, 1], color='coral', label='vel y')
        ax_vel.axvline(x=event_bin * binsize, linestyle='--', color='purple', alpha=0.4)
        ax_vel.set_xlabel('time (ms)')
        ax_vel.set_title('hand velocity', fontsize=10)
        ax_vel.legend(fontsize=8)

        if col == 0:
            _, y_top = ax_spikes.get_ylim()
            ax_spikes.annotate("movement\nonset", xy=(event_bin, y_top), xytext=(event_bin - 10, y_top + 3),
                               arrowprops=dict(facecolor='black', alpha=0.4, arrowstyle='->'),
                               fontsize=7, ha='center', alpha=0.8)

    fig.tight_layout()
    plt.show()


def plot_spikes_and_decoded_behavior(spikes, velocity, velocity_hat, binsize, trials_inds, event_bin):
    """Plumbing: raster (top) and true vs decoded hand velocity (bottom)."""
    n_trials = len(trials_inds)
    fig, axes = plt.subplots(nrows=2, ncols=n_trials, figsize=(4 * n_trials, 6), sharex=False, sharey='row')
    if n_trials == 1:
        axes = axes.reshape(2, 1)

    for col, trial_idx in enumerate(trials_inds):
        trial = spikes[trial_idx]
        reach = velocity[trial_idx]
        decoded_reach = velocity_hat[trial_idx]
        ax_spikes = axes[0, col]
        ax_vel = axes[1, col]

        for neuron_idx in range(trial.shape[-1]):
            spike_times = np.where(trial[:, neuron_idx].cpu() == 1)[0]
            ax_spikes.scatter(spike_times, [neuron_idx] * len(spike_times), s=4, color='gray', marker='|')

        ax_spikes.axvline(x=event_bin, linestyle='--', color='purple', alpha=0.4)
        ax_spikes.set_ylabel('neurons')
        ax_spikes.set_title(f'\nTrial {trial_idx}\n# spikes: {int(torch.sum(trial))}', fontsize=10)
        ax_spikes.set_xlabel('time bins')

        time_axis = torch.arange(reach.shape[0]) * binsize
        ax_vel.plot(time_axis, reach[:, 0], color='navy', linewidth=1.0, label='true vel x' if col == 0 else '')
        ax_vel.plot(time_axis, reach[:, 1], color='coral', linewidth=1.0, label='true vel y' if col == 0 else '')
        ax_vel.plot(time_axis, decoded_reach[:, 0], linestyle='--', linewidth=1.0, color='navy', label='decoded vel x' if col == 0 else '')
        ax_vel.plot(time_axis, decoded_reach[:, 1], linestyle='--', linewidth=1.0, color='coral', label='decoded vel y' if col == 0 else '')
        ax_vel.axvline(x=event_bin * binsize, linestyle='--', linewidth=1.0, color='purple', alpha=0.4)
        ax_vel.set_xlabel('time (ms)')
        ax_vel.set_title('\nhand velocity', fontsize=10)

        if col == 0:
            _, y_top = ax_spikes.get_ylim()
            ax_spikes.annotate("movement\nonset", xy=(event_bin, y_top), xytext=(event_bin - 10, y_top + 3),
                               arrowprops=dict(facecolor='black', alpha=0.4, arrowstyle='->'),
                               fontsize=7, ha='center', alpha=0.8)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    plt.show()


def pca_vs_r2(z_train, z_test, vel_train, vel_test, max_pcs=40, alpha=0.01):
    flatten = lambda x: x.reshape(-1, x.shape[2]).detach().cpu().numpy()
    flatten_vel = lambda v: v.reshape(-1, 2).detach().cpu().numpy()

    X_train, X_test = flatten(z_train), flatten(z_test)
    y_train, y_test = flatten_vel(vel_train), flatten_vel(vel_test)

    r2_scores = []
    clf = None
    for k in range(1, max_pcs + 1):
        pca = PCA(n_components=k)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        clf = Ridge(alpha=alpha).fit(X_train_pca, y_train)
        r2_scores.append(r2_score(y_test, clf.predict(X_test_pca), multioutput='uniform_average'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, max_pcs + 1), r2_scores, marker='o')
    axes[0].set_xlabel("num PCs used for decoding")
    axes[0].set_ylabel("R2")
    axes[0].set_title("velocity decoding from latents")
    axes[0].grid(True)

    importance = np.abs(clf.coef_).mean(axis=0)  # last fit uses all max_pcs PCs
    axes[1].bar(np.arange(len(importance)), importance[np.argsort(-importance)])
    axes[1].set_xlabel('latent PC (sorted)')
    axes[1].set_ylabel('mean |w| (vx & vy)')
    axes[1].set_title('PC importance for decoding')
    plt.tight_layout()
    plt.show()

    return np.array(r2_scores)


def plot_trials(true_rates, generated_rates, n=4, spike_threshold=0.1):
    """Plumbing: observed spike raster (top) vs generated rate heatmap (bottom)."""
    true_rates = true_rates.cpu() if isinstance(true_rates, torch.Tensor) else true_rates
    generated_rates = generated_rates.cpu() if isinstance(generated_rates, torch.Tensor) else generated_rates

    trials = true_rates.shape[0]
    n = min(n, trials)
    random_indices = np.random.choice(trials, size=n, replace=False)

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 8), sharex=True, sharey='row')
    im = None
    for idx, trial_i in enumerate(random_indices):
        spikes = true_rates[trial_i]
        ax_raster = axes[0, idx]
        spike_times, neuron_ids = np.where(spikes > spike_threshold)
        ax_raster.scatter(spike_times, neuron_ids, s=2, color='black')
        if idx == 0:
            ax_raster.set_ylabel("neuron")

        ax_gen = axes[1, idx]
        im = ax_gen.imshow(generated_rates[trial_i].T, aspect='auto', cmap='viridis', origin='lower')
        if idx == 0:
            ax_gen.set_ylabel("neuron")
            ax_gen.set_xlabel("time bins")

    cbar_ax = fig.add_axes([1., 0.12, 0.015, 0.33])
    fig.colorbar(im, cax=cbar_ax, label="firing rate")
    axes[0, 0].set_title("Observed spikes (raster)", fontsize=12)
    axes[1, 0].set_title("Generated firing rates", fontsize=12)
    plt.tight_layout()
    plt.show()
