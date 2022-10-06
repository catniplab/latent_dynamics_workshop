import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py

import sys
sys.path.append("../nlb_tools/")
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5


def get_spikes_from_dict(spike_dict):
    held_in_spk = spike_dict['train_spikes_heldin']
    held_out_spk = spike_dict['train_spikes_heldout']
    stacked_spikes = np.concatenate([held_in_spk, held_out_spk], axis=2)

    return stacked_spikes


def main():
    # dandi download https://dandiarchive.org/dandiset/000138 for mc_maze 00128 for regular
    torch.set_default_dtype(torch.float64)
    dataset_name = 'mc_maze'
    datapath = 'data/000138/sub-Jenkins/' #NOTE: we have used the large dataset here
    dataset = NWBDataset(datapath)

    # Extract neural data and lagged hand velocity
    binsize = 5
    dataset.resample(binsize)

    n_val_trials = 250
    start = -100
    end = 450
    lag = 80

    trial_info_save_path = 'data/info_per_trial_{}.pkl'
    spikes_per_trial_save_path = 'data/spikes_per_trial.h5'
    rates_per_trial_save_path = 'data/rates_per_trial_{}.npy'
    velocity_per_trial_save_path = 'data/velocity_per_trial_{}.npy'
    position_per_trial_save_path = 'data/position_per_trial_{}.npy'

    # Extract neural data and lagged hand velocity
    trial_info = dataset.trial_info.dropna()
    trial_info['color'] = None
    trial_info['position_id'] = None
    dataset.smooth_spk(50, name='smth_50')
    trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(start-lag, end-lag))
    lagged_trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(start, end))

    rates = trial_data.spikes_smth_50.to_numpy()
    velocity = lagged_trial_data.hand_vel.to_numpy()

    trial_length = int(end - start) // binsize
    n_trials = rates.shape[0] // trial_length

    spikes_per_trial = trial_data.spikes.values.reshape(n_trials, trial_length, -1)
    rates_per_trial = rates.reshape(n_trials, trial_length, -1)
    velocity_per_trial = velocity.reshape(n_trials, trial_length, -1)
    trajectory_per_trial = np.cumsum(velocity_per_trial, axis=1) * dataset.bin_width / 1000

    # high_fr_dx = torch.where(C_bias > 0.0)[0]
    # print(high_fr_dx.shape[0])
    #
    # spikes_per_trial = spikes_per_trial[:, :, high_fr_dx]
    # rates_per_trial = rates_per_trial[:, :, high_fr_dx]

    for dx, (row_id, row_ss) in enumerate(trial_info.iterrows()):
        reach_angle = np.arctan2(*trajectory_per_trial[dx-1, -1])
        trial_info.at[row_id, 'color'] = plt.cm.hsv(reach_angle / (2 * np.pi) + 0.5)
        trial_info.at[row_id, 'position_id'] = int(row_ss['trial_type'])

    trial_info.iloc[:n_val_trials].to_pickle(trial_info_save_path.format('val'))
    trial_info.iloc[n_val_trials:].to_pickle(trial_info_save_path.format('train'))

    np.save(rates_per_trial_save_path.format('val'), rates_per_trial[:n_val_trials])
    np.save(rates_per_trial_save_path.format('train'), rates_per_trial[n_val_trials:])

    np.save(velocity_per_trial_save_path.format('val'), velocity_per_trial[:n_val_trials])
    np.save(velocity_per_trial_save_path.format('train'), velocity_per_trial[n_val_trials:])

    np.save(position_per_trial_save_path.format('val'), trajectory_per_trial[:n_val_trials])
    np.save(position_per_trial_save_path.format('train'), trajectory_per_trial[n_val_trials:])

    spikes_per_trial_h5 = h5py.File(spikes_per_trial_save_path, 'w')
    spikes_per_trial_h5.create_dataset(name='Y', data=spikes_per_trial[n_val_trials:])
    spikes_per_trial_h5.create_dataset(name='Y_val', data=spikes_per_trial[:n_val_trials])
    spikes_per_trial_h5.close()


if __name__ == '__main__':
    main()
