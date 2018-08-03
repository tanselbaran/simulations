import numpy as np
from scipy import signal as sig
from tqdm import tqdm

def generate_spike_train(time, rate):
    poisson_rate = rate /len(time)
    spike_train = np.zeros(len(time))
    random = np.random.rand(len(time))
    spike_inds = np.where(random < poisson_rate)[0]
    spike_train[spike_inds] = 1

    return spike_train

def generate_volume_field_per_neuron(spike_train, spike_lfp):
    field_in_time = np.zeros((len(spike_lfp), len(spike_lfp[0]), len(spike_lfp[1]), len(spike_train)))
    spike_inds = np.where(spike_train == 1)[0]

    for spike in spike_inds:
        if spike < (len(spike_train) - (len(spike_lfp[0,0,0]))):
            field_in_time[:,:,:,spike:spike+len(spike_lfp[0,0,0])] = np.add(field_in_time[:,:,:,spike:spike+len(spike_lfp[0,0,0])], spike_lfp)

    return field_in_time

def generate_neurons_in_volume(density, dimensions, dx):
    x = np.arange(dimensions[0][0], dimensions[0][1], dx)
    y = np.arange(dimensions[1][0], dimensions[1][1], dx)
    z = np.arange(dimensions[2][0], dimensions[2][1], dx)

    volume_inds = {'x': x, 'y': y, 'z': z}
    volume = (x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0])
    num_neurons = int(density * volume)

    neuron_coords = np.zeros((num_neurons, 3))

    for neuron in range(num_neurons):
        neuron_coords[neuron,0] = int(np.random.rand(1) * len(x))
        neuron_coords[neuron,1] = int(np.random.rand(1) * len(y))
        neuron_coords[neuron,2] = int(np.random.rand(1) * len(z))

    return neuron_coords, volume_inds

def get_volume_bounds_and_neuron_lfp_bounds(neuron_coords, neuron_field, volume_lfp):
    volume_bounds = np.zeros((3,2))
    neuron_lfp_bounds = np.zeros((3,2))
    for dimension in range(3):
        volume_bounds[dimension,0] = max(neuron_coords[dimension] - neuron_field.shape[dimension]/2, 0)
        volume_bounds[dimension,1] = min(neuron_coords[dimension] + neuron_field.shape[dimension]/2, volume_lfp.shape[dimension])

        neuron_lfp_bounds[dimension,0] = neuron_field.shape[dimension] / 2 - neuron_coords[dimension]
        if neuron_lfp_bounds[dimension,0] < 0:
            neuron_lfp_bounds[dimension,0] = 0

        neuron_lfp_bounds[dimension,1] = neuron_field.shape[dimension]/2 + volume_lfp.shape[dimension] - neuron_coords[dimension]
        if neuron_lfp_bounds[dimension,1] > neuron_field.shape[dimension]:
            neuron_lfp_bounds[dimension,1] = neuron_field.shape[dimension]

        volume_bounds = volume_bounds.astype('int')
        neuron_lfp_bounds = neuron_lfp_bounds.astype('int')

    return volume_bounds, neuron_lfp_bounds

def add_neuron_lfps_to_volume(neuron_field, neuron_coords, volume_inds, volume_lfp):
    volume_bounds,neuron_lfp_bounds = get_volume_bounds_and_neuron_lfp_bounds(neuron_coords, neuron_field, volume_lfp)
    volume_lfp[volume_bounds[0,0]:volume_bounds[0,1], volume_bounds[1,0]:volume_bounds[1,1], volume_bounds[2,0]:volume_bounds[2,1],:] += neuron_field[neuron_lfp_bounds[0,0]:neuron_lfp_bounds[0,1], neuron_lfp_bounds[1,0]:neuron_lfp_bounds[1,1], neuron_lfp_bounds[2,0]:neuron_lfp_bounds[2,1]]

    return volume_lfp

def generate_volume_simulation(density, firing_rate, dimensions, dx, time, spike_lfp, active):
    neuron_coords, volume_inds = generate_neurons_in_volume(density, dimensions, dx)
    volume_lfp = np.zeros((len(volume_inds['x']), len(volume_inds['y']), len(volume_inds['z']), len(time)))

    spike_trains = np.zeros((len(neuron_coords), len(time)))
    for neuron in tqdm(range(len(neuron_coords))):
        if np.random.rand < active:
            spike_trains[neuron] = generate_spike_train(time, firing_rate)
            neuron_field = generate_volume_field_per_neuron(spike_trains[neuron], spike_lfp)
            volume_lfp = add_neuron_lfps_to_volume(neuron_field, neuron_coords[neuron], volume_inds, volume_lfp)
            neuron_field = []

    return volume_lfp
