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

def get_projection_matrix(num_neurons_1, num_neurons_2, connectivity):
    projection_matrix = np.zeros((num_neurons_1, num_neurons_2))
    for neuron in range(num_neurons_1):
        rand_sequence = np.random.rand(num_neurons_2)
        projection_matrix[neuron,np.where(rand_sequence<connectivity)[0]] = 1
    return projection_matrix

def remove_isi_violations(spike_trains, refractory_period):
    for cell in range(len(spike_trains)):
        spike_times = np.where(spike_trains[cell] == 1)[0]
        isi_violations = np.where(np.diff(spike_times) < refractory_period)[0] + 1
        spike_trains[cell,spike_times[isi_violations]] = 0


def generate_volume_simulation(exc_density, inh_density, firing_rate, dimensions, dx, time, spike_lfp, active):
    exc_neuron_coords, exc_volume_inds = generate_neurons_in_volume(exc_density, dimensions, dx)
    inh_neuron_coords, inh_volume_inds = generate_neurons_in_volume(inh_density, dimensions, dx)
    volume_lfp = np.zeros((len(volume_inds['x']), len(volume_inds['y']), len(volume_inds['z']), len(time)))

    exc_spike_trains = np.zeros((len(exc_neuron_coords), len(time)))
    exc_to_inh_projections = get_projection_matrix(len(inh_neuron_coords), len(exc_neuron_coords), connectivity)

    #Simulating fields of excitatory neurons
    for neuron in tqdm(range(len(exc_neuron_coords))):
        if np.random.rand(1) < active:
            exc_spike_trains[neuron] = generate_spike_train(time, exc_firing_rate)
            neuron_field = generate_volume_field_per_neuron(exc_spike_trains[neuron], exc_spike_lfp)
            volume_lfp = add_neuron_lfps_to_volume(neuron_field, exc_neuron_coords[neuron], exc_volume_inds, volume_lfp)
            neuron_field = []

    #Simulating fields of inhibitory num_neurons
    inh_spike_trains = np.matmul(exc_to_inh_connectivity, exc_spike_trains)
    inh_spike_trains[np.where(inh_spike_trains > 1)[0]] = 1
    inh_spike_trains = remove_isi_violations(inh_spike_trains, 64)

    for neuron in tqdm(range(len(inh_neuron_coords))):
        neuron_field = generate_volume_field_per_neuron(inh_spike_trains[neuron], inh_spike_lfp)
        volume_lfp = add_neuron_lfps_to_volume(neuron_field, inh_neuron_coords[neuron], inh_volume_inds, volume_lfp)
        neuron_field = []

    return volume_lfp
