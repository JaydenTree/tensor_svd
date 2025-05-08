import math
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import io as spio
import empad_func as pad
from tensor_svd_denoise import scree_plots, tensor_svd_denoise
import time

data_specs = [
    {
        'noisy_path': 'test_data/Simulation_noisy_SiDisl_slice_5_4000FPS_cropped_100layers.mat',
        'truth_path': 'test_data/Simulation_truth_SiDisl_slice_5_4000FPS_cropped_100layers.npy',
        'dataset': 'datacube'
    },
    {
        'noisy_path': 'test_data/Simulation_noisy_SiDisl_slice5_1000FPS_crpped_400layers.npy',
        'truth_path': 'test_data/Simulation_truth_SiDisl_slice5_1000FPS_crpped_400layers.npy'
    },
    {
        'noisy_path': 'test_data/Simulation_noisy_SiDisl_slice_5_1000FPS_cropped_100layers.npy',
        'truth_path': 'test_data/Simulation_truth_SiDisl_slice_5_1000FPS_cropped_100layers.npy'
    },
    {
        'noisy_path': 'test_data/Simulation_noisy_STO_slice_5_1000FPS_cropped_100layers.mat',
        'dataset': 'datacube'
    }
]

def load_data(path, data_spec=None):
    match(os.path.splitext(path)[1]): # File extension
        case '.npy':
            return np.load(path)
        case '.mat':
            f = spio.loadmat(path)
            keys = data_spec['dataset'].split('/')
            data = f[keys[0]]
            for key in keys[1:]:
                data = data[key]
            return data
        case '.h5':
            with h5py.File(path, "r") as f:
                keys = data_spec['dataset'].split('/')
                data = f[keys[0]]
                for key in keys[1:]:
                    data = data[key]
                return np.array(data)
        case '.raw':
            return pad.ReadRaw(path)
            
def display_data(data):
    fig, ax = plt.subplots()
    ax.imshow(data)
    fig.show()

def fold_data(data):
    if len(data.shape) != 3:
        return data.reshape(data.shape[0], data.shape[1], -1)
    return data

def calculate_rank(data, ndim=[100, 100, 100], dy_min=0.02):
    # Define the number of components that will be returned from scree_plots function, if ndim is not defined, ncomponents along
    # each dimension will be set to the full size of that dimension
    scree = scree_plots(data, ndim)

    ranks = [0] * len(scree)
    for i in range(len(scree)):
        logvars = np.log(scree[i])
        slopes = np.diff(logvars)
        slopethresh = np.percentile(np.abs(slopes), 30)
        for j in range(len(slopes) - 2):
            window = slopes[j:j+2] # Window to avoid outlier slopes
            slope = np.average(window)
            if abs(slope) < slopethresh: # Level slope
                ranks[i] = j
                break

    # for i in range(len(scree)):
    #     prev_logvar = np.log(scree[i][0])
    #     for j in range(1, len(scree[i])):
    #         logvar = np.log(scree[i][j])
    #         if prev_logvar - logvar > dy_min:
    #             ranks[i] = j
    #         prev_logvar = logvar

    return ranks

def calculate_PSNR(data_truth, data_noisy):
    mse = np.mean((data_truth - data_noisy) ** 2)
    max_intensity = np.max(data_truth)
    return 10 * np.log10((max_intensity ** 2) / mse)

def plot_denoised(data_denoised, data_noisy):
    # Plot the denoised result next to ground truth
    plt.figure(figsize=(10,10))

    plt.subplot(121)
    plt.imshow(data_denoised[:,:,10])
    plt.title('Denoised Data', fontsize = 16)

    plt.subplot(122)
    plt.imshow(data_noisy[:,:,10])
    plt.title('Noisy Data', fontsize = 16)

    plt.show()

def plot_all_denoised(noisy_image, denoised_images, labels, output="denoised.png"):
    # Plot the denoised result next to ground truth
    plt.figure(figsize=(10,10))

    num_plots = len(denoised_images) + 1
    cols = math.floor(num_plots ** 0.5)
    rows = math.ceil(num_plots ** 0.5)

    plt.subplot(rows, cols, 1)
    plt.imshow(noisy_image)
    plt.title('Noisy Data')

    for i, denoised in enumerate(denoised_images):
        plt.subplot(rows, cols, i + 2)
        plt.imshow(denoised)
        plt.title(labels[i])

    plt.savefig(output)
    plt.show()

def plot_benchmark(durations, psnrs, labels, output="benchmark.png"):
    plt.figure(figsize=(8,8))
    plt.plot(durations, psnrs, marker='o', linestyle="None")
    plt.title('PSNR vs Time', fontsize = 16)
    plt.xlabel("Duration (s)")
    plt.ylabel("PSNR (dB)")
    for i, label in enumerate(labels):
        plt.annotate(str(label), (durations[i], psnrs[i]))
    plt.grid(True)
    plt.savefig(output)
    plt.show()

def main():
    for data_spec in data_specs:
    # data_spec = data_specs[4]
        print("SVD:", data_spec)
        data_truth = fold_data(load_data(data_spec['truth_path'], data_spec)) if 'truth_path' in data_spec is not None else None
        data_noisy = fold_data(load_data(data_spec['noisy_path'], data_spec))
        print("Shape:", data_noisy.shape)

        durations = []
        psnrs = []
        denoised_images = []
        ranks = []
        scales = range(100, 20, -10)

        for scale in scales:
            Nx = int(data_noisy.shape[0] * (scale / 100))
            Ny = int(data_noisy.shape[1] * (scale / 100))
            data_cropped = np.copy(data_noisy[:Nx, :Ny, :])
            print("Done cropping:", data_cropped.shape)

            start = time.time()
            rank = calculate_rank(data_cropped, dy_min=0.005)
            # for i in range(len(rank) - 1):
            #     rank[i] = int(rank[i] * (100 / scale))
            data_denoised = tensor_svd_denoise(data_noisy, rank)
            end = time.time()
            duration = end - start
            print('Time elapsed:', "{:.2f}".format(duration), 'sec')
            durations.append(duration)

            denoised_images.append(data_denoised[:,:,10])

            ranks.append(rank)
            print("Rank:", rank)

            if data_truth is not None:
                psnr = calculate_PSNR(data_truth, data_denoised)
                psnrs.append(psnr)
                print("PSNR:", psnr)

        labels = [str(scales[i]) + "%, " + str(ranks[i]) for i in range(len(scales))]
        if len(psnrs) != 0:
            plot_benchmark(durations, psnrs, labels, data_spec["noisy_path"] + "-benchmark.png")        
            labels = [label + ", {:.2f} dB".format(psnrs[i]) for i, label in enumerate(labels)]
        plot_all_denoised(data_noisy[:,:,10], denoised_images, labels, data_spec["noisy_path"] + "-denoised.png")
    
if __name__ == "__main__":
    main()