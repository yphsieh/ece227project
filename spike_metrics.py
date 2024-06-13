import os
import numpy as np
import scipy
import matplotlib.pyplot as plt

def compute_firing_rate(spk_wave_raster, time_pad):
    T = time_pad[-1]
    firing_rates = [len(spk) / T for spk in spk_wave_raster]
    return np.array(firing_rates)

def compute_sttc(spk_wave_raster, dt):
    Ne = len(spk_wave_raster)
    sttc_matrix = np.zeros((Ne, Ne))
    for i in range(Ne):
        for j in range(i+1, Ne):
            spikes_i = spk_wave_raster[i].nonzero()[0]
            spikes_j = spk_wave_raster[j].nonzero()[0]
            if len(spikes_i) > 0 and len(spikes_j) > 0:
                T_minus = sum([1 for t in spikes_i if any(np.abs(t - spikes_j) <= dt)])
                T_plus = sum([1 for t in spikes_j if any(np.abs(t - spikes_i) <= dt)])
                TA = T_minus / len(spikes_i)
                TB = T_plus / len(spikes_j)
                PA = len(spikes_i) / len(time_pad)
                PB = len(spikes_j) / len(time_pad)
                sttc_matrix[i, j] = (TA - PA) / (1 - PA * TB)
                sttc_matrix[j, i] = sttc_matrix[i, j]
    return sttc_matrix

def compute_isi(spk_wave_raster):
    isi_list = []
    for i, spk in enumerate(spk_wave_raster):
        spikes = spk.nonzero()[0]
        if len(spikes) > 1:
            isi_list.append(np.diff(spikes))
    return isi_list


def compute_cv_isi(isi_list):
    cv_isi = []
    for isi in isi_list:
        if len(isi) > 1:
            cv_isi.append([np.mean(isi), np.std(isi) / np.mean(isi)])
    return np.array(cv_isi)


def detect_bursts(spk_wave_raster, burst_threshold=0.1, data=False):
    bursts = []
    for i, spk in enumerate(spk_wave_raster):
        if not data: spikes = spk.nonzero()[0]
        else: spikes = spk
        bursts.append([])
        current_burst = []
        for j in range(1, len(spikes)):
            if spikes[j] - spikes[j - 1] <= burst_threshold:
                if len(current_burst) == 0:
                    current_burst.append(spikes[j - 1])
                current_burst.append(spikes[j])
            else:
                if len(current_burst) > 0:
                    bursts[i].append(current_burst)
                    current_burst = []
        if len(current_burst) > 0:
            bursts[i].append(current_burst)

    return bursts


def compute_synchrony_index(spk_wave_raster):
    Ne = len(spk_wave_raster)
    synchrony = np.zeros((Ne, Ne))
    for i in range(Ne):
        for j in range(i+1, Ne):
            spikes_i = spk_wave_raster[i].nonzero()[0]
            spikes_j = spk_wave_raster[j].nonzero()[0]
            if len(spikes_i) > 0 and len(spikes_j) > 0:
                synchrony[i, j] = len(set(spikes_i) & set(spikes_j)) / len(set(spikes_i) | set(spikes_j))
                synchrony[j, i] = synchrony[i, j]
    return np.mean(synchrony[np.triu_indices(Ne, 1)])

def load_all_raster(spk_folder):
    # load and plot all raster data
    nrow, ncol = 8,8 # it is 8x8, memory limit
    spk_wave_raster = []
    # fig, axs = plt.subplots(nrow, ncol, figsize=(10,10))
    for row in range(nrow):
        for col in range(ncol):
            try: data = scipy.io.loadmat(os.path.join(spk_folder,'raster_spkwave_WELL32_EL' + str(row+1) + str(col+1) + '.mat'))
            except: continue
            time_pad, spk_raster = data['time_pad'][:,0], data['spk_wave_raster'][0]

            time_pts = np.linspace(time_pad[0], time_pad[1], num=int(time_pad[2]), endpoint=True)
            for neu in spk_raster:
                neu = neu.nonzero()[0]
                spk_train = time_pts[neu]
                spk_wave_raster.append(spk_train)

    #         for neu in range(len(spk_raster[0])):
    #             axs[row, col].plot(spk_raster[0, neu], label = 'neu' + str(neu+1))
    #         axs[row, col].legend(loc = 'right')
    
    # plt.rc('font', size=5)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('all_spk_raster.png', format = 'PNG')
    return spk_wave_raster, time_pts

if __name__ == "__main__":
    date_vec = ['160824', '161001', '170310']
    for date in date_vec:
        spk_folder = "/Users/melhsieh/Documents/24 spring/ece 227/ece 227 project/WELL32_spk_" + date

        spk_wave_raster, time_pad = load_all_raster(spk_folder)

        # print(spk_wave_raster)
        # Raster plot
        plt.figure(figsize=(10, 6))
        for i, spikes in enumerate(spk_wave_raster):
            plt.vlines(spikes, i + 0.5, i + 1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron index')
        plt.title('Raster plot of neuronal network')
        # plt.savefig(date + '_spike_activity_rastor_data.png', dpi=400, format='PNG')

        ## spike activity metrics
        firing_rates = compute_firing_rate(spk_wave_raster, time_pad)
        print(f"Firing rates: {np.mean(firing_rates)}")

        # sttc_matrix = compute_sttc(spk_wave_raster, dt=0.1)
        # print(f"Spike Time Tiling Coefficient (STTC) Matrix: {sttc_matrix}")

        isi_list = compute_isi(spk_wave_raster)
        cv_isi = compute_cv_isi(isi_list)
        print(f"Mean of ISI: {np.mean(cv_isi[:, 0])}")
        print(f"Mean Coefficient of Variation (CV) of ISI: {np.mean(cv_isi[:, 1])}")

        bursts = detect_bursts(spk_wave_raster, burst_threshold=50, data=True)
        print(f"Mean detected # of bursts: {np.mean([len(bursts_neuron) for bursts_neuron in bursts])}")

        synchrony_index = compute_synchrony_index(spk_wave_raster)
        print(f"Synchrony index: {synchrony_index}\n")


