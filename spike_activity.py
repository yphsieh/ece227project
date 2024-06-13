import os
import math
import copy
import yaml
import scipy
import powerlaw
import datetime
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

from spike_metrics import *

def generate_izhikevich_neuronal_network(G, T=200000, tau=0.02, fe=400, fi=100):
    """
    Simulate the Izhikevich neuronal network using a given NetworkX directed graph G.
    
    Parameters:
    - G: A directed NetworkX graph.
    - T: Total time steps for the simulation.
    - tau: Time step (ms).
    - fe: Excitatory connection strength.
    - fi: Inhibitory connection strength.
    
    Returns:
    - v_vec: Voltage vector over time.
    - u_vec: Recovery variable vector over time.
    - spk_wave_raster: Raster plot data.
    - time_pad: Time padding information.
    """
    
    Ne = len(G.nodes)
    Ni = 0  # No inhibitory neurons in this example
    
    re = 1 * (1 - 2 * np.random.rand(Ne, 1))
    ri = .5 + .2 * (1 - 2 * np.random.rand(Ni, 1))
    a = np.vstack((0.02 * (np.ones((Ne, 1)) + np.random.rand(Ne, 1)), 0.02 + 0.08 * ri))
    b = np.vstack((0.2 * np.ones((Ne, 1)) + 0.01 * np.random.rand(Ne, 1), 0.25 - 0.05 * ri))
    c = np.vstack((-65 + 15 * re**2, -65 * np.ones((Ni, 1))))
    d = np.vstack((6 - 2 * re**2, 2 * np.ones((Ni, 1))))
    
    # Adjacency matrix
    A = nx.adjacency_matrix(G)
    S = fe * A - fi * sp.csr_matrix(([], ([], [])), shape=(Ne + Ni, Ne + Ni))
    
    # Simulation variables
    v = -70 * np.ones((Ne + Ni, 1))
    u = b * v
    v_vec = np.zeros((Ne + Ni, T + 1))
    u_vec = np.zeros((Ne + Ni, T + 1))
    v_vec[:, 0] = v[:, 0]
    u_vec[:, 0] = u[:, 0]
    I = np.zeros((Ne + Ni, 1))
    
    # Initial simulation loop
    for t in range(10000):
        if t % 50 == 0:
            I[:Ne] = 10 * np.random.randn(Ne, 1) * np.array([1] + [0] * (Ne - 1)).reshape(-1, 1)
            if Ni > 0:
                I[Ne:] = 2 * np.random.randn(Ni, 1)
        fired = np.where(v >= 30)[0]
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]
        I += 0 * np.sum(S[:, fired], axis=1).reshape(-1, 1)
        v += tau * 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
        v += tau * 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
        u += tau * a * (b * v - u)
        v_vec[:, t + 1] = v[:, 0]
        u_vec[:, t + 1] = u[:, 0]
    
    # Main simulation loop
    TT1 = 0
    TT2 = round(np.random.rand() * T / Ne)
    nsource_vec = np.random.permutation(Ne)
    for _ in range(10):
        nsource_vec = np.concatenate((nsource_vec, np.random.permutation(Ne)))
    nsource = nsource_vec[0]

    kk = 1
    while True:
        for t in range(TT1 + 1, TT2):
            if t % 10 == 0:
                I[:Ne] = 15 * np.random.randn(Ne, 1) * np.array([0] * (nsource) + [1] + [0] * (Ne - nsource - 1)).reshape(-1, 1)
                if Ni > 0:
                    I[Ne:] = 2 * np.random.randn(Ni, 1)
            fired = np.where(v >= 30)[0]
            v[fired] = c[fired]
            u[fired] = u[fired] + d[fired]
            I += np.sum(S[:, fired] * (1 - .8 * np.random.rand(Ne, len(fired))), axis=1).reshape(-1, 1)
            v += tau * 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
            v += tau * 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
            u += tau * a * (b * v - u)
            v_vec[:, t + 1] = v[:, 0]
            u_vec[:, t + 1] = u[:, 0]
        TT1 = TT2
        TT2 = min(round(np.random.rand() * T / Ne) + TT1, T)
        kk += 1
        nsource = nsource_vec[kk]
        if TT1 == T:
            break
    
    v_vec[v_vec < -50] = 0
    
    v_vec_raster = [np.zeros((T + 1, 1)) for _ in range(Ne)]
    for i in range(Ne):
        peaks = np.where(v_vec[i, :] > 0)[0]
        v_vec_raster[i][peaks] = 1
    time_pad = tau * (np.arange(T + 1))
    spk_wave_raster = [sp.csr_matrix(x) for x in v_vec_raster]
    
    return v_vec, u_vec, spk_wave_raster, time_pad


def time_averaging(date_vec):
    date_vec.sort()
    ave_idx_list = []
    startDate, endDate = date_vec[0].split('-')[0], date_vec[-1].split('-')[0]
    startDate, endDate = datetime.datetime.strptime(startDate, '%y%m%d'), datetime.datetime.strptime(endDate, '%y%m%d')

    print(date_vec, len(date_vec))
    print(startDate, endDate)
    
    currDate, startCurrDates = startDate, []
    while currDate < endDate:
        startCurrDates.append(currDate.strftime('%Y-%m-%d'))

        idx_list = []
        for i in range(len(date_vec)):
            date = date_vec[i].split('-')[0]
            date = datetime.datetime.strptime(date, '%y%m%d')
            
            if currDate > date: continue
            elif date < currDate + datetime.timedelta(days=21):
                idx_list.append(i)
            else:
                currDate += datetime.timedelta(days=11)
                i -= 1 # point to the last date included
                break

        ave_idx_list.append(idx_list)
        if i == len(date_vec)-1: break
    
    print(ave_idx_list)
    return ave_idx_list, startCurrDates


def setPltParams(): 
    # Set the default text font size
    plt.rc('font', size=5)
    # Set the axes title font size
    plt.rc('axes', titlesize=5)
    # Set the axes labels font size
    plt.rc('axes', labelsize=5)
    # Set the font size for x tick labels
    plt.rc('xtick', labelsize=5)
    # Set the font size for y tick labels
    plt.rc('ytick', labelsize=6)
    # Set the legend font size
    plt.rc('legend', fontsize=5)
    # Set the font size of the figure title
    plt.rc('figure', titlesize=6)


setPltParams()
remove_zeros = True
loadmetrics = False

pick_well = '32'
well_vec = [pick_well] #,'22','23','24','31','32','33','34']
date_vec = [f.split('_')[0] for f in os.listdir('/Users/melhsieh/Documents/muotrilab/MEA analysis/code/mat_9mo_may24') if (f.endswith('.mat') and f.find('Well' + pick_well) != -1)]
dlen, wlen = len(date_vec), len(well_vec)

date_regroup_idx, ave_date_vec = time_averaging(date_vec)

'''
if loadmetrics:
    with open('config_inout.yaml', "r") as fh:
        metrics = yaml.load(fh, Loader=yaml.SafeLoader)
else:
    metrics = {'n_edge': None} # n_edge


time_ave_metrics = copy.deepcopy(metrics)
for m in metrics.keys():
    metrics[m] = np.zeros((wlen, dlen))
    time_ave_metrics[m] = np.empty((wlen, len(date_regroup_idx)))

print(metrics.keys())
'''

def calculate_edge_probability(G):
    # Get the number of nodes
    num_nodes = len(G.nodes())

    # Get the number of edges
    num_edges = len(G.edges())

    # Calculate the total possible number of edges
    total_possible_edges = num_nodes * (num_nodes - 1) / 2

    # Calculate edge probability
    edge_probability = num_edges / total_possible_edges

    return edge_probability


# load all data
G_list = [[0] * dlen] * wlen

ifweight = False
for well in range(wlen):
    nxgraphs = []
    for date in range(0, dlen):
        # if ifweight: adjmat = scipy.io.loadmat(os.path.join('/Users/melhsieh/Documents/muotrilab/MEA analysis/code/mat_9mo', date_vec[date] + '_matfiles_WELL' + well_vec[well] + '.mat'))['EC_delay_adjmat']
        # else: adjmat = scipy.io.loadmat(os.path.join('/Users/melhsieh/Documents/muotrilab/MEA analysis/code/mat_9mo', date_vec[date] + '_matfiles_WELL' + well_vec[well] + '.mat'))['adjmat_']
        if ifweight: adjmat = scipy.io.loadmat(os.path.join('/Users/melhsieh/Documents/muotrilab/MEA analysis/code/mat_9mo', date_vec[date] + '_matfiles_WELL' + well_vec[well] + '.mat'))['EC_delay_adjmat']
        else: adjmat = scipy.io.loadmat(os.path.join('/Users/melhsieh/Documents/muotrilab/MEA analysis/code/mat_9mo', date_vec[date] + '_matfiles_WELL' + well_vec[well] + '.mat'))['adjmat_']
        
        G = nx.DiGraph()
        if ifweight:
            num_nodes = len(adjmat)
            for i in range(num_nodes):
                G.add_node(i)

            # Add edges with weights
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):  # Only look at upper triangle to avoid duplicating edges
                    if adjmat[i][j] != float('inf'):  # Only add an edge if there is a connection
                        G.add_edge(i, j, weight=1/adjmat[i][j]) # strength as the inverse of delays
        else:
            rows, cols = np.where(adjmat >= 1)
            edges = zip(rows.tolist(), cols.tolist())
            G.add_edges_from(edges)
            
        if nx.is_empty(G): continue
        
        nxgraphs.append(G)

        if date_vec[date] != '160810': continue
        print(date_vec[date])
        # Example usage
        try:
            print('generating...')
            v_vec, u_vec, spk_wave_raster, time_pad = generate_izhikevich_neuronal_network(G)

            # Plotting example neuron membrane potential
            plt.figure(figsize=(10, 6))
            plt.plot(time_pad, v_vec[0, :])
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane potential (mV)')
            plt.title('Membrane potential of neuron 1')
            plt.savefig(date_vec[date] + '_spike_activity_neuron1.png', dpi=400, format='PNG')

            # Raster plot
            plt.figure(figsize=(10, 6))
            for i, spk_train in enumerate(spk_wave_raster):
                spikes = spk_train.nonzero()[0]
                plt.vlines(spikes, i + 0.5, i + 1.5)
            plt.xlabel('Time (ms)')
            plt.ylabel('Neuron index')
            plt.title('Raster plot of neuronal network')
            plt.savefig(date_vec[date] + '_spike_activity_rastor.png', dpi=400, format='PNG')

            ## spike activity metrics
            firing_rates = compute_firing_rate(spk_wave_raster, time_pad)
            print(f"Firing rates: {np.mean(firing_rates)}")

            # sttc_matrix = compute_sttc(spk_wave_raster, dt=0.1)
            # print(f"Spike Time Tiling Coefficient (STTC) Matrix: {sttc_matrix}")

            isi_list = compute_isi(spk_wave_raster)
            cv_isi = compute_cv_isi(isi_list)
            print(f"Mean of ISI: {np.mean(cv_isi, axis=0)[0]}")
            print(f"Mean Coefficient of Variation (CV) of ISI: {np.mean(cv_isi, axis=0)[1]}")

            bursts = detect_bursts(spk_wave_raster, burst_threshold=50)
            print(f"Mean detected # of bursts: {np.mean([len(bursts_neuron) for bursts_neuron in bursts])}")

            synchrony_index = compute_synchrony_index(spk_wave_raster)
            print(f"Synchrony index: {synchrony_index}")
        except:
            continue