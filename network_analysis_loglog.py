#!/usr/bin/env python
# coding: utf-8

import os
import math
import copy
import yaml
import scipy
import powerlaw
import numpy as np
import datetime
import networkx as nx
import matplotlib.pyplot as plt



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

pick_well = '34'
well_vec = [pick_well] #,'22','23','24','31','32','33','34']
date_vec = [f.split('_')[0] for f in os.listdir('/Users/melhsieh/Documents/muotrilab/MEA analysis/code/mat_9mo_may24') if (f.endswith('.mat') and f.find('Well' + pick_well) != -1)]
dlen, wlen = len(date_vec), len(well_vec)

date_regroup_idx, ave_date_vec = time_averaging(date_vec)

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
#     fig, axes = plt.subplots(1,5, figsize=(15, 3))
    nxgraphs = []
    for date in range(dlen):
        if ifweight: adjmat = scipy.io.loadmat(os.path.join('/Users/melhsieh/Documents/muotrilab/MEA analysis/code/mat_9mo_may24', date_vec[date] + '_matfiles_WELL' + well_vec[well] + '.mat'))['EC_delay_adjmat']
        else: adjmat = scipy.io.loadmat(os.path.join('/Users/melhsieh/Documents/muotrilab/MEA analysis/code/mat_9mo_may24', date_vec[date] + '_matfiles_WELL' + well_vec[well] + '.mat'))['EC_adjmat']
        
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
            rows, cols = np.where(adjmat == 1)
            edges = zip(rows.tolist(), cols.tolist())
            G.add_edges_from(edges)
            
        if nx.is_empty(G): continue
        
        nxgraphs.append(G)
        
        ######################### log-log plot #######################
        deg_seq = [d for n, d in G.degree()] #, reverse=True)
        #deg_seq = [sum(weight for _, _, weight in G.edges(node, data='weight')) for node in G.nodes()]
        
        degree_count = np.unique(deg_seq, return_counts=True)
        degree_prob = degree_count[1] / len(G.nodes())

        # random network
        edge_prob = calculate_edge_probability(G)
        print(len(G.nodes()), edge_prob)
        random_G = nx.erdos_renyi_graph(len(G.nodes()), edge_prob/2)
        random_degree_sequence = sorted([d for n, d in random_G.degree()], reverse=True)
        random_degree_count = np.unique(random_degree_sequence, return_counts=True)
        random_degree_prob = random_degree_count[1] / len(random_G.nodes())

        
        # Plot the degree distribution
        plt.figure()
        plt.scatter(degree_count[0], degree_prob, marker='.', color='b')plt.xscale('log')
        plt.yscale('log')
        plt.title("Degree distribution (log-log plot)")
        plt.xlabel("Degree (log)")
        plt.ylabel("Probability (log)")


        fit = powerlaw.Fit(deg_seq, xmin=1)
        
        print(f'Estimated alpha: {fit.power_law.alpha}')
        print(f'xmin: {fit.power_law.xmin}')
        
        # Fit the exponential model
        fit_exp = powerlaw.Fit(deg_seq, distribution='exponential', xmin=1)

        # Print the estimated lambda for exponential
        print(f'Estimated lambda for exponential: {fit_exp.exponential.parameter1}')
        
        alpha_power_law = fit.power_law.alpha
        xmin_power_law = fit.power_law.xmin
        lambda_exponential = fit_exp.exponential.parameter1

        R, p = fit.distribution_compare('power_law', 'exponential')
        print(f'Log-likelihood ratio: {R}')
        print(f'p-value: {p}')
        
        fig = fit.power_law.plot_pdf(color='r', linestyle='--')
        fit_exp.exponential.plot_pdf(color='g', linestyle='--', ax=fig, label='Exponential fit')
        
        
        plt.text(0.55, 0.9, f'Estimated alpha for power-law: {alpha_power_law:.2f}', transform=plt.gca().transAxes)
        plt.text(0.55, 0.85, f'Estimated lambda for exponential: {lambda_exponential:.2f}', transform=plt.gca().transAxes)
        plt.text(0.55, 0.8, f'Log-likelihood ratio (power-law vs. exponential): {R:.2f}', transform=plt.gca().transAxes)
        plt.text(0.55, 0.75, f'p-value: {p:.4f}', transform=plt.gca().transAxes)


        plt.xlabel('Degree')
        plt.ylabel('Probability Density')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Degree Distribution with Power-Law and Exponential Fits')
        plt.legend(['Data', 'Power-law fit', 'Exponential fit'])
        
        plt.savefig('results_may24/EC/loglog/loglog_Well' + well_vec[well] + '_' + date_vec[date] + '.png', dpi=400, format='PNG', bbox_inches='tight', pad_inches=0.1)

        ###############################################################
        

        metrics['n_edge'][well, date] = G.number_of_edges()
    
    for i in range(1, len(nxgraphs)):
        dis = nx.graph_edit_distance(nxgraphs[i-1], nxgraphs[i])
        print(dis)


# time averaging within 21 days
for m in metrics.keys():
	for well in range(len(metrics[m])):
		for idx, indices in enumerate(date_regroup_idx):
			time_ave_metrics[m][well, idx] = np.mean(metrics[m][well, indices])

metrics_list = list(metrics.keys())

print(metrics_list)
plot_title_dict={'n_neuron': '# of nodes (active neurons)', \
                 'n_edge': '# of edges (connections)', \
                 'max_clstr': 'max of clustering coefficient', 'mean_clstr': 'ave. of clustering coefficient', 'min_clstr': 'min of clustering coefficient', \
                 'num_clstr': '# of neurons with > 0.9\nclustering coeff.', 'perc_clstr': '% of neurons with > 0.9\nclustering coeff.', \
                 'mean_non_zero_clstr': 'ave. of clustering coeff.\nafter removing zeros', \
                 'min_non_zero_clstr': 'min of clustering coeff.\nafter removing zeros', \
                 'num_zero_cluster': '# of neurons with zero clustering coeff.', \
                 'num_small_clstr': '# of neurons with < 0.2\nclustering coeff.', 'perc_small_clstr': '% of neurons with < 0.2\nclustering coeff.', \
                 'perc_non_zero_clstr': '% of > 0.9 \nafter removing zeros', 'perc_non_zero_small_clstr': '% of < 0.2 \nafter removing zeros'}

datetime_vec = [datetime.datetime.strptime(d,'%Y-%m-%d') for d in ave_date_vec]
startdate = datetime_vec[0]-datetime.timedelta(days=1)
datetime_vec = [int(str(date-startdate).split(' ')[0]) for date in datetime_vec]
print(datetime_vec)
                            
for w in range(wlen):
    np.set_printoptions(precision=2)
    
    #'''
    fig1, axes1 = plt.subplots(1,1,figsize=(5,5))
    fig2, axes2 = plt.subplots(1,1,figsize=(5,5))

    axes1.plot(metrics[m][w], linewidth=0.5)
    axes1.set_title(plot_title_dict[m])
    axes1.set_xticks(ticks = np.arange(0,len(date_vec), 5), labels=[date_vec[i] for i in range(len(date_vec)) if i % 5 == 0], rotation = 90)

    axes2.plot(datetime_vec, time_ave_metrics[m][w], linewidth=0.5)
    axes2.set_title(plot_title_dict[m]) # + '\n(averaged within 21 days)')
    axes2.set_xticks(np.arange(0, datetime_vec[-1], 30), labels=np.arange(10, datetime_vec[-1], 30), rotation=90)
    axes2.set_xlabel('Day')    


    nRes = 0
    
    fig1.savefig('results_may24/FC/neurons_wo_time_ave_well' + str(well_vec[w]) + '-' + str(nRes) + '.png', dpi=400, format='PNG', bbox_inches='tight', pad_inches=0.1)
    fig2.savefig('results_may24/FC/neurons_w_time_ave_well' + str(well_vec[w]) + '-' + str(nRes) + '.png', dpi=400, format='PNG', bbox_inches='tight', pad_inches=0.1)





