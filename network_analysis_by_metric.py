#!/usr/bin/env python
# coding: utf-8

import os
import math
import copy
import yaml
import scipy
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


# In[3]:


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

## better run well by well, change the if argument in date_vec too!
for pick_well in ['31', '32', '33', '34']:
	for metric_name in ['degree centrality', 'betweenness centrality', 'closeness centrality', 'eigenvector centrality', 'page rank', 'clustering coefficient']:

		well_vec = [pick_well] #,'31','32','33','34']
		date_vec = [f.split('_')[0] for f in os.listdir('/Users/melhsieh/Documents/muotrilab/MEA analysis/code/mat_9mo_may24') if (f.endswith('.mat') and f.find('Well'+ pick_well) != -1)]
		dlen, wlen = len(date_vec), len(well_vec)

		date_regroup_idx, ave_date_vec = time_averaging(date_vec)

		metrics = {'max_clstr': None, 'mean_clstr': None, 'min_clstr': None, \
		           'std_clstr': None, 'num_small_clstr': None, 'num_clstr': None } #, \
		           #'ent_clstr': None}


		# In[184]:


		time_ave_metrics = copy.deepcopy(metrics)
		for m in metrics.keys():
			metrics[m] = np.zeros((wlen, dlen))
			time_ave_metrics[m] = np.empty((wlen, len(date_regroup_idx)))

		print(metrics.keys())


		# In[185]:


		# load all data
		G_list = [[0] * dlen] * wlen


		for well in range(wlen):
		    fig, axes = plt.subplots(1,5, figsize=(15, 3))
		    d_idx = 0
		    all_metric = []
		    
		    for date in range(dlen):
		        adjmat = scipy.io.loadmat(os.path.join('/Users/melhsieh/Documents/muotrilab/MEA analysis/code/mat_9mo_may24', date_vec[date] + '_matfiles_WELL' + well_vec[well] + '.mat'))['FC_adjmat']
		        
		        rows, cols = np.where(adjmat == 1)
		        edges = zip(rows.tolist(), cols.tolist())
		        G = nx.DiGraph()
		        G.add_edges_from(edges)
		        if nx.is_empty(G): continue

		        n_nodes = G.number_of_nodes()
		        n_poss_edges = n_nodes*(n_nodes-1)

		        indegrees = np.array([G.in_degree(n)/(n_nodes-1) for n in G.nodes()]) #if G.in_degree(n) != 0 or not remove_zeros])
		        outdegrees = np.array([G.out_degree(n)/(n_nodes-1) for n in G.nodes()]) # if G.in_degree(n) != 0 or not remove_zeros])
		        sumdegrees = np.array([(G.in_degree(n) + G.out_degree(n))/(n_nodes-1) for n in G.nodes()])
		        cen_deg = np.fromiter(nx.degree_centrality(G).values(), dtype = float)
		        
		        clst_coeff = np.array(list(nx.clustering(G).values()))
		        
		        cen_btwn = np.fromiter(nx.betweenness_centrality(G).values(), dtype = float)
		        cen_close = np.fromiter(nx.closeness_centrality(G, wf_improved=True).values(), dtype = float)
		        cen_load = np.fromiter(nx.load_centrality(G).values(), dtype = float)
		        
		        try:
		            cen_eigen = np.fromiter(nx.eigenvector_centrality(G).values(), dtype = float)
		        except:
		            cen_eigen = np.array([0])
		        
		        pr = np.fromiter(nx.pagerank(G).values(), dtype = float)
		        

		        if metric_name == 'degree centrality': _metric = copy.deepcopy(cen_deg)
		        elif metric_name == 'betweenness centrality': _metric = copy.deepcopy(cen_btwn)
		        elif metric_name == 'closeness centrality': _metric = copy.deepcopy(cen_close)
		        elif metric_name == 'eigenvector centrality': _metric = copy.deepcopy(cen_eigen)
		        elif metric_name == 'page rank': _metric = copy.deepcopy(pr)
		        elif metric_name == 'clustering coefficient': _metric = copy.deepcopy(clst_coeff)
		        
		        hist_title = ['Day 24', 'Day 75', 'Day 128', 'Day 173', 'Day 228']
		        if date_vec[date] in ['160824', '161014', '161206', '170120', '170316-2'] : #% 10 == 4: 
		            axes[d_idx].hist(_metric, bins = 20)
		            x_ticks = np.arange(0,0.5, 0.1)
		            axes[d_idx].set_xticks(ticks = x_ticks, labels = [str("{:.1f}".format(lbl)) for lbl in x_ticks], rotation = 90)    
		            axes[d_idx].set_title(hist_title[d_idx])
		            d_idx = d_idx + 1

		        metrics['max_clstr'][well, date] = np.max(_metric)
		        metrics['mean_clstr'][well, date] = np.mean(_metric)
		        metrics['std_clstr'][well, date] = np.std(_metric)
		        metrics['min_clstr'][well, date] = np.min(_metric)
		        
		        
		        ten_perc_range = (np.max(_metric) - np.min(_metric))/10
		        
		        # number of neurons with mean + std values
		        cutoff = np.mean(_metric) + np.std(_metric)
		        # print(np.mean(_metric), np.mean(_metric) + np.std(_metric))
		        m_idx = np.where(_metric > cutoff)[0]
		        metrics['num_clstr'][well, date] = _metric[m_idx].sum()/max(1, len(m_idx))
		        metrics['num_small_clstr'][well, date] = len(m_idx)
		    
		    fig.savefig('results_may24/FC/' + metric_name + '_hist-well' + well_vec[well] + '.png', dpi=400, format='PNG', bbox_inches='tight', pad_inches=0.1)


		# In[186]:


		# time averaging within 21 days
		for m in metrics.keys():
			for well in range(len(metrics[m])):
				for idx, indices in enumerate(date_regroup_idx):
					time_ave_metrics[m][well, idx] = np.mean(metrics[m][well, indices])

		metrics_list = list(metrics.keys())
		'''
		metrics_list = ['_'.join(m.split('_')[1:]) for m in list(metrics.keys())]
		metrics_list.sort()

		for i, m in enumerate(metrics_list):
			if i%3 == 0:
				metrics_list[i] = 'max_' + metrics_list[i]
			elif i%3 == 1:
				metrics_list[i] = 'mean_' + metrics_list[i] 
			elif i%3 == 2:
				metrics_list[i] = 'min_' + metrics_list[i]
		'''

		print(metrics_list)
		plot_title_dict={'max_clstr': 'Max. of ' + metric_name, 'mean_clstr': 'Ave. of ' + metric_name, 'std_clstr': 'STD of ' + metric_name, 'min_clstr': 'Min. of ' + metric_name, \
		               'num_clstr': 'Ave. of neurons with ' + metric_name+ ' > (Mean+STD)', 'num_small_clstr': '# of neurons with ' + metric_name+ ' > (Mean+STD)', \
		                'ent_clstr': 'Entropy of ' + metric_name}

		datetime_vec = [datetime.datetime.strptime(d,'%Y-%m-%d') for d in ave_date_vec]
		startdate = datetime_vec[0]-datetime.timedelta(days=1)
		datetime_vec = [int(str(date-startdate+datetime.timedelta(days=10)).split(' ')[0]) for date in datetime_vec]

		all_datetime_vec = [datetime.datetime.strptime(d.split('-')[0],'%y%m%d') for d in date_vec]
		all_datetime_vec = [int(str(date-startdate).split(' ')[0]) for date in all_datetime_vec]

		for w in range(wlen):
			np.set_printoptions(precision=2)

			n_res = math.ceil(len(metrics.keys()) / 15)

			for nRes in range(n_res):
				fig1, axes1 = plt.subplots(3,math.ceil(len(metrics.keys())/3),figsize=(5,5))
				fig2, axes2 = plt.subplots(3,math.ceil(len(metrics.keys())/3),figsize=(5,5))
				mkeys = metrics_list[nRes*15:(nRes+1)*15]
		        
				for idx, m in enumerate(mkeys): 
					#print(m, '\n', metrics[m])
					#print(time_ave_metrics[m])

					axes1[idx%3, math.floor(idx/3)].plot(metrics[m][w], linewidth=0.5)
					axes1[idx%3, math.floor(idx/3)].set_title(plot_title_dict[m])
					axes1[idx%3, math.floor(idx/3)].set_xticks(ticks = np.arange(0,len(date_vec), 5), labels=[date_vec[i] for i in range(len(date_vec)) if i % 5 == 0], rotation = 90)

					axes2[idx%3, math.floor(idx/3)].scatter(all_datetime_vec, metrics[m][w], color = 'lightblue', s=2)
					axes2[idx%3, math.floor(idx/3)].plot(datetime_vec, time_ave_metrics[m][w], linewidth=0.5)
					axes2[idx%3, math.floor(idx/3)].set_title(plot_title_dict[m]) # + '\n(averaged within 21 days)')
					axes2[idx%3, math.floor(idx/3)].set_xticks(np.arange(0, datetime_vec[-1], 30), labels=np.arange(0, datetime_vec[-1], 30), rotation=90)
					axes2[idx%3, math.floor(idx/3)].set_xlabel('Day')
		            
				fig1.tight_layout()
				fig2.tight_layout()
		        
				fig1.savefig('results_may24/FC/results by metric/051424-'+ metric_name + '_wo_time_ave_well' + str(well_vec[w]) + '-' + str(nRes) + '.png', dpi=400, format='PNG', bbox_inches='tight', pad_inches=0.1)
				fig2.savefig('results_may24/FC/results by metric/051424-'+ metric_name + '_w_time_ave_well' + str(well_vec[w]) + '-' + str(nRes) + '.png', dpi=400, format='PNG', bbox_inches='tight', pad_inches=0.1)



