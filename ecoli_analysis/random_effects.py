import json
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from transmission_sim.analysis.param_model import ParamComponent, ParamModel, ComponentSite
from transmission_sim.analysis.phylo_loss import PhyloLossIterative, conditional_decorator, use_graph_execution
from transmission_sim.analysis.arrayer import PhyloArrayer, PhyloData
from transmission_sim.analysis.phylo_obj import PhyloObj
from transmission_sim.analysis.optimizer import Optimizer
from transmission_sim.ecoli.plot_test_reg import get_true_effs
from transmission_sim.ecoli.results_obj import ComponentB02
from transmission_sim.ecoli.random_effects_classes import *

import transmission_sim.analysis.phylo_loss

if Path("/home/lenora/Dropbox").exists():
	dropbox_dir = Path("/home/lenora/Dropbox")

else:
	dropbox_dir = Path("/Users/lenorakepler/Dropbox")

def next_parent(parent_idx, parent_name, train_indices):
	"""
	Go to the next parent if:
		- interval in parent name
		- parent not in training set
		- not at root
	"""
	if 'interval' in parent_name:
		do_next = True
	else:
		if parent_idx in train_indices:
			do_next = False
		else:
			do_next = True

	# if not do_next:
	# 	dn = do_next
	# 	pn = parent_name
	# 	breakpoint()

	return do_next

def find_parents(train_indices, fold_data, data):
	full_array = data.array

	parent_idxs = []
	parent_deltas = []
	for i, row in enumerate(fold_data.array):
		next_parent_idx = row['parent_idx']

		# print(f"\n===== {i}/{len(fold_data.array) + 1}: name: {row['name']} =====")
		# print(f"next_parent idx: ({next_parent_idx})")

		# If the parent index is not in the fold data set,
		# keep iterating through ancestors until we find the most
		# recent ancestor that is
		while next_parent_idx != -1:
			parent_idx = next_parent_idx
			parent_row = full_array[full_array['idx'] == parent_idx][0]
			parent_name = parent_row['name']

			if next_parent(parent_idx, parent_name, train_indices):
				# print(f"parent: {parent_name} ({parent_idx}), next parent: {next_parent_idx}")
				next_parent_idx = parent_row['parent_idx']
			else:
				# print(f"parent: {parent_name} ({parent_idx}), next parent: NONE - breaking")
				break

		# Once we've found the parent node, get its event time
		# and calculate the time that has elapsed between it and
		# our branch's end time
		if next_parent_idx != -1:
			parent_time = parent_row['event_time']
		else:
			parent_idx = next_parent_idx
			parent_time = data.root_time
			parent_name = 'root'

		parent_time_delta = row['event_time'] - parent_time

		parent_idxs.append(parent_idx)
		parent_deltas.append(parent_time_delta)

		# print(f"===> FINAL: name={row['name']}, parent_name={parent_name}, parent_time={parent_time:.3f}, delta={parent_time_delta:.3f}")

		if (i % 100 == 0):
			print(f"{i}/{len(fold_data.array) + 1}: name={row['name']}, parent_name={parent_name}, parent_idx={parent_idx}, parent_time={parent_time:.3f}, delta={parent_time_delta:.3f}")

	return parent_idxs, parent_deltas

def get_parent_type_info(fold_train_data, fold_test_data, data):
	# Make a list of all the index numbers contained
	# in the training set. Note that we need to get unique
	# values because we have birth events and edges with the 
	# same idx. Each of these indices will be mapped to a different
	# estimated birth rate
	train_indices = np.append(-1, np.unique(fold_train_data.array['idx'])).tolist()

	# Get indices of closest parent in training data set for
	# phylogeny pieces in both the training and test data sets
	train_parent_idxs, train_parent_deltas = find_parents(train_indices, fold_train_data, data)
	test_parent_idxs, test_parent_deltas = find_parents(train_indices, fold_test_data, data)

	type_info_dict = dict(
		n_types=len(train_indices),
		train=dict(
			type_int=[train_indices.index(i) for i in fold_train_data.array['idx']],
			parent_type_int=[train_indices.index(i) for i in train_parent_idxs],
			parent_time_delta=[float(i) for i in train_parent_deltas],
			),
		test=dict(
			type_int=[train_indices.index(i) for i in test_parent_idxs],
			parent_time_delta=[float(i) for i in test_parent_deltas],
			),
		)
	return type_info_dict

def define_fold_intervals(n_folds, root_time, present_time):
	# -----------------------------------------------------
	# Create interval breakpoints and put into dictionary
	# -----------------------------------------------------
	if folds_start:
		start = folds_start
	else:
		start = root_time

	end = present_time

	# Get starting times of of folds
	fold_times = np.linspace(start, end, n_folds + 1)

	# Get length of first fold, separate that into 
	# test and train periods based on test_proportion
	fold_period = fold_times[1] - fold_times[0]
	test_period = fold_period * test_proportion
	train_period = fold_period - test_period

	# Make dictionary of fold values and interval times
	interval_times = []
	folds = {}
	for i, fold_time in enumerate(fold_times[0:-1]):
		folds[i] = {'train': {}, 'test': {}}

		folds[i]['train'] = {
			'start_time': fold_time,
			'end_time': fold_time + train_period,
			'idx': len(interval_times),
		}
		interval_times.append(fold_time)

		folds[i]['test'] = {
			'start_time': fold_time + train_period,
			'idx': len(interval_times),
		}
		interval_times.append(fold_time + train_period)

		if i != 0:
			folds[i-1]['test']['end_time'] = fold_time
	
	folds[i]['test']['end_time'] = present_time

	return folds

def split_intervals(all_unsegmented_data, all_data, train_data, out_folder, n_folds, test_proportion, root_time, present_time, folds_start=None):
	# This is stupid, there is no way it should be done like this,
	# but I don't want to rewrite stuff and it's late...
	# Read in un-segmented tree file
	
	# Create interval breakpoints and put into dictionary
	folds = define_fold_intervals(n_folds, root_time, present_time)

	# -----------------------------------------------------
	# Get indexes of phylogeny pieces in the train/test
	# datasets for each fold, plus get their self and/or
	# parental fitness index and time deltas
	#
	# We do this on the UNSEGMENTED tree file
	# -----------------------------------------------------

	# Convert things to dataframes so they are much easier to work with...
	unseg_arr = pd.DataFrame(all_unsegmented_data.array)
	train_arr = pd.DataFrame(train_data.array)

	# Add "branch name" column to segmented array
	train_arr["branch_name"] = [n.split("_")[0] for n in train_arr['name']]

	# Split the full, unsegmented data set into folds
	for i, folds_dict in folds.items():
		# Get times corresponding with the train/test
		# datasets for this interval
		train_interval = folds_dict['train']['idx']
		test_interval = folds_dict['test']['idx']
		
		# Add a column describing whether each piece is in the scope of the training set of this fold
		unseg_arr[f"{i}_train"] = unseg_arr['birth_time'] <= folds_dict['train']['end_time']

		# Add a column describing whether each piece is in the scope of the test set of this fold
		unseg_arr[f"{i}_test"] = (unseg_arr['birth_time'] > folds_dict['test']['start_time']) & (unseg_arr['birth_time'] < folds_dict['test']['end_time'])
		
		# Get subsets of unsegmented data corresponding to train, test
		unseg_train_idx = np.nonzero(unseg_arr[f"{i}_train"])
		unseg_test_idx = np.nonzero(unseg_arr[f"{i}_test"])
		unseg_fold_train = all_unsegmented_data.getSubArraySpecific(train_idx)
		unseg_fold_test = all_unsegmented_data.getSubArraySpecific(test_idx)

		# Get self and parent type info of the unsegmented data
		type_info_dict = get_parent_type_info(unseg_fold_train, unseg_fold_test, all_unsegmented_data)

		breakpoint()



		folds_dict['train']['data_idx'] = [int(i) for i in train_idx]
		folds_dict['test']['data_idx'] = [int(i) for i in test_idx]

		# Get fitness indices and time deltas

		

		folds_dict['train'] = {**folds[i]['train'], **type_info_dict['train']}
		folds_dict['test'] = {**folds[i]['test'], **type_info_dict['test']}
		folds_dict['n_types'] = type_info_dict['n_types']

	(out_folder / "fold_params.json").write_text(
		json.dumps(
			dict(
					n_folds=n_folds, 
					test_proportion=test_proportion,
					interval_times=interval_times,
					folds=folds,
				)
			)
		)

