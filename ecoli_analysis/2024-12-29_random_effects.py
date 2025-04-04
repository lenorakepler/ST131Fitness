import json
import numpy as np
import pandas as pd
# from ecoli_analysis.random_effects_classes import *

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

	parents_dict = []
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

		parents_dict.append(dict(name=row['name'], idx=row['idx'], parent_name=parent_name, parent_idx=parent_idx, parent_time=parent_time, delta=parent_time_delta))
		# print(f"===> FINAL: name={row['name']}, parent_name={parent_name}, parent_time={parent_time:.3f}, delta={parent_time_delta:.3f}")

		# if (i % 100 == 0):
		# 	print(f"{i}/{len(fold_data.array) + 1}: name={row['name']}, parent_name={parent_name}, parent_idx={parent_idx}, parent_time={parent_time:.3f}, delta={parent_time_delta:.3f}")

	return pd.DataFrame(parents_dict)

def get_parent_type_info(fold_train_data, fold_test_data, data):
	# Make a list of all the index numbers contained
	# in the training set. Note that we need to get unique
	# values because we have birth events and edges with the 
	# same idx. Each of these indices will be mapped to a different
	# estimated birth rate
	train_indices = np.append(-1, np.unique(fold_train_data.array['idx'])).tolist()

	# Get indices of closest parent in training data set for
	# phylogeny pieces in both the training and test data sets
	train_df = find_parents(train_indices, fold_train_data, data)
	test_df = find_parents(train_indices, fold_test_data, data)

	# This edge, birth, death, and sampling events, but only need one entry per
	train_df = train_df[~train_df.duplicated(keep='first')]
	test_df = test_df[~test_df.duplicated(keep='first')]

	train_df['type_int'] = [train_indices.index(i) for i in train_df['idx']]
	train_df['parent_type_int'] = [train_indices.index(i) for i in train_df['parent_idx']]
	test_df['type_int'] = [train_indices.index(i) for i in test_df['parent_idx']]

	# This doesn't actually get used, but need for consistency
	test_df["parent_type_int"] = test_df['type_int']

	train_dict = train_df.set_index("name").to_dict(orient="index")
	test_dict = test_df.set_index("name").to_dict(orient="index")

	n_types = train_df[["type_int", "parent_type_int"]].max().max() + 1

	return {"train": train_dict, "test": test_dict, "n_types": int(n_types)}

def define_fold_intervals(n_folds, root_time, present_time, folds_start, test_proportion):
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

	return folds, interval_times

def split_intervals(all_data, train_idx, out_folder, n_folds, test_proportion, root_time, present_time, folds_start=None):
	# This is stupid, there is no way it should be done like this,
	# but I don't want to rewrite stuff and it's late...
	
	# Create interval breakpoints and put into dictionary
	folds, interval_times = define_fold_intervals(n_folds, root_time, present_time, folds_start, test_proportion)

	# -----------------------------------------------------
	# Get indexes of phylogeny pieces in the train/test
	# datasets for each fold, plus get their self and/or
	# parental fitness index and time deltas
	#
	# We do this on the UNSEGMENTED tree file
	# -----------------------------------------------------

	# Convert things to dataframes so they are much easier to work with...

	all_arr = pd.DataFrame(all_data.array)
	all_arr['idx'] = list(range(len(all_arr)))

	train_arr = all_arr.loc[train_idx, :]
	
	# Add "branch name" column to training data
	train_arr["branch_name"] = [n.split("_")[0] for n in train_arr['name']]

	# Split the training data set into folds
	for i, folds_dict in folds.items():
		# Get times corresponding with the train/test
		# datasets for this interval
		train_interval = folds_dict['train']['idx']
		test_interval = folds_dict['test']['idx']
		
		# Add a column describing whether each piece is in the scope of the training set of this fold
		fold_train_df = train_arr.loc[train_arr['birth_time'] <= folds_dict['train']['end_time'], :]

		# Add a column describing whether each piece is in the scope of the test set of this fold
		fold_test_df = train_arr.loc[(train_arr['birth_time'] > folds_dict['test']['start_time']) & (train_arr['birth_time'] < folds_dict['test']['end_time']), :]
		
		fold_train = all_data.getSubArraySpecific(fold_train_df['idx'])
		fold_test = all_data.getSubArraySpecific(fold_test_df['idx'])

		# Get self and parent type info of the training data
		folds[i]['params'] = {k: v for k, v in folds_dict.items()}
		folds[i]['idxs'] = [fold_train_df['idx'].tolist(), fold_test_df['idx'].tolist()]
		folds[i].update(get_parent_type_info(fold_train, fold_test, all_data))
		
	(out_folder / "brownian_search_setup.json").write_text(
		json.dumps(
			dict(
					n_folds=n_folds, 
					test_proportion=test_proportion,
					fold_start=folds_start,
					interval_times=interval_times,
					folds=folds,
				), indent=4,
			)
		)

if __name__ == "__main__":
	pass

