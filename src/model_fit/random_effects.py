import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted, ns
from model_fit.results_obj import ResultsObj
import analysis.plot_phylo_standalone as pp

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

def next_parent(parent_row, fold_train_indices, branch_names):
	"""
	Go to the next parent if:
		- parent branch name not in training set
		- parent not in training set
	"""
	if parent_row["branch_name"] in branch_names:
		if parent_row["index"] in fold_train_indices:
			do_next = False
		else:
			do_next = True
	else:
		do_next = True

	# if not do_next:
	# 	dn = do_next
	# 	pn = parent_name
	# 	breakpoint()

	return do_next

def find_parents(data_df, fold_train_indices, indices_to_find, verbose=False):
	data_df["index"] = list(range(len(data_df)))
	data_df = data_df.set_index("idx")

	train_array = data_df.iloc[fold_train_indices, :]
	fold_array = data_df.iloc[indices_to_find, :]
	
	branch_names = natsorted(train_array["branch_name"].unique(), alg=ns.GROUPLETTERS)

	idx_to_name = {i: row["name"] for i, row in data_df[~data_df.index.duplicated()].iterrows()}
	idx_to_name[-1] = 'root_parent'

	parents_dict = []
	for i, (idx, row) in enumerate(fold_array.iterrows()):
		next_parent_idx = row['parent_idx']

		if verbose: print(f"\n===== {i}/{len(fold_array) + 1}: name: {row['name']} =====")
		if verbose: print(f"parent: {idx_to_name[row['parent_idx']]}")

		# If the parent index is not in the fold data set,
		# keep iterating through ancestors until we find the most
		# recent ancestor that is
		while next_parent_idx != -1:
			parent_idx = next_parent_idx
			parent_row = data_df.loc[parent_idx, :]

			# When have both node and edge, get whichever (if any) is in training set, since
			# we will have calculated that branch type.
			if isinstance(parent_row, pd.DataFrame):
				mod_parent_row = parent_row[parent_row['index'].isin(fold_train_indices)==True].copy()
				
				if isinstance(mod_parent_row, pd.Series):
					parent_row = mod_parent_row
				else:
					parent_row = parent_row.iloc[0]

			parent_name = parent_row['name']

			if next_parent(parent_row, fold_train_indices, branch_names):
				next_parent_idx = parent_row['parent_idx']
				if verbose: print(f"parent: {parent_name}, next parent: {next_parent_idx}")
			else:
				if verbose: print(f"parent: {parent_name} ({parent_idx}), next parent: NONE - breaking")
				break

		# Once we've found the parent node, get its event time
		# and calculate the time that has elapsed between it and
		# our branch's end time
		if next_parent_idx != -1:
			parent_time = parent_row['event_time']
			parent_name = parent_row['name']
		else:
			parent_idx = next_parent_idx
			parent_name = 'root'
			parent_row = row
			parent_time = parent_row['birth_time']

		parent_time_delta = row['event_time'] - parent_time
		parents_dict.append(
			{
				**row.to_dict(), 
				**dict(
					parent_name=parent_name, parent_idx=parent_idx, parent_index=parent_row["index"],
					parent_branch_name=parent_row["branch_name"],
					parent_time=parent_time, parent_time_delta=parent_time_delta,
					)
			}
		)

		if verbose: print(f"===> FINAL: name={row['name']}, true_parent={idx_to_name[row['parent_idx']]}, sigma_parent={parent_name}")

		# if (i % 100 == 0):
		# 	print(f"{i}/{len(fold_array) + 1}: name={row['name']}, parent_name={parent_name}, parent_idx={parent_idx}, parent_time={parent_time:.3f}, delta={parent_time_delta:.3f}")

	return pd.DataFrame(parents_dict)

def get_parent_type_info(data_df, fold_train_indices, fold_test_indices, all_branch_names):
	# Get indices of closest parent in training data set for
	# phylogeny pieces in both the training and test data sets
	train_df = find_parents(data_df, fold_train_indices, fold_train_indices)
	test_df = find_parents(data_df, fold_train_indices, fold_test_indices)

	train_df['type_int'] = [all_branch_names.index(i) for i in train_df['branch_name']]
	train_df['parent_type_int'] = [all_branch_names.index(i) for i in train_df['parent_branch_name']]
	test_df['type_int'] = [all_branch_names.index(i) for i in test_df['branch_name']]
	test_df['parent_type_int'] = [all_branch_names.index(i) for i in test_df['parent_branch_name']]

	train_dict = train_df.set_index("index").to_dict(orient="index")
	test_dict = test_df.set_index("index").to_dict(orient="index")

	return {"train": train_dict, "test": test_dict}

def split_intervals(all_data, train_idx, out_folder, n_folds, test_proportion, root_time, present_time, folds_start=None, alt=False):
	# This is stupid, there is no way it should be done like this,
	# but I don't want to rewrite stuff and it's late...
	
	# Create interval breakpoints and put into dictionary
	folds, interval_times = define_fold_intervals(n_folds, root_time, present_time, folds_start, test_proportion)

	# -----------------------------------------------------
	# Get indexes of phylogeny pieces in the train/test
	# datasets for each fold, plus get their self and/or
	# parental fitness index and time deltas
	# -----------------------------------------------------

	# Convert things to dataframes so they are much easier to work with...

	all_arr = pd.DataFrame(all_data.array)
	all_arr['index'] = list(range(len(all_arr)))
	
	if alt:
		all_arr["branch_name"] = all_arr["name"]
	else:
		all_arr["branch_name"] = all_arr["name"].apply(lambda n: n.split("_interval")[0])

	all_branch_names = natsorted(all_arr["branch_name"].unique(), alg=ns.GROUPLETTERS)

	train_arr = all_arr.loc[train_idx, :]
	
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
		
		fold_train_indices = fold_train_df['index'].tolist()
		fold_test_indices = fold_test_df['index'].to_list()

		# Get self and parent type info of the training data
		folds[i]['params'] = {k: v for k, v in folds_dict.items()}
		folds[i]['params']["n_types"]: int(len(all_branch_names))
		folds[i]['idxs'] = [fold_train_indices, fold_test_indices]
		folds[i].update(get_parent_type_info(all_arr, fold_train_indices, fold_test_indices, all_branch_names))
	
	fname = "brownian_search_setup.json"
	if alt:
		fname = fname.replace(".json", "_alt.json")

	(out_folder / fname).write_text(
		json.dumps(
			dict(
					n_folds=n_folds, 
					test_proportion=test_proportion,
					fold_start=folds_start,
					interval_times=interval_times,
					n_types = int(len(all_branch_names)),
					folds=folds,
					names=all_branch_names,
					int_to_name={i: name for i, name in enumerate(all_branch_names)},
				), indent=4,
			)
		)

def phylo_plot_in_train_test(tree_file, present_time, train_data, folds, out_file):
	# Load tree
	tt = pp.loadTree(
		tree_file,
		internal=True,
		abs_time=present_time
	)

	n_folds = len(folds)
	fig, axs = plt.subplots(1, n_folds, figsize=(12 * n_folds, 25))
	axs = axs.ravel()

	for fold_num, fold_dict in folds.items():
		test_start = fold_dict['params']['test']['start_time']
		test_end = fold_dict['params']['test']['end_time']

		fold_trait = {
			**{n['name']: "Train" for n in fold_dict['train'].values()},
			**{n['name']: "Test" for n in fold_dict['test'].values()},
		}

		colors, c_func = pp.categoricalFunc(fold_trait, 'name', legend=True, null_color="red")

		axs[fold_num] = pp.plotTraitAx(
			axs[fold_num],
			tt,
			edge_c_func=c_func,
			node_c_func=c_func,
			s_func=lambda x: 4,
			tip_names=False,
			zoom=False,
			title=f"Fold {fold_num}",
		)
		axs[fold_num].axvline(x=test_start, linestyle="--", color="red")
		axs[fold_num].axvline(x=test_end, linestyle="--", color="blue")

	pp.add_legend(colors, axs[fold_num], lloc="lower left")

	plt.tight_layout()
	plt.savefig(out_file, dpi=300)
	plt.close("all")

def prep_data_for_hyperparam_search(analysis_dir, n_folds=3, test_proportion=(1/2), folds_start=1960, plot=False, alt=False):
	"""
	Adds "parent_idx" to 
	"""

	# TODO: BUG: do I ever use this???

	RO = ResultsObj(folder=analysis_dir)
	data = RO.data

	# -----------------------------------------------------
	# Load / make time folds info
	# -----------------------------------------------------
	split_intervals(
		all_data=data, 
		train_idx=RO.train_idx, 
		out_folder=RO.folder,
		n_folds=n_folds,
		test_proportion=test_proportion, 
		root_time=data.root_time, 
		present_time=data.present_time, 
		folds_start=folds_start,
		alt=alt,
		)

	fname = "brownian_search_setup.json"
	if alt:
		fname = fname.replace(".json", "_alt.json")

	fold_params = json.loads((RO.folder / fname).read_text())
	folds = {int(i): v for i, v in fold_params['folds'].items()}

	if plot:
		phylo_plot_in_train_test(RO.params['tree_file'], data.present_time, data, folds, RO.folder / "brownian_search_setup.png")

	return RO, folds

def prep_data_for_fitting(analysis_dir, plot=True, alt=False):
	RO = ResultsObj(folder=analysis_dir)
	out_folder = RO.folder
	data = RO.data
	
	all_arr = pd.DataFrame(data.array)
	all_arr['index'] = list(range(len(all_arr)))

	if alt:
		all_arr["branch_name"] = all_arr["name"]
	else:
		all_arr["branch_name"] = all_arr["name"].apply(lambda n: n.split("_interval")[0])
	
	all_branch_names = natsorted(all_arr["branch_name"].unique(), alg=ns.GROUPLETTERS)

	brownian_info_dict = dict(
		n_types=int(len(all_branch_names)),
		names=all_branch_names,
		int_to_name={i: name for i, name in enumerate(all_branch_names)},
		)

	brownian_info_dict["full"] = get_parent_type_info(all_arr, RO.train_idx, RO.validate_idx, all_branch_names)
	for i, [train_idxs, test_idxs] in enumerate(RO.cv_idxs):
		brownian_info_dict[i] = get_parent_type_info(all_arr, train_idxs, test_idxs, all_branch_names)

	# TODO: BUG: What is alt??

	fname = "brownian_fit_setup.json"
	if alt:
		fname = fname.replace(".json", "_alt.json")

	(RO.folder / fname).write_text(json.dumps(brownian_info_dict, indent=4))

	if plot:
		plot_fname = fname.replace(".json", ".png")
		plot_dict = {i: {'params': {'test': {'start_time': data.root_time, 'end_time': data.present_time}}, **fold_dict} for i, fold_dict in brownian_info_dict.items() if isinstance(i, int)}
		phylo_plot_in_train_test(RO.params['tree_file'], data.present_time, data, plot_dict, RO.folder / plot_fname)

if __name__ == "__main__":
	pass

