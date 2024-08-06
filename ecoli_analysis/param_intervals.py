from pathlib import Path
import pandas as pd
import numpy as np
import re
from numpy.lib import recfunctions as rfn
from analysis.phylo_obj import PhyloObj

def make_intervals(interval_dir, original_tree_file, features_file, *start_time_lists):
	"""
	Get all times, add root time if not there, sort,
	make interval tree, return interval list
	"""

	# -----------------------------------------------------
	# Load phylo obj, set dates
	# -----------------------------------------------------
	phylo_obj = PhyloObj(
		tree_file=original_tree_file,
		tree_schema="newick",
		features_file=features_file,
		params={}
	)
	# Date the tree
	last_sample_date = 2023
	
	for n in phylo_obj.tree.nodes():
		n.age = n.age + (last_sample_date - phylo_obj.present_time)

	phylo_obj.root = phylo_obj.tree.seed_node
	phylo_obj.root_time = phylo_obj.root.age - (phylo_obj.root.edge_length if phylo_obj.root.edge_length else 0)
	phylo_obj.present_time = last_sample_date

	# -----------------------------------------------------
	# Get list of all interval times, make interval tree
	# -----------------------------------------------------
	interval_times = sorted(list(set([phylo_obj.root_time] + [item for sublist in start_time_lists for item in sublist])))
	
	interval_dir.mkdir(exist_ok=True, parents=True)

	interval_tree = interval_dir / "phylo.nwk"
	phylo_obj.createIntervals(
		interval_times=interval_times,
		save_name=interval_tree,
	)

	np.savetxt(str(interval_dir / "interval_times.txt"), np.array(interval_times), delimiter=',')
	
	return interval_times, interval_tree

def constrain_sampling(data, phylo_obj, s, bioproject_times):
	"""
	Given data object, set sampling upon removal rate for each
	phylogeny segment based on its bioproject
	"""

	bioproject_times = pd.read_csv(bioproject_times, index_col=0)
	
	bioprojs = [c for c in phylo_obj.features_df.columns if 'PRJNA' in c]
	bioproj_features = phylo_obj.features_df[bioprojs]
	sample_bioproj = bioproj_features.apply(lambda row: np.where(row==1)[0][0], axis=1).to_dict()

	n_obs = data.array.size
	n_int = len(data.param_interval_times)

	s_arr = np.zeros((n_obs, n_int))

	interval_times = [float(t) for t in (phylo_obj.tree_file.parent / "interval_times.txt").read_text().splitlines()]
	interval_idx = {i: idx for idx, i in enumerate(interval_times)}
	bp_loc = [i for i, n in enumerate(phylo_obj.feature_names) if 'PRJNA' in n]

	sample_df = bioproject_times[["min_time", "max_time"]]

	name_re = re.compile(r"_interval\d*$")

	for i, (sample, time) in enumerate(zip(data.array['name'], data.array['event_time'])):
		name = re.sub(name_re, "", sample)
		proj = bioprojs[sample_bioproj[name]]
		start, end = sample_df.loc[proj, :].apply(lambda x: interval_idx[x]).values
		s_arr[i, start:end] = s

	single_sample = list(np.choose(data.array['param_interval'], s_arr.T))

	names = ["s"] + [f"s_{i}" for i in range(s_arr.shape[1])]
	values = [single_sample] + [list(row) for row in s_arr.T]

	constrained_sampling_rates = {'names': names, 'values': values}

	data.array = rfn.append_fields(
		data.array, 
		names,
		values, 
		dtypes=[float for _ in values],
		usemask=False
		)

	return data, constrained_sampling_rates