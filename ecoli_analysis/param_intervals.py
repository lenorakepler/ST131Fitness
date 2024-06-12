from pathlib import Path
import pandas as pd
import numpy as np
import re
from numpy.lib import recfunctions as rfn
from transmission_sim.utils.commonFuncs import ppp
from transmission_sim.analysis.arrayer import PhyloArrayer
from transmission_sim.analysis.phylo_obj import PhyloObj
from transmission_sim.ecoli.prep_tree import plot_tree_intervals
from transmission_sim.ecoli.general import get_data

if Path("/home/lenora/Dropbox").exists():
	dropbox_dir = Path("/home/lenora/Dropbox")

else:
	dropbox_dir = Path("/Users/lenorakepler/Dropbox")

dir = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset/final"

# Get all times, add root time if not there, sort,
# make interval tree, return interval list
def make_intervals(interval_dir, *start_time_lists):
	if Path("/home/lenora/Dropbox").exists():
		dropbox_dir = Path("/home/lenora/Dropbox")

	else:
		dropbox_dir = Path("/Users/lenorakepler/Dropbox")

	dir = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset/final"

	# -----------------------------------------------------
	# Load phylo obj, set dates
	# -----------------------------------------------------
	features_file = dir / "combined_ancestral_states_binary_grouped_diverse_uncorr.csv"
	tree_file = dir / "named.tree_lsd.date.noref.pruned_unannotated.nwk"

	phylo_obj = PhyloObj(
		tree_file=tree_file,
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
	
	# -----------------------------------------------------
	# Plot interval tree
	# -----------------------------------------------------
	plot_tree_intervals(interval_tree, phylo_obj.present_time, interval_times)

	return interval_times, interval_tree

# Given data object, set sampling times based on 
# bioproject info
def constrain_sampling(data, phylo_obj, s):
	sample_times_df = pd.read_csv(dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset/final" / "bioproject_info.csv", index_col=0)

	bioprojs = [c for c in phylo_obj.features_df.columns if 'PRJNA' in c]
	bioproj_features = phylo_obj.features_df[bioprojs]
	sample_bioproj = bioproj_features.apply(lambda row: np.where(row==1)[0][0], axis=1).to_dict()

	n_obs = data.array.size
	n_int = len(data.param_interval_times)

	s_arr = np.zeros((n_obs, n_int))

	interval_times = [float(t) for t in (phylo_obj.tree_file.parent / "interval_times.txt").read_text().splitlines()]
	interval_idx = {i: idx for idx, i in enumerate(interval_times)}
	bp_loc = [i for i, n in enumerate(phylo_obj.feature_names) if 'PRJNA' in n]

	sample_df = sample_times_df[["min_time", "max_time"]]

	name_re = re.compile(r"_interval\d*$")

	for i, (sample, time) in enumerate(zip(data.array['name'], data.array['event_time'])):
		name = re.sub(name_re, "", sample)
		proj = bioprojs[sample_bioproj[name]]
		start, end = sample_df.loc[proj, :].apply(lambda x: interval_idx[x]).values
		s_arr[i, start:end] = s

	single_sample = list(np.choose(data.array['param_interval'], s_arr.T))

	names = ["s"] + [f"s_{i}" for i in range(s_arr.shape[1])]
	values = [single_sample] + [list(row) for row in s_arr.T]

	data.array = rfn.append_fields(
		data.array, 
		names,
		values, 
		dtypes=[float for _ in values],
		usemask=False
		)

	return data

if __name__ == "__main__":
	bioproject_info = pd.read_csv(dir / "bioproject_info.csv", index_col=0)

	# # -----------------------------------------------------
	# # Load and date tree, set time intervals
	# # -----------------------------------------------------
	# interval_times, interval_tree = make_intervals(
	# 	dir / "interval_trees" / "2003-2013-bioprojsampling",
	# 	bioproject_info['min_time'].to_list(), 
	# 	bioproject_info['max_time'].to_list(),
	# 	[2003, 2013],
	# 	)

	# # Set birth-death model parameters for data
	# first_sample_time = min([t.age for t in phylo_obj.tree.leaf_node_iter()])
	# 

	# bd_array_params = dict(
	# 	s=([s if t > (first_sample_time - 1) else 0 for t in interval_times], True),
	# 	d=(1, False),
	# 	gamma=(0, False),
	# 	rho=([0 for t in interval_times], True),
	# 	# b0=(1.2, False)
	# )
	# data.addArrayParams(**bd_array_params)
