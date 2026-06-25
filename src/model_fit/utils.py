from pathlib import Path
import pandas as pd
import numpy as np
import re
from model_fit.phylo_obj import PhyloObj
import sys

"""
Functions for formatting data files as necessary for input into the model

** 
THESE DO NOT CURRENTLY PUT FILES IN THE RIGHT DIRECTORIES AND 
PROBABLY WILL NOT ACTUALLY RUN AS IS ... cleanup incoming.
**
"""

def concat_marginal_states(pastml_probabilities_dir, out_file):
	"""
	Input:	Directory containing pastml's marginal reconstructed
		    character probability files (usually /work)
	
	Output:	Dataframe as CSV file with probability that a tree
			node (row) is in a given state (column)
	"""

	dir = Path(pastml_probabilities_dir)
	prob_files = list(dir.glob("marginal_probabilities.character_*.tab"))
	
	if not prob_files:
		sys.exit(f"No pastml files matching 'marginal_probabilities.character_*.tab' found in {dir}. Exiting.")

	formatted_feature_probability_dfs = []
	for f in prob_files:
		feature_state_probabilities = pd.read_csv(f, sep="\t", index_col=0)

		# files named, e.g. params.character_aac6_AMR.method_MPPA.model_F81
		# so feature name is everything between "character_" and ".method"
		feature = re.search(r"character_(.*?)\.method", f.name).group(1)

		# If we have a binary feature (e.g. presence or absence)
		# pastml codes these states as '0' and '1' but we only
		# need to account for the probability that the feature
		# is present. 
		if feature_state_probabilities.columns.to_list() == ['0', '1']:
			feature_type = "binary"

			one_prob = feature_state_probabilities['1']
			one_prob.name = feature
			formatted_feature_probability_dfs.append(one_prob)

		# If multi-state (e.g. color), we need to account for 
		# all probabilities and columns should be named to
		# specify feature and feature state the probabilities
		# correspond to. (Also, should often then encode these
		# with a dummy variable, e.g. with utils.marginal_for_analysis())
		else:
			feature_type = "multi-state"

			renamed_columns = {c: f"{feature}_{state}" for state in fdf.columns}
			formatted_feature_probability_df = feature_state_probabilities.rename(columns=renamed_columns)

	if feature_type == "binary":
		formatted_feature_probability_df = pd.concat(formatted_feature_probability_dfs, axis=1)

	formatted_feature_probability_df.to_csv(out_file)

def marginal_for_analysis(name, pastml_dir, pastml_dict_file, meta=False, reference=False, drop_features=[], out_dir=""):
	dir = Path(pastml_dir)

	marginal_file = dir / "work" / "marginal_states.csv"
	
	if not marginal_file.exists():
		concat_marginal_states(dir / "work", dir / "work" / "marginal_states.csv")

	mar = pd.read_csv(marginal_file, index_col=0)

	# We need to override marginal states of tips because it allows to be diff than specified
	features_file = list(dir.glob("tip_features_*"))[0]
	features = pd.read_csv(features_file, index_col=0)
	tip_features = features.loc[features.index.str.contains("SAMN") == True]

	# Convert tip states to dummy to match marginal
	tip_features = pd.get_dummies(tip_features, drop_first=False).astype(int)

	missing = [i for i in tip_features.index if i not in mar.index]
	if missing:
		print(f"Samples not in marginal columns: {missing}")

	# Get only sampled tips
	tip_features = tip_features.loc[[i for i in tip_features.index if i in mar.index], :]

	mar = mar[tip_features.columns]

	# Update marginal tips with known feature value
	mar.loc[tip_features.index, tip_features.columns] = tip_features

	if meta:
		mar = mar.rename(columns=lambda c: c.rsplit("_", maxsplit=1)[-1] + "_META")

	if pastml_dict_file:
		# Change names back to allow special characters
		disp_names = load(Path(pastml_dict_file).read_text(), Loader=Loader)
		disp_names = {v: k for k, v in disp_names.items()}
		mar = mar.rename(columns=lambda c: disp_names[c])

	# Save and remove dropped column from marginal
	if reference:
		if reference not in mar.columns:
			sys.exit(f"{reference} not in feature columns. Exiting.\n{mar}")
		mar = mar.drop(columns=[reference])
		(dir / f"reference_{name}.txt").write_text(reference)

	if drop_features:
		not_in_columns = [c for c in drop_features if c not in mar.columns]
		if not_in_columns:
			sys.exit(f"Features to drop not found in columns: {not_in_columns}.\nCurrent states:\n{mar}\nExiting.")
		
		mar = mar.drop(columns=drop_features)
		(dir / f"dropped_features_{name}.txt").write_text(','.join(drop_features))

	if out_dir:
		out_dir = Path(out_dir)
	else:
		out_dir = dir

	mar.to_csv(out_dir / f"{name}_marginal_states_for_analysis.csv")

def make_intervals(interval_dir, original_tree_file, last_sample_date, *start_time_lists):
	"""
	Get all times, add root time if not there, sort,
	make interval tree, return interval list
	"""

	interval_dir = Path(interval_dir)
	interval_dir.mkdir(exist_ok=True, parents=True)

	# -----------------------------------------------------
	# Load phylo obj, set dates
	# -----------------------------------------------------
	phylo_obj = PhyloObj(
		tree_file=original_tree_file,
		tree_schema="newick",
	)
	
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
		verbose=False,
	)

	np.savetxt(str(interval_dir / "interval_times.txt"), np.array(interval_times), delimiter=',')
	
	return interval_times, interval_tree

def get_bioproject_times_list(bioproject_times_file, changepoints_out_file):
	"""
	Input:	CSV file with columns:
			- bioproject feature names (formatted with category name, e.g. PRJNA248737_META)
			- true_min_time (time of first sample in dataset from bioproject)
			- true_max_time (time of last sample in dataset from bioproject)
			- min_time (rounded time slightly before true min time)
			- max_time (rounded time slightly after true max time)

	Output: Concatenates min and max times to list, outputs as text file
	"""
	bioproject_times = pd.read_csv(bioproject_times_file, index_col=0)

	# Output a list of the sampling changepoints
	sampling_changepoints = sorted(set(bioproject_times["min_time"].to_list() + bioproject_times["max_time"].to_list()))
	Path(changepoints_out_file).write_text(",".join([str(s) for s in sampling_changepoints]))

def make_sampling_mask(bioproject_times_file, interval_times_file, reference_bioproject, mask_out_file):
	"""
	"""

	# Read in interval times, but remove last
	interval_times = [float(t) for t in Path(interval_times_file).read_text().splitlines()]

	# Read in CSV of bioproject times
	bioproject_times = pd.read_csv(bioproject_times_file, index_col=0)
	bioprojects = bioproject_times.index.to_list()

	n_int = len(interval_times)
	n_proj = len(bioprojects)

	# Initialize matrix of zeroes with rows for each sample, column for each parameter interval
	s_mask = pd.DataFrame(0, index=bioprojects, columns=interval_times, dtype=float)

	for t in interval_times:
		# bioprojects where this time point falls within active period (= or after start, = or before end)
		active = bioproject_times[(bioproject_times['min_time'] <= t) & (bioproject_times['max_time'] >= t)]
		print(f"{t}: {active.index.to_list()}")
		
		# set mask for these bioprojects at interval t to 1 to allow sampling to occur
		s_mask.loc[active.index, t] = 1

	s_mask = s_mask.drop(columns=[interval_times[-1]])

	# Drop reference bioproject
	s_mask = s_mask.drop(index=[reference_bioproject])

	out_dir = Path(mask_out_file).parent
	out_dir.mkdir(parents=True, exist_ok=True)
	s_mask.to_csv(mask_out_file)

def concat_meta_features(out_file, *feature_files):
	df = pd.concat([pd.read_csv(f, index_col=0) for f in feature_files], axis=1)
	df.to_csv(out_file)

# def inferred_dates(nex_tree, sample_dates_out_file, last_sample_out_file):
# 	tree = Path(nex_tree).read_text()
# 	tip_dates = re.findall(r"(SAMN\d*?)\[&date=(.*?)\]", tree)
# 	tip_dates = {s: float(d) for (s, d) in tip_dates if s in samples}
	
# 	last_sample_date = max(tip_dates.values())
# 	Path(last_sample_out_file).write_text(f"{last_sample_date}")

# 	breakpoint()

# def make_bioprojects_times_and_max_date(states_file, nex_tree, bp_times_out_file):
	

# 	PhyloObj(
# 		tree_file=tree_file,
# 		tree_schema="nexus",
# 		last_sample_date=last_sample_date,
# 	)
# 	breakpoint()
# 	states = pd.read_csv(states_file, index_col=0)

# 	# breakpoint()
# 	pass

def load_trees(**trees):
	from types import SimpleNamespace

	td = SimpleNamespace()

	for name, file in trees.items():
		file_type = 'newick' if 'nwk' in file else 'nexus'
		po = PhyloObj(tree_file=file, tree_schema=file_type)
		setattr(td, name, po)

	return td

if __name__ == "__main__":
	# bioproject_ancestral_file = "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/meta_features_bp/combined_ancestral_states.tab"
	# bioproject_times_file = "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/model_input/bioproject_times.csv"
	# interval_times_file = "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/interval_trees/2003-2013-bioprojsampling/interval_times.txt"
	# mask_out_file = "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/model_input/sampling_mask.csv"
	# reference_bioproject = "PRJNA248737_META"

	# states_file = "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/meta_features_bp/marginal_states.csv"
	# bp_times_out_file = "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/model_input/bioproject_times.csv"
	
	# tree_file = "/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/named.tree_lsd.date.noref.pruned.nwk"
	# nex_tree = "/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/lsd.date.nexus"
	# sample_dates_out_file = "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/processed/sample_dates.csv"
	# last_sample_out_file = "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/processed/last_sample_date.txt"
	# # inferred_dates(nex_tree, sample_dates_out_file, last_sample_out_file)
	# # make_bioprojects_times_file(states_file, nex_tree, bp_times_out_file)

	# # make_sampling_mask(bioproject_times_file, interval_times_file, reference_bioproject, mask_out_file)

	# td = load_trees(
	# 	final_named_tree_lsd_noref_pruned="/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/named.tree_lsd.date.noref.pruned.nwk",
	# 	final_lsd_date_nex="/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/lsd.date.nexus",
	# 	final_lsd_date_noref="/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/lsd.date.noref.nwk",
	# 	data_input_three_int_tree="/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/model_input/three_interval_tree.nwk",
	# 	data_3_int_tree_phylo="/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/3_interval_tree/phylo.nwk",
	# 	data_int_tree_20032013_phylo="/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/interval_trees/2003-2013-bioprojsampling/phylo.nwk",
	# 	final_ml_outlier_pruned="/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/ml_outlier_pruned.nwk"
	# 	)

	# samples = pd.read_csv("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/sample_info.csv", index_col=0)

	# inclist = samples[samples.exclusion_reason.isna() == True].index.to_list()
	# incset = set(inclist)

	# weird = []
	# for name, po in  td.__dict__.items():
	# 	po.leaf_nodes = [ln.taxon._label for ln in po.tree.leaf_nodes()]
	# 	n_samples=len(po.leaf_nodes)
	# 	not_in_tree = list(incset - set(po.leaf_nodes))
	# 	not_in_samples = list(set(po.leaf_nodes) - incset)
	# 	weird += not_in_tree
	# 	weird += not_in_samples
	# 	print(f"{name}: {n_samples} tips\n\tnot in tree: {not_in_tree}\n\textra: {not_in_samples}\n")
	
	# weird = list(set(weird))
	# notindf = [s for s in weird if s not in samples.index]
	# print(f"{notindf=}")

	# w = samples.loc[[s for s in weird if s in samples.index]]
	# print(w)

	# breakpoint()
	base_dir = Path("/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data")
	marginal_for_analysis(
		"bioproject",
		pastml_dir=base_dir / "meta_features_bp", 
		pastml_dict_file=None, 
		meta=True, 
		reference="PRJNA248737_META", 
		drop_features=["PRJNA587095_META", "PRJNA269984_META"], 
		out_dir=base_dir / "model_input"
		)

	marginal_for_analysis(
		"specimen",
		pastml_dir=base_dir / "meta_features_specimen", 
		pastml_dict_file=None, 
		meta=True, 
		reference="urine_META", 
		drop_features=[], 
		out_dir=base_dir / "model_input"
		)

	make_sampling_mask(
		base_dir / "model_input/bioproject_times.csv",
		base_dir / "model_input/interval_times.txt", 
		"PRJNA248737_META",
		base_dir / "model_input/sampling_mask.csv"
	)


