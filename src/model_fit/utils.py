from pathlib import Path
import pandas as pd
import numpy as np
import re
# from model_fit.phylo_obj import PhyloObjPlain

"""
Functions for formatting data files as necessary for input into the model

** 
THESE DO NOT CURRENTLY PUT FILES IN THE RIGHT DIRECTORIES AND 
PROBABLY WILL NOT ACTUALLY RUN AS IS ... cleanup incoming.
**
"""

def concat_marginal_states(pastml_probabilities_dir, out_file):
	"""
	Input:  Directory containing pastml's marginal reconstructed
		    character probability files (usually /work)
	
	Output: Dataframe as CSV file with probability that a tree
			node (row) is in a given state (column)
	"""

	dir = Path(pastml_probabilities_dir)
	prob_files = list(dir.glob("marginal_probabilities.character_*.tab"))
	
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

	feature_type == "binary"
		formatted_feature_probability_df = pd.concat(formatted_feature_probability_dfs, axis=1)

	formatted_feature_probability_df.to_csv(out_file)

def marginal_for_analysis(pastml_dir, pastml_dict_file, meta=False, drop_first=False):
	dir = Path(pastml_dir)

	marginal_file = dir / "work" / "marginal_states.csv"
	
	if not marginal_file.exists():
		concat_marginal_states(dir / "work")

	mar = pd.read_csv(marginal_file, index_col=0)

	# We need to override marginal states of tips because it allows to be diff than specified
	features_file = list(dir.glob("tip_features_*"))[0]
	features = pd.read_csv(features_file, index_col=0)
	tip_features = features.loc[features.index.str.contains("SAMN") == True]

	if meta:
		# Convert tip states to dummy to match marginal
		tip_features = pd.get_dummies(tip_features, drop_first=drop_first).astype(int)

	missing = [i for i in tip_features.index if i not in mar.index]

	# Get only sampled tips
	tip_features = tip_features.loc[[i for i in tip_features.index if i in mar.index], :]

	# Save and remove dropped column from marginal
	removed = set(mar.columns.to_list()) ^ set(tip_features.columns.to_list())
	removed_bioproject = removed.pop()
	(dir / "reference_bioproject.txt").write_text(removed_bioproject)

	mar = mar[tip_features.columns]

	# Update marginal tips with known feature value
	mar.loc[tip_features.index, tip_features.columns] = tip_features

	if pastml_dict_file:
		# Change names back to allow special characters
		disp_names = load(Path(pastml_dict_file).read_text(), Loader=Loader)
		disp_names = {v: k for k, v in disp_names.items()}
		mar = mar.rename(columns=lambda c: disp_names[c])

	if meta:
		mar = mar.rename(columns=lambda c: c.rsplit("_", maxsplit=1)[-1] + "_META")

	if 'specimen' in tip_features.columns[0]:
		mar = mar.drop(columns=['urine_META'])

	mar.to_csv(dir / "marginal_states.csv")

# def make_intervals(interval_dir, original_tree_file, last_sample_date, *start_time_lists):
# 	"""
# 	Get all times, add root time if not there, sort,
# 	make interval tree, return interval list
# 	"""

# 	interval_dir = Path(interval_dir)
# 	print(start_time_lists)

# 	# -----------------------------------------------------
# 	# Load phylo obj, set dates
# 	# -----------------------------------------------------
# 	phylo_obj = PhyloObjPlain(
# 		tree_file=original_tree_file,
# 		tree_schema="newick",
# 	)
	
# 	for n in phylo_obj.tree.nodes():
# 		n.age = n.age + (last_sample_date - phylo_obj.present_time)

# 	phylo_obj.root = phylo_obj.tree.seed_node
# 	phylo_obj.root_time = phylo_obj.root.age - (phylo_obj.root.edge_length if phylo_obj.root.edge_length else 0)
# 	phylo_obj.present_time = last_sample_date

# 	# -----------------------------------------------------
# 	# Get list of all interval times, make interval tree
# 	# -----------------------------------------------------
# 	interval_times = sorted(list(set([phylo_obj.root_time] + [item for sublist in start_time_lists for item in sublist])))
	
# 	interval_dir.mkdir(exist_ok=True, parents=True)

# 	interval_tree = interval_dir / "phylo.nwk"
# 	phylo_obj.createIntervals(
# 		interval_times=interval_times,
# 		save_name=interval_tree,
# 		verbose=False,
# 	)

# 	np.savetxt(str(interval_dir / "interval_times.txt"), np.array(interval_times), delimiter=',')
	
# 	return interval_times, interval_tree

def get_bioproject_times_list(bioproject_times_file, changepoints_out_file):
	"""
	Just get list of all bioproject starts and ends, output to file
	Can put in config
	"""
	bioproject_times = pd.read_csv(bioproject_times_file, index_col=0)

	# Output a list of the sampling changepoints
	sampling_changepoints = list(set(bioproject_times["min_time"].to_list() + bioproject_times["max_time"].to_list()))
	Path(changepoints_out_file).write_text(",".join([str(s) for s in sampling_changepoints]))

def make_sampling_mask(bioproject_ancestral_file, bioproject_times_file, interval_times_file, mask_out_file, uncertainty=True):
	"""
	Given data object, set sampling upon removal rate for each
	phylogeny segment based on its bioproject
	"""

	# Read in interval times, but remove last
	interval_times = [float(t) for t in Path(interval_times_file).read_text().splitlines()]

	# Read in CSV of bioproject times
	bioproject_times = pd.read_csv(bioproject_times_file, index_col=0)
	bioproject_times = bioproject_times[["min_time", "max_time"]]
	bioproject_times.index = [i.split("_")[0] for i in bioproject_times.index]
	bioprojects = bioproject_times.index.to_list()

	# Get reconstructed bioproject feature states
	bp_anc = pd.read_csv(bioproject_ancestral_file, sep="\t")[['node', 'bioproject_id_META']]
	bp_anc = bp_anc.dropna(subset="bioproject_id_META")

	n_obs = len(bp_anc['node'].unique())
	n_int = len(interval_times)

	# Initialize matrix of zeroes with rows for each sample, column for each parameter interval
	s_arr = np.zeros((n_obs, n_int), dtype=float)

	# For each sample, figure out window in which sampling could have occurred based on the
	# first and last sample from that bioproject. 

	# If uncertain=True, we also take into account ancestral uncertainty in bioproject. 
	# Otherwise, we just use the first listed ancestral state
	if not uncertainty:
		bp_anc = bp_anc.loc[bp_anc.index.drop_duplicates(keep="first"), :]

	nodes = []
	for i, (sample, sdf) in enumerate(bp_anc.groupby('node')):
		nodes.append(sample)

		# sdf is a dataframe of all potential bioprojects, which we weight as equally likely
		prob = 1 / len(sdf) # TODO: BUG: WAIT WHY DO WE WAIT THESE AS EQUALLY LIKELY!?

		# For each potential bioproject, add marginal probability to time intervals that occur
		# during the window in which that project's sampling was taking place
		for r, row in sdf.iterrows():

			# Determine bioproject for sample
			proj = row['bioproject_id_META']

			# Get start and end of bioproject sampling
			if proj in bioproject_times.index:
				start, end = bioproject_times.loc[proj, :].apply(lambda x: interval_times.index(x)).values
				
				# Add to the mask
				s_arr[i, start:end] += prob

	s_df = pd.DataFrame(s_arr, columns=interval_times, index=nodes)
	s_df = s_df.drop(columns=[interval_times[-1]])
	s_df.to_csv(mask_out_file)

def concat_meta_features(out_file, *feature_files):
	df = pd.concat([pd.read_csv(f, index_col=0) for f in feature_files], axis=1)
	df.to_csv(out_file)

if __name__ == "__main__":

	pastml_work_dir = "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025 No Results/data_new/functional_groups_corr_pastml/work/"
	concat_marginal_states(pastml_work_dir, "")