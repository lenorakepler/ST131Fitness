from pathlib import Path
import pandas as pd
import numpy as np
from ecoli_analysis.param_intervals import make_intervals
from ecoli_analysis.results_obj import load_data_and_RO_from_file, load_data_and_RO
from ecoli_analysis.likelihood_profile import make_profiles, get_CIs

# ==================================================================================
# ~~~ Analysis-specific files and variables ~~~
#

data_dir = Path() / "data"

# Specify analysis name
# -----------------------------------------------------
name = "3-interval_constrained-sampling_test"

# Specify the feature set we are going to be using
# -----------------------------------------------------
features_file = data_dir / "combined_ancestral_states_binary_grouped_diverse_uncorr.csv"

# -----------------------------------------------------
# We will split the original tree into time intervals 
# corresponding to the beginning and end of each 
# BioProject collection period (min and max of existing 
# samples for that BioProject), so that sampling can
# be constrained by these dates. We will also split at 
# 2003 and 2013, to give three time intervals with
# different background fitness rates that will be 
# estimated
# -----------------------------------------------------
original_tree_file = data_dir / "named.tree_lsd.date.noref.pruned_unannotated.nwk"
bioproject_times = pd.read_csv(data_dir / "bioproject_times.csv", index_col=0)
interval_tree_name = "2003-2013-bioprojsampling"
birth_rate_changepoints = [2003, 2013]

# Specify the probability of being sampled upon removal
# given that a sample is within a BioProject collection
# period
# -----------------------------------------------------
constrained_sampling_rate = 0.0001

# Specify other birth-death rates/probabilities that 
# are not being estimated
# -----------------------------------------------------
bd_array_params = dict(
	d=(1, False),
	gamma=(0, False),
	rho=(0, False),
)

# Specify whether birth-death variables will be 
# estimated and whether they are time varying
# -----------------------------------------------------
bdm_params = dict(
	b0=[True, True],
	site=[True, False],
	gamma=[False],
	d=[False],
	s=[False, True],
	rho=[False, True],
)

# Specify hyperparameters to test combinations of
# -----------------------------------------------------
# hyper_param_values = dict(
#   reg_type=['l1', 'l2'],
#   lamb=[0, 25, 50, 100],
#   offset=[1], # penalize values that are far from 1, since multiplicative
# )

hyper_param_values = dict(
  reg_type=['l1'],
  lamb=[1],
  offset=[1], # penalize values that are far from 1, since multiplicative
)

# Specify learning rate and number of epochs that the 
# optimizer will use
# -----------------------------------------------------
# n_epochs = 20000
# lr = 0.00005

n_epochs = 5
lr = 0.00005

# ~~~ End analysis-specific files and variables ~~~
# ==================================================================================

# -----------------------------------------------------
# Load or create interval tree
# -----------------------------------------------------
interval_tree = interval_tree = data_dir / "interval_trees" / interval_tree_name / "phylo.nwk"
bioproject_times_file = data_dir / "bioproject_times.csv"

if not interval_tree.exists():
	original_tree_file = data_dir / "named.tree_lsd.date.noref.pruned_unannotated.nwk"
	bioproject_times = pd.read_csv(bioproject_times_file, index_col=0)

	interval_times, interval_tree = make_intervals(
			data_dir / "interval_trees" / interval_tree_name,
			original_tree_file,
			features_file,
			bioproject_times['min_time'].to_list(), 
			bioproject_times['max_time'].to_list(),
	  )

# -----------------------------------------------------
# Load or create results object (RO) + formatted data
# -----------------------------------------------------
out_dir = data_dir / "analysis" / name

if (out_dir / "params.json").exists():
	data, phylo_obj, RO, params, out_dir = load_data_and_RO_from_file(out_dir)
else:																											
	data, phylo_obj, RO, params, out_dir = load_data_and_RO(data_dir / "analysis", name, interval_tree, features_file, bd_array_params, bioproject_times_file, constrained_sampling_rate, birth_rate_changepoints, n_epochs)

# -----------------------------------------------------
# Do the cross validation
# -----------------------------------------------------
RO.crossvalidate(bdm_params, hyper_param_values, out_dir, phylo_obj.feature_names, n_epochs, lr)

# -----------------------------------------------------
# Format and save the results of cross-validation
# -----------------------------------------------------
estimating = {k: v for k, v in bdm_params.items() if v[0] == True}
result_key = ('+').join(sorted([f"{k}_TV" if (len(v) > 1 and v[1] == True) else k for k, v in estimating.items()]))
results = RO.results_dict[result_key]["full"]

site_names = phylo_obj.features_df.columns
estimate_results = []

if bdm_params['b0'][0]:
	b0_estimates = results['estimates']['b0']

	if isinstance(b0_estimates, np.ndarray) or isinstance(b0_estimates, list):
		if constrained_sampling_rate:
			times_list = params["birth_rate_changepoints"]
		else:
			times_list = data.param_interval_times

		for i, (t, e) in enumerate(zip(times_list, b0_estimates)):
			if (ni := i + 1) != len(b0_estimates):
				t1 = data.param_interval_times[ni]
			else:
				t1 = phylo_obj.present_time
			estimate_results += [[f"Interval_{t:.0f}-{t1:.0f}", e]]

	else:
		estimate_results += [[f"b0", b0_estimates]]

if bdm_params['site'][0]:
	site_estimates = results['estimates']['site'][0]
	for e, t in zip(site_names, site_estimates):
		estimate_results += [[e, t]]

df = pd.DataFrame(estimate_results, columns=["feature", "estimate"])

print("===================================================================")
print("===================================================================")
print(f"Best hyperparameters: {results['h_combo']}")
print(f"Train loss: {results['train_loss']}, Test loss: {results['test_loss']}")
print("===================================================================")
print("===================================================================")

df.to_csv(out_dir / "estimates.csv")

# -----------------------------------------------------
# Create likelihood profiles to get 95% CIs
# -----------------------------------------------------
train = RO.loadDataByIdx(RO.train_idx)

estimates = RO.results_dict[result_key]["full"]["estimates"]
bdm_params = RO.results_dict[result_key]["bdm_params"]
h_combo = RO.results_dict[result_key]["full"]["h_combo"]

make_profiles(train, estimates, features_file, bdm_params, h_combo, params["birth_rate_idx"], analysis_dir, figures_dir, plot_effect_profiles=True)
get_CIs(analysis_dir)
