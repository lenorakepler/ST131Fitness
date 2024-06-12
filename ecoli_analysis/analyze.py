from pathlib import Path
import pandas as pd
import numpy as np

# ==================================================================================
# ~~~ Analysis-specific files and variables ~~~
#

# Specify analysis name
name = "3-interval_constrained-sampling"

features_file = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/combined_ancestral_states_binary_grouped_diverse_uncorr.csv"
interval_tree = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/interval_trees/2003-2013-bioprojsampling" / "phylo.nwk"

# Specify birth-death rates/probabilities that are not being estimated
bd_array_params = dict(
	d=(1, False),
	gamma=(0, False),
	rho=(0, False),
)

constrained_sampling_rate = 0.0001

# Specify whether birth-death variables will be estimated
# and whether they are time varying
bdm_params = dict(
	b0=[True, True],
	site=[True, False],
	gamma=[False],
	d=[False],
	s=[False, True],
	rho=[False, True],
)

# Specify hyperparameters to test combinations of
hyper_param_values = dict(
  reg_type=['l1', 'l2'],
  lamb=[0, 25, 50, 100],
  offset=[1], # penalize values that are far from 1, since multiplicative
)

n_epochs = 20000
lr = 0.00005

# ~~~
# ==================================================================================


# -----------------------------------------------------
# Load or create results object (RO) + formatted data
# -----------------------------------------------------
out_dir = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset" / "final" / "analysis" / name

if (out_dir / "params.json").exists():
	data, phylo_obj, RO, params, out_dir = load_data_and_RO_from_file(name)
else:
	data, phylo_obj, RO, params, out_dir = load_data_and_RO(name, tree_file, features_file, bd_array_params, constrained_sampling_rate, n_epochs)

birth_rate_changepoints = [phylo_obj.root_time, 2003, 2013]

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
	b0_estimates = results['estimates']['b0:0']

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
	site_estimates = results['estimates']['site:0'][0]
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
