from yte import process_yaml

# Snakemake should be able to handle this, but it doesn't seem to be working
config = process_yaml(Path("config.yaml").read_text())

analysis_name = config['analysis_name']
residual_name = config['residual_name']

data_dir = config['data_dir']
analysis_dir=f"{data_dir}/analysis/{analysis_name}"
residual_dir=f"{analysis_dir}/{residual_name}"
figures_dir=f"{analysis_dir}/figures"

interval_tree_name = config['interval_tree_name']
interval_length = config['interval_length']
interval_cutoff = config['interval_cutoff']

rule all:
	input:
		f"{analysis_dir}/profile_CIs.csv",
		f"{residual_dir}/total_decomp_fractions_intervallength-{interval_length}_cutoff-{interval_cutoff}.csv",
		f"{residual_dir}/all_edges.csv",
		f"{residual_dir}/edge_fitness_components.csv",

rule create_interval_tree:
	"""
	# --------------------------------------------------------------------------------
	Create a tree that is split at start and end of each bioproject collection window
	Plus where birth rate changepoints occur
	# --------------------------------------------------------------------------------
	"""
	input:
		original_tree_file = config['original_tree_file'],
		bioproject_times_file = config['bioproject_times_file'],
		features_file = config['features_file'],
	output:
		interval_tree_file = f"{data_dir}/interval_trees/{interval_tree_name}/phylo.nwk",
	run:
		import pandas as pd
		from ecoli_analysis.param_intervals import make_intervals

		bioproject_times = pd.read_csv(input.bioproject_times_file, index_col=0)

		interval_times, interval_tree = make_intervals(
			Path(output.interval_tree_file),
			Path(f"{data_dir}/interval_trees/{interval_tree_name}"),
			Path(input.features_file),
			bioproject_times['min_time'].to_list(), 
			bioproject_times['max_time'].to_list(),
			config['birth_rate_changepoints'],
		)

rule fit_model:
	# --------------------------------------------------------------------------------
	# Do the cross validation and fit the full training set using optimal hyperparams
	# --------------------------------------------------------------------------------
	input:
		interval_tree_file = f"{data_dir}/interval_trees/{interval_tree_name}/phylo.nwk",
		features_file = config['features_file'],
	output:
		estimates = f"{analysis_dir}/estimates.csv",
	params:
		bdm_params = config['bdm_params'],
		hyper_param_values = config['hyper_param_values'],
		n_epochs = config['n_epochs'],
		lr = config['lr'],
	run:
		from ecoli_analysis.fit_model import fit_model

		fit_model(analysis_dir, analysis_name, input.interval_tree_file, input.features_file, config)

rule residual_fitness:
	# --------------------------------------------------------------------------------
	# 
	# --------------------------------------------------------------------------------
	input:
		estimates=f"{analysis_dir}/estimates.csv"
	output:
		f"{residual_dir}/edge_random_effects.csv"
	run:
		from ecoli_analysis.random_effects_ecoli import do_crossval, analyze_fit, plot_random_branch_fitness
		import pandas as pd
		from pathlib import Path

		results = pd.read_csv(input.estimates, index_col=0)
		results = results.set_index("feature").squeeze()

		b0 = results[[i for i in results.index if 'Interval' in i]].values
		site = results[[i for i in results.index if 'Interval' not in i]].values.reshape(1, -1)

		# Do cross-validation to determine optimal sigma hyperparameter
		do_crossval(Path(analysis_dir), residual_name, 
			n_sigmas=config["n_sigmas"], sigma_start=config["sigma_start"], sigma_stop=config["sigma_stop"],
			est_site={'b0': b0, 'site': site}, 
			n_epochs=config["n_epochs"], lr=config["lr"])

		# Using optimal sigma, compute residual branch fitness
		analyze_fit(Path(analysis_dir), residual_name, est_site={'b0': b0, 'site': site}, n_epochs=config["n_epochs"], lr=config["lr"])

rule fitness_components:
	input:
		f"{analysis_dir}/estimates.csv",
		f"{residual_dir}/edge_random_effects.csv",
	output:
		f"{residual_dir}/all_edges.csv",
		f"{residual_dir}/edge_log_fitness_components.csv",
	run:
		from ecoli_analysis.fitness_decomp import calc_fitness_totals

		calc_fitness_totals(analysis_dir, residual_dir)

rule fitness_decomposition:
	# --------------------------------------------------------------------------------
	#
	# --------------------------------------------------------------------------------
	input:
		f"{analysis_dir}/estimates.csv",
		f"{residual_dir}/edge_random_effects.csv"
	output:
		f"{residual_dir}/edge_fitness_components.csv",
		f"{residual_dir}/variance_decomposition_intervallength-{interval_length}_cutoff-{interval_cutoff}.csv",
		f"{residual_dir}/total_decomp_fractions_intervallength-{interval_length}_cutoff-{interval_cutoff}.csv",
	params:
		interval_length = interval_length,
		interval_cutoff = interval_cutoff,
	run:
		from ecoli_analysis.fitness_decomp import do_decomp

		do_decomp(Path(analysis_dir), residual_name, total=True, interval_length=params.interval_length, interval_cutoff=params.interval_cutoff)
		do_decomp(Path(analysis_dir), residual_name, total=False, interval_length=params.interval_length, interval_cutoff=params.interval_cutoff)

rule calc_CIs:
	# --------------------------------------------------------------------------------
	# Create likelihood profiles to get 95% CIs
	# This takes a while but would be extremely easy to parallelize
	# --------------------------------------------------------------------------------
	input:
		estimates = f"{analysis_dir}/estimates.csv",
	output:
		f"{analysis_dir}/profile_CIs.csv"
	run:
		from ecoli_analysis.results_obj import load_data_and_RO_from_file, load_data_and_RO
		from ecoli_analysis.likelihood_profile import make_profiles, get_CIs

		estimating = {k: v for k, v in config["bdm_params"].items() if v[0] == True}
		result_key = ('+').join(sorted([f"{k}_TV" if (len(v) > 1 and v[1] == True) else k for k, v in estimating.items()]))

		data, phylo_obj, RO, params = load_data_and_RO_from_file(Path(analysis_dir))
		train = RO.loadDataByIdx(RO.train_idx)

		estimates = RO.results_dict[result_key]["full"]["estimates"]
		bdm_params = RO.results_dict[result_key]["bdm_params"]
		h_combo = RO.results_dict[result_key]["full"]["h_combo"]

		make_profiles(
			train, 
			estimates, 
			params["features_file"], 
			bdm_params, 
			h_combo, 
			params["birth_rate_idx"], 
			Path(analysis_dir), 
			Path(figures_dir),
			plot_effect_profiles=True
		)

		get_CIs(Path(analysis_dir))

