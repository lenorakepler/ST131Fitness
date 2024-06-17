data_dir = config['data_dir']
interval_tree_name = config['interval_tree_name']
analysis_name = config['analysis_name']
residual_name = config['residual_name']

analysis_dir=f"{data_dir}/{analysis_name}"
residual_dir=f"{analysis_dir}/{residual_name}"

rule create_interval_tree:
	"""
	# --------------------------------------------------------------------------------
	Create a tree that is split at start and end of each bioproject collection window
	Plus where birth rate changepoints occur
	# --------------------------------------------------------------------------------
	"""
	input:
		original_tree_file = config['original_tree_file'],
		bioproject_times_file = f"{data_dir}/interval_trees/{interval_tree_name}/phylo.nwk",
		features_file = config['features_file'],
	output:
		interval_tree_file = interval_tree_file,
	python:
		import pandas as pd
		from ecoli_analysis.param_intervals import make_intervals

		bioproject_times = pd.read_csv(input.bioproject_times_file, index_col=0)

		interval_times, interval_tree = make_intervals(
			Path(input.interval_tree_file),
			Path(input.original_tree_file),
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
		interval_tree_file = interval_tree_file,
		features_file = config['features_file'],
	output:
		estimates = f"{analysis_dir}/estimates.csv",
	params:
		bdm_params = config['bdm_params'],
		hyper_param_values = config['hyper_param_values'],
		n_epochs = config['n_epochs'],
		lr = config['lr'],
	python:
		from ecoli_analysis.results_obj import load_data_and_RO_from_file, load_data_and_RO

		# Load or create results object (RO) + formatted data
		# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
		if (analysis_dir / "params.json").exists():
			data, phylo_obj, RO, params, out_dir = load_data_and_RO_from_file(Path(analysis_dir))
		else:																											
			data, phylo_obj, RO, params, out_dir = load_data_and_RO(
				data_dir / "analysis",
				analysis_name, 
				interval_tree, 
				input.features_file,
				config['bd_array_params'], 
				bioproject_times_file, 
				constrained_sampling_rate, 
				birth_rate_changepoints, 
				n_epochs
				)

		# Do the cross-validation
		# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
		RO.crossvalidate(
			params.bdm_params, 
			params.hyper_param_values, 
			out_dir, phylo_obj.feature_names, 
			paramns.n_epochs, 
			params.lr
			)

		# Format and save the results of cross-validation
		# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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
		df.to_csv(out_dir / "estimates.csv")

		print("===================================================================")
		print(f"Best hyperparameters: {results['h_combo']}")
		print(f"Train loss: {results['train_loss']}, Test loss: {results['test_loss']}")
		print("===================================================================")

rule residual_fitness:
	# --------------------------------------------------------------------------------
	#
	# --------------------------------------------------------------------------------
	input:
		estimates=f"{analysis_dir}/estimates.csv"
	output:
		f"{residual_dir}/edge_random_effects.csv"
	python:
		from ecoli_analysis.random_effects_ecoli import do_crossval, analyze_fit, plot_random_branch_fitness
		import pandas as pd
		from pathlib import Path

		results = pd.read_csv(input.estimates, index_col=0)
		results = results.set_index("feature").squeeze()

		b0 = results[[i for i in results.index if 'Interval' in i]].values
		site = results[[i for i in results.index if 'Interval' not in i]].values.reshape(1, -1)

		# Do cross-validation to determine optimal sigma hyperparameter
		do_crossval(Path(analysis_dir), residual_name, est_site={'b0': b0, 'site': site})

		# Using optimal sigma, compute residual branch fitness
		analyze_fit(Path(analysis_dir), residual_name, est_site={'b0': b0, 'site': site})

rule fitness_decomposition:
	# --------------------------------------------------------------------------------
	#
	# --------------------------------------------------------------------------------
	input:
		f"{analysis_dir}/estimates.csv",
		f"{residual_dir}/edge_random_effects.csv"
	output:
		f"{residual_dir}/edge_fitness_components.csv"
		f"{residual_dir}/variance_decomposition_intervallength-{interval_length}_cutoff-{interval_cutoff}.csv",
		f"{residual_dir}/total_decomp_fractions_intervallength-{interval_length}_cutoff-{interval_cutoff}.csv",
	params:
		interval_length=config['interval_length'],
		interval_cutoff=config['interval_cutoff'],
	python:
		from ecoli_analysis.fitness_decomp import do_decomp

		do_decomp(analysis_name, residual_name, total=True, interval_length=params.interval_length, interval_cutoff=params.interval_cutoff)
		do_decomp(analysis_name, residual_name, total=False, interval_length=params.interval_length, interval_cutoff=params.interval_cutoff)

rule calc_CIs:
	# --------------------------------------------------------------------------------
	# Create likelihood profiles to get 95% CIs
	# This takes a while but would be extremely easy to parallelize
	# --------------------------------------------------------------------------------
	input:
		estimates = f"{analysis_dir}/estimates.csv",
	output:
		f"{analysis_dir}/profile_CIs.csv"
	python:
		from ecoli_analysis.results_obj import load_data_and_RO_from_file, load_data_and_RO
		from ecoli_analysis.likelihood_profile import make_profiles, get_CIs

		data, phylo_obj, RO, params, out_dir = load_data_and_RO_from_file(Path(analysis_dir))
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
			analysis_dir, 
			figures_dir, 
			plot_effect_profiles=True
		)

		get_CIs(analysis_dir)

