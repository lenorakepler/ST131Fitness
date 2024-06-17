data_dir = config['data_dir']
interval_tree_name = config['interval_tree_name']
analysis_name = config['analysis_name']
residual_name = config['residual_name']

analysis_dir=f"{data_dir}/{analysis_name}"
residual_dir=f"{analysis_dir}/{residual_name}"
figures_dir = config['figures_dir']

# data, phylo_obj, RO, params, _ = load_data_and_RO_from_file(analysis_dir)
# train = RO.loadDataByIdx(RO.train_idx)

# estimates = RO.results_dict[model]["full"]["estimates"]
# bdm_params = RO.results_dict[model]["bdm_params"]
# h_combo = RO.results_dict[model]["full"]["h_combo"]
# features_file = params[model]

rule all:
	input:
		f"{figures_dir}/Figure-1_phylo_fitness_effect_matrix_no-bioproj.png",
		f"{figures_dir}/Figure-3_Time_Mean_Fitness_Decomp_Stacked-Line_intervallength-{interval_length}_cutoff-{interval_cutoff}.png",
		f"{figures_dir}/Figure-4_profile_CIs_boxplot_non-background_sig.png",
		f"{figures_dir}/Figure-5_Fitness-Components_ByClade_{'-'.join([c for c in config['fig5_categories']])}-{interval_length}_cutoff-{interval_cutoff}_combined-alt.png",
		f"{figures_dir}/Figure-6_Time_Variance_Decomp_Stacked_Abs-Prop_intervallength-{interval_length}_cutoff-{interval_cutoff}.png",

rule fig1_phylogeny_matrix:
	# -----------------------------------------------------
	# Fig 1. Maximum-likelihood time-calibrated ST131 
	# 		 phylogeny
	# -----------------------------------------------------
	input:
		clade_ancestral_file = f"{data_dir}/combined_ancestral_states.csv",
	output:
		f"{figures_dir}/Figure-1_phylo_fitness_effect_matrix_no-bioproj.png"
	python:
		from ecoli_analysis.feature_matrix import significant_phylo_matrix
		significant_phylo_matrix(data_dir, analysis_dir, input.clade_ancestral_file, figures_dir)

rule fig2_clade_sample_proportions:
	# -----------------------------------------------------
	# Fig 2. Proportion of sampled ST131 isolates from 
	# 		 each clade by year
	# -----------------------------------------------------
	output:
	python:
		from ecoli_analysis.plot_clades import plot_clade_sampling_over_time

		plot_clade_sampling_over_time(clade_ancestral_file, analysis_dir)

rule fig3_fig6_decomp_through_time:
	# -----------------------------------------------------
	# Fig 3. Changes in total ST131 fitness due to 
	# 		 different components of fitness through time
	#
	# Fig 6. Contribution of each model component to 
	# 		 fitness variation over time
	# -----------------------------------------------------
	input:
		f"{residual_dir}/variance_decomposition_intervallength-{interval_length}_cutoff-{interval_cutoff}.csv",
	output:
		f"{figures_dir}/Figure-3_Time_Mean_Fitness_Decomp_Stacked-Line_intervallength-{interval_length}_cutoff-{interval_cutoff}.png",
		f"{figures_dir}/Figure-6_Time_Variance_Decomp_Stacked_Abs-Prop_intervallength-{interval_length}_cutoff-{interval_cutoff}.png"
	python:
		from ecoli_analysis.fitness_decomp import components_by_clade

	do_plots(analysis_name, random_name, out_folder, interval_length=interval_length, interval_cutoff=interval_cutoff)

rule fig4_effects_boxplot:
	# -----------------------------------------------------
	# Fig 4. Box plot of genetic features with significant 
	# 		 estimated fitness effects
	# -----------------------------------------------------
	input:
		f"{analysis_dir}/profile_CIs.csv"
	output:
		f"{figures_dir}/Figure-4_profile_CIs_boxplot_non-background_sig.png"
	python:
		from ecoli_analysis.likelihood_profile import do_box_plots
		do_box_plots(analysis_dir, figures_dir)

rule fig5_amr_vir_by_clade:
	# -----------------------------------------------------
	# *Fig 5. Fitness contributions of AMR and virulence 
	# 		 through time
	# -----------------------------------------------------
	input:
		f"{residual_dir}/interval_times_intervallength-{interval_length}_cutoff-{interval_cutoff}.npy"
	output:
		f"{figures_dir}/Figure-5_Fitness-Components_ByClade_{'-'.join([c for c in config['fig5_categories']])}-{interval_length}_cutoff-{interval_cutoff}_combined-alt.png"
	python:
		from ecoli_analysis.fitness_decomp import components_by_clade

		components_by_clade(
			analysis_dir, 
			random_name,
			categories=categories,
			out_folder=figures_dir,
			interval_length=interval_length, 
			interval_cutoff=interval_cutoff,
			kind='combined-alt',
			return_fig=False,
		)

# -----------------------------------------------------
# Fig S1. Feature correlation
# -----------------------------------------------------


# -----------------------------------------------------
# Fig S2. Evolutionary history of GyrA
# -----------------------------------------------------
# plot_ancestral.py:plot_presences()
