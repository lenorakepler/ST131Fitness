from yte import process_yaml

# Snakemake should be able to handle this, but it doesn't seem to be working
config = process_yaml(Path("config.yaml").read_text())

data_dir = config['data_dir']
interval_tree_name = config['interval_tree_name']
analysis_name = config['analysis_name']
residual_name = config['residual_name']

analysis_dir=f"{data_dir}/analysis/{analysis_name}"
residual_dir=f"{analysis_dir}/{residual_name}"
figures_dir = config['figures_dir']

interval_length = config["interval_length"]
interval_cutoff = config["interval_cutoff"]

rule all:
	input:
		f"{figures_dir}/Figure-1_phylo_fitness_effect_matrix_no-bioproj.png",
		f"{figures_dir}/Figure-2_clade_count_time_stacked.png",
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
		clade_ancestral_file = config["clades_file"]
	output:
		f"{figures_dir}/Figure-1_phylo_fitness_effect_matrix_no-bioproj.png"
	run:
		from ecoli_analysis.feature_matrix import significant_phylo_matrix
		significant_phylo_matrix(data_dir, analysis_dir, input.clade_ancestral_file, figures_dir)

rule fig2_clade_sample_proportions:
	# -----------------------------------------------------
	# Fig 2. Proportion of sampled ST131 isolates from 
	# 		 each clade by year
	# -----------------------------------------------------
	output:
	run:
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
		f"{figures_dir}/{residual_name}/Figure-3_Time_Mean_Fitness_Decomp_Stacked-Line_intervallength-{interval_length}_cutoff-{interval_cutoff}.png",
		f"{figures_dir}/{residual_name}/Figure-6_Time_Variance_Decomp_Stacked_Abs-Prop_intervallength-{interval_length}_cutoff-{interval_cutoff}.png"
	run:
		from ecoli_analysis.fitness_decomp import do_plots

		do_plots(analysis_dir, config["residual_name"], figures_dir, interval_length=interval_length, interval_cutoff=interval_cutoff)

rule fig4_effects_boxplot:
	# -----------------------------------------------------
	# Fig 4. Box plot of genetic features with significant 
	# 		 estimated fitness effects
	# -----------------------------------------------------
	input:
		f"{analysis_dir}/profile_CIs.csv"
	output:
		f"{figures_dir}/Figure-4_profile_CIs_boxplot_non-background_sig.png"
	run:
		from ecoli_analysis.likelihood_profile import do_box_plots

		do_box_plots(analysis_dir, figures_dir, config["category_info_file"])

rule fig5_amr_vir_by_clade:
	# -----------------------------------------------------
	# Fig 5. Fitness contributions of AMR and virulence 
	# 		 through time
	# -----------------------------------------------------
	input:
		f"{residual_dir}/interval_times_intervallength-{interval_length}_cutoff-{interval_cutoff}.npy",
		f"{residual_dir}/all_edges.csv",
	output:
		f"{figures_dir}/Figure-5_Fitness-Components_ByClade_{'-'.join([c for c in config['fig5_categories']])}-{interval_length}_cutoff-{interval_cutoff}_combined-alt.png"
	run:
		from ecoli_analysis.fitness_decomp import components_by_clade

		components_by_clade(
			residual_dir,
			clades_file=config["clades_file"],
			category_info_file=config["category_info_file"],
			categories=config["fig5_categories"],
			out_folder=figures_dir,
			interval_length=interval_length, 
			interval_cutoff=interval_cutoff,
			kind='combined-alt',
			return_fig=False,
		)
