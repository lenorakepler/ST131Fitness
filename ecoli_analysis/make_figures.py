from pathlib import Path
from transmission_sim.ecoli.analyze import load_data_and_RO_from_file
from ecoli_analysis.likelihood_profile import do_box_plots
from ecoli_analysis.feature_matrix import significant_phylo_matrix

analysis_name = "3-interval_constrained-sampling"

data_dir = Path() / "data"
analysis_dir = data_dir / "analysis" / analysis_name
figures_dir = data_dir / "figures"

# data, phylo_obj, RO, params, _ = load_data_and_RO_from_file(analysis_dir)
# train = RO.loadDataByIdx(RO.train_idx)

# estimates = RO.results_dict[model]["full"]["estimates"]
# bdm_params = RO.results_dict[model]["bdm_params"]
# h_combo = RO.results_dict[model]["full"]["h_combo"]
# features_file = params[model]

# -----------------------------------------------------
# Fig 1. Maximum-likelihood time-calibrated ST131 
# 		 phylogeny
# -----------------------------------------------------
clade_ancestral_file = data_dir / "combined_ancestral_states.csv"
significant_phylo_matrix(analysis_dir, clade_ancestral_file, figures_dir)

# -----------------------------------------------------
# Fig 2. Proportion of sampled ST131 isolates from 
# 		 each clade by year
# -----------------------------------------------------
plot_clades.py:plot_clade_sampling_over_time()

# -----------------------------------------------------
# Fig 3. Changes in total ST131 fitness due to 
# 		 different components of fitness through time
# -----------------------------------------------------
# fitness_decomp.py:do_plots()

# -----------------------------------------------------
# Fig 4. Box plot of genetic features with significant 
# 		 estimated fitness effects
# -----------------------------------------------------
do_box_plots(analysis_dir, figures_dir)

# -----------------------------------------------------
# Fig 5. Fitness contributions of AMR and virulence 
# 		 through time
# -----------------------------------------------------
# fitness_decomp.py:components_by_clade()

# -----------------------------------------------------
# Fig 6. Contribution of each model component to 
# 		 fitness variation over time
# -----------------------------------------------------
# fitness_decomp.py:do_plots()

# -----------------------------------------------------
# Fig 7. Workflow for phylogenetic and ancestral 
# 		 feature reconstruction
# -----------------------------------------------------
# flowchart.py:draw_ecoli_pipeline()

# -----------------------------------------------------
# Fig S1. Feature correlation
# -----------------------------------------------------


# -----------------------------------------------------
# Fig S2. Evolutionary history of GyrA
# -----------------------------------------------------
# plot_ancestral.py:plot_presences()
