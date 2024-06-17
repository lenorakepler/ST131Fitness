from pathlib import Path
import pandas as pd

from ecoli_analysis.random_effects_ecoli import do_crossval, analyze_fit, plot_random_branch_fitness

# ==================================================================================
# ~~~ Analysis-specific files and variables ~~~
#

data_dir = Path() / "data"

# Specify analysis name
# -----------------------------------------------------
analysis_name = "3-interval_constrained-sampling_test"

# Specify random effects (residual fitness) name
# -----------------------------------------------------
random_name = "Est-Random_Fixed-BetaSite_test"

# ~~~ End analysis-specific files and variables ~~~
# ==================================================================================

analysis_dir = data_dir / "analysis" / analysis_name
results = pd.read_csv(analysis_dir / "estimates.csv", index_col=0)
results = results.set_index("feature").squeeze()

b0 = results[[i for i in results.index if 'Interval' in i]].values
site = results[[i for i in results.index if 'Interval' not in i]].values.reshape(1, -1)

# Do cross-validation to determine optimal sigma hyperparameter
do_crossval(analysis_dir, random_name, est_site={'b0': b0, 'site': site})

# Using optimal sigma, compute residual branch fitness
analyze_fit(analysis_dir, random_name, est_site={'b0': b0, 'site': site})

# Plot residual branch fitness
plot_random_branch_fitness(analysis_dir, random_name, est_site=False)