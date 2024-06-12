# ST131Fitness



## Running Analysis

### param_intervals.py

make_intervals(interval_dir, *start_time_lists)

```
# -----------------------------------------------------
# Split tree into time intervals at beginning and end 
# of each BioProject collection time (min and max of
# existing samples for that BioProject), plus at 2003
# and 2013.
# -----------------------------------------------------
bioproject_info = pd.read_csv(dir / "bioproject_times.csv", index_col=0)

interval_times, interval_tree = make_intervals(
  dir / "interval_trees" / "2003-2013-bioprojsampling",
  bioproject_info['min_time'].to_list(), 
  bioproject_info['max_time'].to_list(),
  [2003, 2013],
  )
```



### analyze.py

transmission_sim.utils.file_locs as locs: *remove/integrate in*
transmission_sim.utils.commonFuncs import ppp: *remove/integrate in*

ecoli.results_obj: *remove ResultsObj, only using ResultsObj2*

from transmission_sim.ecoli.general import get_data: *move this method to results_obj*



```
from transmission_sim.analysis.optimizer import Optimizer
from transmission_sim.analysis.param_model import Site
from transmission_sim.analysis.phylo_loss import PhyloLossIterative

features_file = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/combined_ancestral_states_binary_grouped_diverse_uncorr.csv"
interval_tree = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/interval_trees/2003-2013-bioprojsampling" / "phylo.nwk"

# Specify birth-death rates/probabilities that are not being estimated
bd_array_params = dict(
  d=(1, False), # removal rate is 1
  gamma=(0, False), # there is no migration
  rho=(0, False), # there is no concerted sampling
)

# Specify whether variables will be estimated
# and whether they are time varying
bdm_params = dict(
  b0=[True, True], # b0 is estimated and time-varying
  site=[True, False], # site fitness is estimated but constant
  gamma=[False], # migration is specified
  d=[False], # removal rate is specified
  s=[False, True], # sampling rate is specified and time-varying
  rho=[False, True], # concerted sampling rate is specified and time varying
)

# Specify hyperparameters to test combinations of
hyper_param_values = dict(
  reg_type=['l1', 'l2'],
  lamb=[0, 25, 50, 100],
  offset=[1], # penalize values that are far from 1, since multiplicative
)

# Run the analysis
cli_run_analysis(
  name="3-interval_constrained-sampling",
  tree_file=interval_tree,
  features_file=features_file,
  bd_array_params=bd_array_params,
  constrained_sampling_rate=0.0001,
  bdm_params=bdm_params,
  hyper_param_values=hyper_param_values,
  n_epochs=20000,
  lr=0.00005,
  )
```



### random_effects_ecoli.py

```
from transmission_sim.analysis.PhyloRegressionTree import PhyloBoost
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import transmission_sim.utils.plot_phylo_standalone as pp

analysis_name = "3-interval_constrained-sampling"
random_name = "Est-Random_Fixed-BetaSite"

analysis_dir = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset" / "final" / "analysis" / analysis_name
results = pd.read_csv(analysis_dir / "estimates.csv", index_col=0)
results = results.set_index("feature").squeeze()

b0 = results[[i for i in results.index if 'Interval' in i]].values
site = results[[i for i in results.index if 'Interval' not in i]].values.reshape(1, -1)

do_crossval(analysis_name, random_name, est_site={'b0': b0, 'site': site})
analyze_fit(analysis_name, random_name, est_site={'b0': b0, 'site': site})
plot_random_branch_fitness(analysis_name, random_name, est_site=False)
```



### likelihood_profile.py

```

```



## Figures and Plotting

branch_fitness.py



## Individual Figures

### Fig 1. Maximum-likelihood time-calibrated ST131 phylogeny

feature_matrix.py: significant_phylo_matrix()



### Fig 2. Proportion of sampled ST131 isolates from each clade by year

plot_clades.py:plot_clade_sampling_over_time()



### Fig 3. Changes in total ST131 fitness due to different components of fitness
through time

fitness_decomp.py:do_plots()



### Fig 4. Genetic features with significant estimated fitness effects

likelihood_profile.py:box_plot()



### Fig 5. Fitness contributions of AMR and virulence through time.

fitness_decomp.py:components_by_clade()



### Fig 6. Contribution of each model component to fitness variation over time

fitness_decomp.py:do_plots()



### Fig 7. Workflow for phylogenetic and ancestral feature reconstruction

flowchart.py:draw_ecoli_pipeline()



### S1 Fig. Feature correlation



### S2 Fig. Evolutionary history of GyrA

plot_ancestral.py:plot_presences()

```
tree_file = dir / "named.tree_lsd.date.noref.pruned_unannotated.nwk"

tt = pp.loadTree(
  tree_file,
  internal=True,
  abs_time=2023,
)

all_ancestral_file = final_dir / "features" / "combined_ancestral_states_binary.csv"
all_ancestral_features = all_ancestral.columns.to_list()

plot_presences(
  tt, 
  all_ancestral_file, 
  [f for f in all_ancestral_features if 'gyrA' in f], 
  out_dir / "binary" / "grouped", 
  fname=f"all_{feature}_changepoints",
  pastml_dir=None, 
  plot_changepoints=True,
)
```



### Table 1. Maximum-likelihood estimates of genetic features that were not dropped out of the model