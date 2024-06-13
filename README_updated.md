# ST131Fitness



## Running Analysis



### 1. Estimate the fitness effects of genetic mutations, background features, and time

analysis.py



### 2. Given time, genetic, and background effects, estimate the remaining (branch-specific) fitness not captured by the existing model

random_effects.py

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