from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import transmission_sim.utils.plot_phylo_standalone as pp
import json
import numpy as np
import seaborn as sns
import yaml

from ecoli_analysis.utils import cat_display

def get_clade_changepoints(tt, data, clade_ancestral_file, out_file_changepoints):
	# Read in file of ancestral clade states, apply to data df
	anc_df = pd.read_csv(clade_ancestral_file, index_col=0).squeeze()
	data['clade'] = [anc_df[n.split("_")[0]] for n in data.index]

	# Find "changepoints" where clades switch
	changepoints = []
	for node in tt.Objects:
		if node.branchType == 'node':
			node_name = node.traits['name'].split("_")[0]
			node_clade = anc_df[node_name]
			
			for child in node.children:
				child_name = child.traits['name'].split("_")[0]
				child_clade = anc_df[child_name]

				if child_clade != node_clade:
					change_date = node.absoluteTime
					changepoints.append(dict(
						change_date=change_date,
						parent_clade=node_clade,
						child_clade=child_clade,
						parent_name=node_name,
						child_name=child_name,
						str=f"{change_date:.1f}: {node_clade} -> {child_clade}",
						leaves=len(child.leaves) if child.branchType=="node" else 0,
						))

	# Save changepoints to csv
	changepoints = pd.DataFrame(changepoints)
	changepoints.to_csv(out_file_changepoints, index=False)

	changepoints.set_index("parent_name", inplace=True)

	# Only display core divergences
	changepoints = changepoints[changepoints["leaves"] > 10]

	return data, changepoints

def load_info(analysis_dir):
	# -----------------------------------------------------
	# Load analysis parameters
	# -----------------------------------------------------
	params = json.loads((analysis_dir / "params.json").read_text())

	# -----------------------------------------------------
	# Load and sort estimates
	# -----------------------------------------------------
	est_df = pd.read_csv(analysis_dir / "estimates.csv", index_col=0)
	est_dict = est_df.set_index("feature").squeeze().to_dict()

	est_sites_dict = {k: v for k, v in est_dict.items() if "Interval" not in k}
	est_times_dict = {k: v for k, v in est_dict.items() if "Interval" in k}

	log_site_effs = np.log(list(est_sites_dict.values()))
	time_effs = list(est_times_dict.values())

	# -----------------------------------------------------
	# Load data object, extract feature types
	# -----------------------------------------------------
	data = pd.read_csv(analysis_dir / "data.csv")

	# Remove nodes, since we don't treat them differently
	data = data[data['event'] == 4]

	# -----------------------------------------------------
	# Get site fitness
	# -----------------------------------------------------
	feature_types = np.array([np.fromiter(ft, int) for ft in data['ft']])
	data['site_fitness'] = np.exp(np.matmul(feature_types, log_site_effs))
	
	# -----------------------------------------------------
	# Get time fitness
	# -----------------------------------------------------
	data['birth_interval'] = np.take(params['birth_rate_idx'], data['param_interval'])
	data['time_fitness'] = np.take(time_effs, data['birth_interval'])

	# -----------------------------------------------------
	# Calculate total fitness, set name as index
	# -----------------------------------------------------
	data["total_fitness"] = data["site_fitness"] * data["time_fitness"]
	data.set_index("name", inplace=True)

	# -----------------------------------------------------
	# Get features of sampled tips only
	# -----------------------------------------------------
	features = pd.read_csv(params["features_file"], index_col=0)
	sample_features = features.loc[[i for i in features.index if 'SAMN' in i], :]

	# -----------------------------------------------------
	# Get feature est., CI info, calc. whether significant
	# -----------------------------------------------------
	est = pd.read_csv(analysis_dir / f"profile_CIs.csv", index_col=0)

	# -----------------------------------------------------
	# Create tree object
	# -----------------------------------------------------
	tt = pp.loadTree(
		params['tree_file'],
		internal=True,
		abs_time=2023,
	)

	return data, params, sample_features, est, tt

def plot_phylo_matrix(tt, matrix, tree_c_func, m_cmap, m_vmin, m_vmax, changepoints, out_file, zoom=None, cbar=True, save=True):
	# -----------------------------------------------------
	# Set up a plot with a tree axis and a matrix axis
	# -----------------------------------------------------
	fig = plt.subplots(figsize=(50, 30), facecolor='w')
	gs = GridSpec(1, 2, width_ratios=[3, 7], wspace=.05)

	ax_tree = plt.subplot(gs[0])
	ax_matrix = plt.subplot(gs[1])

	n_traits = len(matrix.columns)

	# -----------------------------------------------------
	# Plot the phylogeny
	# -----------------------------------------------------
	ax_tree = pp.plotTraitAx(
		ax=ax_tree, 
		tt=tt, 
		edge_c_func=tree_c_func, 
		node_c_func=tree_c_func, 
		title="", 
		tips=False,
		zoom=zoom,
		)
	
	# -----------------------------------------------------
	# Annotate phylogeny with clade divergence times
	# -----------------------------------------------------
	# Annotate only branches that are in the 'changepoints' df
	target_func = lambda k: k.traits['name'] in changepoints.index.to_list()
	
	# Annotate with corresponding string from df
	text_func = lambda k: changepoints.loc[k.traits['name'], "str"]

	# Set x and y text coordinates, positioning
	text_x_attr = lambda k: k.absoluteTime - 2
	text_y_attr = lambda k: k.y - 5
	kwargs = {'va':'top','ha':'right','size': 20}

	# Add text and divergence points
	tt.addText(ax_tree, x_attr=text_x_attr, y_attr=text_y_attr, target=target_func, text=text_func, **kwargs)
	tt.plotPoints(ax_tree, x_attr=lambda k: k.absoluteTime, y_attr=lambda k: k.y, target=target_func, size=36, colour="black")

	# -----------------------------------------------------
	# Make colorbar axis
	# -----------------------------------------------------
	if cbar:
		divider = make_axes_locatable(ax_matrix)
		cax = divider.append_axes('right', size='5%', pad=0.1)
		
	# -----------------------------------------------------
	# Plot the matrix
	# -----------------------------------------------------
	# Sort leaf names by y position
	leaves_sorted = sorted([k for k in tt.Objects if k.branchType=='leaf'], key=lambda k: k.y)
	leaf_names = [k.name for k in leaves_sorted]

	# Re-sort matrix by leaf's y position
	matrix = matrix.loc[leaf_names, :]

	# Set absence (0) to NA so will get masked (white) in heatmap
	matrix = matrix.replace(0, np.nan)

	sns.heatmap(matrix, cmap=m_cmap, vmin=m_vmin, vmax=m_vmax, yticklabels=False, cbar_ax=cax, ax=ax_matrix)

	# Make axis limits the same as phylogeny
	ax_matrix.set_ylim(ax_tree.get_ylim())

	# Add vertical white lines to give the impression of spacing
	# between features
	[ax_matrix.axvline(x, color='w') for x in range(n_traits)]

	# Set colorbar label size
	cbar_obj = ax_matrix.collections[0].colorbar
	cbar_obj.ax.tick_params(labelsize=20)

	# -----------------------------------------------------
	# Format axis labels, etc.
	# -----------------------------------------------------
	# Set x ticks, remove bottom, left, and right
	ax_matrix.set_xticks(np.arange(0.5, n_traits + 0.5), labels=[c for c in matrix.columns], rotation=45, ha="left")
	ax_matrix.set_xlim(0, n_traits)
	[ax_matrix.spines[loc].set_visible(False) for loc in ['right', 'left', 'bottom']]

	ax_matrix.tick_params(size=1, labelsize=20)
	ax_tree.set_yticklabels([])
	ax_matrix.xaxis.set_ticks_position('top')
	ax_tree.grid(axis='x')

	if save:
		plt.tight_layout()
		plt.savefig(out_file, dpi=300)
		plt.close("all")

	else:
		return plt, fig, ax_tree, ax_matrix

def feature_display(feature, feature_dict):
	f = feature.split("_")[0]
	finfo = feature_dict[f]
	return f"{finfo['display_name']} ({cat_display(finfo['category'].upper())})"

def significant_phylo_matrix(data_dir, analysis_dir, clade_ancestral_file, figure_out_dir):
	data, params, sample_features, est, tt = load_info(analysis_dir)

	# -----------------------------------------------------
	# Create matrix, extract branch fitness
	# -----------------------------------------------------
	# Sort matrix columns by absolute value of fitness
	est['mle_abs'] = est['mle'].abs()
	est.sort_values(by="mle_abs", ascending=False, inplace=True)

	# Subset features to only those that are significant
	sig_mle = est.loc[est['significant'] == True, 'mle'].squeeze()
	matrix = sample_features[sig_mle.index]

	# Instead of a matrix of 0 and 1, make it a matrix 
	# with 0 if the sample does not have a feature, and
	# the estimated feature fitness if it does
	# (multiply each column by estimated fitness)
	for feat, feat_fit in sig_mle.items():
		matrix.loc[:, feat] = matrix.loc[:, feat] * feat_fit

	# Transform feature names to their display names
	dnames = yaml.load((data_dir / "group_short_name_to_display_manual.yml").read_text())
	matrix = matrix.rename(columns={f: feature_display(f, dnames) for f in matrix.columns})

	# Extract branch fitness
	branch_fitness = data['site_fitness']

	# -----------------------------------------------------
	# Create normed color mapping centered at 1
	# for fitness estimates, create lambda function that
	# will get the color to plot
	# -----------------------------------------------------
	all_values = pd.concat([matrix, branch_fitness], axis=1)
	min_val = all_values.min().min()
	max_val = all_values.max().max()

	center = 1

	# Adjust min and max so color center is value center
	min_val, max_val = pp.get_center(center, min_val, max_val)

	# Set cmap with extremes
	cmap = sns.color_palette("coolwarm", as_cmap=True).with_extremes(bad='white', over='black')

	# sns will do normalization of heatmap, but need to set specifically for phylogeny
	norm = mpl.colors.Normalize(min_val, max_val)
	cmapper = lambda v: cmap(norm(v))
	tree_c_func = lambda k: cmapper(branch_fitness[k.traits['name']])
	
	# -----------------------------------------------------
	# Get clades and clade changepoints
	# ----------------------------------------------------- 
	clade_data, changepoints = get_clade_changepoints(tt, data, clade_ancestral_file, analysis_dir / "Clade_Changepoints.csv")
	
	# # Add clades to matrix
	# clades = pd.get_dummies(pd.Series({n: clade_data.loc[n, 'clade'] for n in matrix.index}, name="clade"))

	# # Replace "1" (presence) with 100 to enable correct coloring in heatmap
	# clades = clades.replace(1, 100)

	# # Add to existing matrix
	# matrix = pd.concat([clades, matrix], axis=1)

	# -----------------------------------------------------
	# Output the plot
	# -----------------------------------------------------
	genomic_only_matrix = matrix[[c for c in matrix.columns if 'Meta' not in c]]
	plot_phylo_matrix(tt, genomic_only_matrix, tree_c_func, cmap, min_val, max_val, changepoints, figure_out_dir / "Figure-1_phylo_fitness_effect_matrix_no-bioproj.png")
	plt.close("all")

	


