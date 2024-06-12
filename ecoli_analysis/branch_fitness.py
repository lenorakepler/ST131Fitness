from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import transmission_sim.utils.plot_phylo_standalone as pp
import json
import numpy as np
import re
import seaborn as sns
import matplotlib as mpl

if Path("/home/lenora/Dropbox").exists():
	dropbox_dir = Path("/home/lenora/Dropbox")

else:
	dropbox_dir = Path("/Users/lenorakepler/Dropbox")

ncbi = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset"
dir = ncbi / "final"

def color_tree_components(params, data, out_file_phylo, zoom=None):
	tt = pp.loadTree(
		Path(params['tree_file']),
		internal=True,
		abs_time=2023,
	)

	df = data
	vmax = df.max().max()
	vmin = df.min().min()

	plt.rcParams.update({'font.size': 20})
	n_cols = len(df.columns)
	fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(13 * n_cols, 30))
	
	if n_cols > 1:
		axs = axs.ravel()

	else:
		axs = [axs]

	for c, ax in zip(df.columns, axs):
		fit_dict = df[c].to_dict()

		# sns.color_palette("flare", as_cmap=True)
		c_func, cmap, norm = pp.continuousFunc(trait_dict=fit_dict, trait="name", cmap="viridis", vmin=vmin, vmax=vmax, norm="lognorm")
		
		ax = pp.plotTraitAx(
			ax,
			tt,
			edge_c_func=c_func,
			node_c_func=c_func,
			tip_names=False,
			zoom=zoom,
			title=c,
			tips=False,
		)

		ax.grid(axis='x',ls='--')

		# ax = pp.plotInternalNodes(tt, ax, c_func=lambda k: "red" if "interval" in k.traits["name"] else "black", outline_func=lambda k: "black", size_func=lambda k: 10)
		
		# ax = pp.set_vertical(ax, line_style=(0.0, [1, 3]), color="lightgray")

		# for it in params['interval_times']:
		# 	ax.axvline(x=it, linestyle="--")
		
	pp.add_cmap_colorbar(ax, cmap, norm=norm)
	
	plt.tight_layout()
	plt.savefig(out_file_phylo)

	mpl.rcParams.update(mpl.rcParamsDefault)

def color_tree(tree_file, fit_dict, out_file_phylo, center=False):
	tt = pp.loadTree(
		tree_file,
		internal=True,
		abs_time=2023,
	)

	c_func, cmap, norm = pp.continuousFunc(trait_dict=fit_dict, trait="name", cmap='coolwarm', center=center)

	fig, ax = plt.subplots(figsize=(12, 30))
	ax = pp.plotTraitAx(
		ax,
		tt,
		edge_c_func=c_func,
		node_c_func=c_func,
		tip_names=False,
		zoom=None,
		title="Estimated Branch Fitness",
	)
	
	# ax = pp.add_legend(clade_colors, "lower left", ax)
	pp.add_cmap_colorbar(fig, ax, cmap, norm=norm)
	plt.tight_layout()
	plt.savefig(out_file_phylo)

def get_time_interval(time, interval_times):
	try:
		param_interval = np.where(time > interval_times)[0][-1]
	except:
		param_interval = 0

	return param_interval

def effects_to_fitness(estimates_file, features_file, tree_file, interval_times, interval_tree=False):
	est_dict = pd.read_csv(estimates_file, index_col=0)
	
	if (est_times_file := (estimates_file.parent / "time_estimates.csv")).exists():
		est_sites_dict = est_dict.squeeze().to_dict()
		est_times_dict = pd.read_csv(est_times_file, index_col=0).squeeze().to_dict()

	else:
		est_dict = est_dict.set_index("feature").squeeze().to_dict()

		est_sites_dict = {k: v for k, v in est_dict.items() if "Interval" not in k}
		est_times_dict = {k: v for k, v in est_dict.items() if "Interval" in k}

	log_site_effs = np.log(list(est_sites_dict.values()))
	time_effs = list(est_times_dict.values())

	features_df = pd.read_csv(features_file, index_col=0)
	features_df["site_fitness"] = features_df.apply(lambda row: np.exp(np.matmul(row.to_numpy(), log_site_effs)), axis=1)

	tt = pp.loadTree(
		tree_file,
		internal=True,
		abs_time=2023,
	)

	if interval_tree:
		name_re = re.compile(r"_interval\d*$")
		interval_df = pd.DataFrame(index=[n for n in tt.Objects if "interval" in n.traits["name"]], columns=features_df.columns)
		for n in tt.Objects:
			name = n.traits["name"]
			row_idx = re.sub(name_re, "", name)
			interval_df.loc[name, :] = features_df.loc[row_idx, :].to_dict()

		features_df = pd.concat([features_df, interval_df], axis=0)

	tree_times = {n.traits['name']: n.absoluteTime for n in tt.Objects}

	interval_times = np.array(interval_times)
	features_df["time_fitness"] = [time_effs[get_time_interval(tree_times[s], interval_times)] if s in tree_times else float("nan") for s in features_df.index]
	features_df["total_fitness"] = features_df["site_fitness"] * features_df["time_fitness"]

	params = dict(
		tree_file=tree_file,
		interval_times=interval_times,
		)

	return features_df, params

def effects_to_fitness_tsim(analysis_dir, random_name):
	# -----------------------------------------------------
	# Load and sort estimates
	# -----------------------------------------------------
	est_dict = pd.read_csv(analysis_dir / "estimates.csv", index_col=0)
	est_dict = est_dict.set_index("feature").squeeze().to_dict()

	est_sites_dict = {k: v for k, v in est_dict.items() if "Interval" not in k}
	est_times_dict = {k: v for k, v in est_dict.items() if "Interval" in k}

	log_site_effs = np.log(list(est_sites_dict.values()))
	time_effs = list(est_times_dict.values())

	# -----------------------------------------------------
	# Load data object, extract feature types
	# -----------------------------------------------------
	data = pd.read_csv(analysis_dir / "data.csv")
	data.set_index("name", inplace=True)

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
	params = json.loads((analysis_dir / "params.json").read_text())
	data['birth_interval'] = np.take(params['birth_rate_idx'], data['param_interval'])
	data['time_fitness'] = np.take(time_effs, data['birth_interval'])

	# -----------------------------------------------------
	# Get random fitness
	# -----------------------------------------------------
	edge_random = pd.read_csv(analysis_dir / random_name / "edge_random_effects_all.csv", index_col=0)
	data = pd.concat([data, edge_random], axis=1)

	# -----------------------------------------------------
	# Calculate total fitness
	# -----------------------------------------------------
	data["total_model_fitness"] = data["site_fitness"] * data["time_fitness"]
	data["total_fitness"] = data["total_model_fitness"] * data["random_fitness"]
	
	return data, params

def site_effects_to_fitness_tsim(analysis_dir):
	# -----------------------------------------------------
	# Load and sort estimates
	# -----------------------------------------------------
	est_dict = pd.read_csv(analysis_dir / "estimates.csv", index_col=0)
	est_dict = est_dict.loc[est_dict['feature'].str.contains("Interval")==False]

	features = est_dict["feature"].values
	estimates = est_dict["estimate"].values

	# -----------------------------------------------------
	# Load data object, extract feature types
	# -----------------------------------------------------
	data = pd.read_csv(analysis_dir / "data.csv")

	# Remove nodes, since we don't treat them differently
	data = data[data['event'] == 4]

	# -----------------------------------------------------
	# Get site fitness
	# -----------------------------------------------------
	feature_groups = ["AMR", "VIR", "PLASMID", "STRESS", "PRJNA"]
	feature_types = np.array([np.fromiter(ft, int) for ft in data['ft']])

	for feature_group in feature_groups:
		locs = np.array([i for i, c in enumerate(features) if feature_group in c])
		log_site_effs = np.log(np.take(estimates, locs))

		ft = np.take(feature_types, locs, axis=1)

		data[feature_group] = np.exp(np.matmul(ft, log_site_effs))
	
	data.set_index("name", inplace=True)

	params = json.loads((analysis_dir / "params.json").read_text())
	return data[feature_groups], params

def plot_with_random_vs_without(data, out_dir):

	data = data[data['time_step'] <= 20]
	data = data[data['event_time'] >= 1960]

	sns.set_style("whitegrid")
	sns.scatterplot(data=data, x="total_model_fitness", y="total_fitness", hue="event_time", s=3)
	plt.tight_layout()
	plt.savefig(out_dir / "scatter_with-random-vs-without_color-by-event-time.png")
	plt.close("all")

	sns.scatterplot(data=data, x="total_model_fitness", y="total_fitness", hue="time_step", s=3)
	plt.tight_layout()
	plt.savefig(out_dir / "scatter_with-random-vs-without_color-by-time-step.png")
	plt.close("all")

	sns.scatterplot(data=data, x="time_step", y="random_fitness", hue="event_time", s=3)
	plt.tight_layout()
	plt.savefig(out_dir / "random-eff-vs-time_color-by-time-step.png")
	plt.close("all")

	sns.scatterplot(data=data, x="event_time", y="random_fitness", hue="time_step", s=3)
	plt.tight_layout()
	plt.savefig(out_dir / "random-eff-vs-time_color-by-event-time.png")
	plt.close("all")

if __name__ == "__main__":
	import numpy as np



	analysis_dir = dir / "analysis" / "3-interval_constrained-sampling"

	data, params = effects_to_fitness_tsim(analysis_dir, random_name = "Est-Random_Fixed-BetaSite")

	# color_tree_components(params, data[["site_fitness", "time_fitness", "random_fitness", "total_fitness"]], analysis_dir / "phylo_fitness_components.png", zoom=[1960, 2024])
	# color_tree_components(params, data[["total_model_fitness", "total_fitness"]], analysis_dir / "total-model_vs_with-random.png", zoom=[1960, 2024])

	# plot_with_random_vs_without(data, analysis_dir)

	data["num_features"] = [sum(list(map(int, f))) for f in data['ft']]
	color_tree_components(params, data[["num_features"]], analysis_dir / "num_features.png", zoom=None)

	
	# get_plot_clade_membership(dir / "clades.csv", data, params, dir / "clade_membership.png", analysis_dir / "clade_color.png")

	# data, params = site_effects_to_fitness_tsim(analysis_dir)
	# color_tree_components(params, data, analysis_dir / "phylo_site_fitness_components.png")

	# analysis_dir = dir / "phyloTF2" / "TVb0_iterativePE_L2-lamb50"
	# estimates_file = analysis_dir / "estimates.csv"
	# tree_file = dir / 'interval_trees/1862-1985-2008-2013-2018/phylo.nwk'
	# features_file = dir / 'combined_ancestral_states_binary_grouped_diverse_uncorr.csv'
	# interval_times = [1985] + list(np.arange(2008, 2023 + 1 - 5, 5))
	# data, params = effects_to_fitness(estimates_file, features_file, tree_file, interval_times, interval_tree=True)
	# color_tree_components(params, data, analysis_dir / "phylo_fitness_components.png")

