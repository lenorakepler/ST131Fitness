import json
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from transmission_sim.ecoli.analyze import load_data_and_RO_from_file, dropbox_dir
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import yaml
from natsort import natsorted
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

def do_decomp(analysis_name, random_name, decomp_name=None, edge_set=None, total=False, interval_length=5, interval_cutoff=None, ot=None):
	all_data, phylo_obj, RO, params, analysis_out_dir = load_data_and_RO_from_file(analysis_name)
	out_folder = analysis_out_dir / random_name

	# -----------------------------------------------------
	# Make / load dataframe of total (log) fitness for each edge
	# in the tree for each category of fitness
	# -----------------------------------------------------
	# Make dataframe from array of edges,
	all_edges = pd.DataFrame(all_data.getEventArray("edge"))
	all_edges.set_index('name', inplace=True)

	# Add (log) random fitness estimates to edge array
	edge_random = pd.read_csv(out_folder / "edge_random_effects_all.csv", index_col=0)
	all_edges = pd.concat([all_edges, np.log(edge_random)], axis=1)

	# Load feature estimates
	feature_est = pd.read_csv(analysis_out_dir / "estimates.csv", index_col=0)
	feature_est.set_index('feature', inplace=True)
	log_feature_est = np.log(feature_est)

	log_time_est = log_feature_est.loc[[i for i in log_feature_est.index if 'Interval' in i]]
	log_site_est = log_feature_est.loc[[i for i in log_feature_est.index if 'Interval' not in i]]

	# Get presence/absence of each site effect for each edge in array
	feature_types = pd.DataFrame([np.fromiter(ft, int) for ft in all_edges['ft']], columns=log_site_est.index, index=all_edges.index)

	# Get (log) fitness of each site effect (0 if edge doesn't have, log(site fitness) if it does)
	edge_fitness_components = feature_types * log_site_est.values.reshape(1, -1)

	# Get corresponding (log) birth rate for each edge
	time_effs = log_time_est.values
	all_edges['birth_interval'] = np.take(params['birth_rate_idx'], all_edges['param_interval'])
	all_edges['time_fitness'] = np.take(time_effs, all_edges['birth_interval'])

	# edge_fitness_components is a dataframe with a row for each edge, columns for each individual effect
	# e.g. each genetic feature, time-varying beta, random effect, etc.
	edge_fitness_components = pd.concat([edge_fitness_components, all_edges['time_fitness'], all_edges['random_fitness']], axis=1)

	# all_edges.to_csv(out_folder / "all_edges.csv")
	# edge_fitness_components.to_csv(out_folder / "edge_fitness_components.csv")

	# -----------------------------------------------------
	# If we are only doing this on a subset, 
	# subset dataframes
	# -----------------------------------------------------
	if edge_set:
		all_edges = all_edges.loc[edge_set, :]
		edge_fitness_components = edge_fitness_components.loc[edge_set, :]

	if interval_cutoff:
		all_edges = all_edges[all_edges['event_time'] < interval_cutoff]

	# -----------------------------------------------------
	# Create a dataframe with variance calculations 
	# for each time interval
	# -----------------------------------------------------
	# intervals = params['birth_rate_changepoints'] + [all_data.present_time]

	start_time = all_data.root_time
	end_time = interval_cutoff if interval_cutoff else all_data.present_time
	n_intervals = int(np.floor((end_time - start_time) / interval_length))
	if not total:
		intervals = np.linspace(start_time, end_time, n_intervals)
		np.save(out_folder / f"interval_times_intervallength-{interval_length}_cutoff-{interval_cutoff}.npy", intervals, allow_pickle=True)
	else:
		intervals = [0]

	components = edge_fitness_components.columns.to_list()

	# Specify the categories and the features in each of the categories
	feature_categories = {
		'AMR': [c for c in log_site_est.index if 'AMR' in c],
		'Plasmid Replicon': [c for c in log_site_est.index if 'PLASMID' in c],
		'Virulence': [c for c in log_site_est.index if 'VIR' in c],
		'Stress': [c for c in log_site_est.index if 'STRESS' in c],
		# 'Genetic': [c for c in log_site_est.index if 'META' not in c],
		'Background': [c for c in log_site_est.index if 'META' in c],
		'Time': ['time_fitness'],
		'Random': ['random_fitness']
	}

	categories = list(feature_categories.keys())
	category_avgs = [c + "_MULT_FITNESS" for c in categories]
	category_combinations = list(combinations(categories, 2))
	category_combination_strings = [f"{c[0]}_{c[1]}_COV" for c in category_combinations]
	summary = ['TOTAL', 'TOTAL_DIRECT', 'TOTAL_SUMMED', 'LINEAR_AVG_FIT', 'LINEAR_VAR_FIT']

	var_columns = ['n'] + components + categories + category_avgs + category_combination_strings + summary
	var_df = pd.DataFrame(index=intervals, columns=var_columns)

	# At each time interval...

	if total:
		if ot:
			if ot == 'samples':
				interval_df = edge_fitness_components[(edge_fitness_components.index.str.contains("SAMN") == True) & (edge_fitness_components.index.str.contains("interval") == False)]
			elif ot == 'nodes':
				interval_df = edge_fitness_components[edge_fitness_components.index.str.contains("interval") == False]

		else:
			all_intervals = []
			for time in np.linspace(start_time, end_time, n_intervals):
				# Get all edges that were alive at this time
				alive = all_edges[(all_edges['birth_time'] <= time) & (all_edges['event_time'] >= time)].index
				interval_df = edge_fitness_components.loc[alive, :]
				all_intervals.append(interval_df)

			interval_df = pd.concat(all_intervals, axis=0)

	for time in intervals:
		if total:
			# Get ALL edges
			# interval_df = edge_fitness_components
			interval_df = interval_df
		else:
			# Get all edges that were alive at this time
			alive = all_edges[(all_edges['birth_time'] <= time) & (all_edges['event_time'] >= time)].index
			interval_df = edge_fitness_components.loc[alive, :]

		var_df.loc[time, 'n'] = len(interval_df)

		# Create covariance matrix from these edges' fitness components
		feature_cov = interval_df.cov()

		# Grab variance of each individual feature (on diagonal of covariance)
		var_df.loc[time, components] = np.diag(feature_cov.values)

		# Sum covariance of each feature group
		for category, category_features in feature_categories.items():
			var_df.loc[time, category] = feature_cov.loc[category_features, category_features].sum().sum()

		# Sum total variance
		var_df.loc[time, 'TOTAL'] = feature_cov.sum().sum()

		# Calculate total log fitness of edges
		linear_fit = interval_df.sum(axis=1)
		var_df.loc[time, 'LINEAR_AVG_FIT'] = linear_fit.mean()
		var_df.loc[time, 'LINEAR_VAR_FIT'] = np.var(linear_fit) # Pandas var gives different value, b/c pd assumes ddof=0, numpy is ddof=1

		# Calculate the sum of fitness effects for each category, for each edge
		edge_fitness = {category: interval_df[category_features].sum(axis=1) for category, category_features in feature_categories.items()}

		# Get average fitness contribution of each category
		for category, category_features in feature_categories.items():
			var_df.loc[time, category + "_MULT_FITNESS"] = np.exp(edge_fitness[category].mean())

		# Calculate the sum of the covariances of each category
		for cat1, cat2 in category_combinations:
			var_df.loc[time, f"{cat1}_{cat2}_COV"] = 2 * np.cov(edge_fitness[cat1], edge_fitness[cat2])[0][1]

		# Sum category variance and between-category co-variance
		var_df.loc[time, 'TOTAL_SUMMED'] = var_df.loc[time, categories + category_combination_strings].sum()

	var_df['TOTAL_NO_COV'] = var_df[categories].sum(axis=1)

	for category in categories:
		var_df[f'FRAC_{category.upper()}'] = var_df[category] / var_df['TOTAL_NO_COV']
	
	var_df.index.name = 'Year'

	if decomp_name:
		out_folder = out_folder / decomp_name
		out_folder.mkdir(exist_ok=True, parents=True)

	if total:
		var_df['FRAC_MODEL'] = np.sum([var_df[f'FRAC_{category.upper()}'] for category in categories if category != 'Random'])
		var_df['FRAC_GENETIC'] = np.sum([var_df[f'FRAC_{category.upper()}'] for category in categories if category not in ['Random', 'Time', 'Background']])
		var_df['FRAC_BGTIME'] = var_df[f'FRAC_TIME'] + var_df[f'FRAC_BACKGROUND']
		var_df.T.to_csv(out_folder / f"variance_decomposition_total_intervallength-{interval_length}_cutoff-{interval_cutoff}_type-{ot}.csv")

		var_df[[c for c in var_df if 'FRAC' in c]].T.to_csv(out_folder / f"total_decomp_fractions_intervallength-{interval_length}_cutoff-{interval_cutoff}_type-{ot}.csv")
	else:
		var_df.to_csv(out_folder / f"variance_decomposition_intervallength-{interval_length}_cutoff-{interval_cutoff}.csv")

def cat_display(cat):
	cds = {
		'AMR': 'AMR',
		'VIR': 'Virulence',
		'STRESS': 'Stress',
		'PLASMID': 'Plasmid Replicon',
		'nan': 'Background'
	}
	if cat in cds:
		return cds[str(cat)]
	else:
		return cat

def do_plots(analysis_name, random_name, out_folder, decomp_name="", interval_length=5, interval_cutoff=None):
	in_folder = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset" / "final" / "analysis" / analysis_name / random_name

	if decomp_name:
		in_folder = in_folder / decomp_name

	out_folder.mkdir(exist_ok=True, parents=True)

	df = pd.read_csv(in_folder / f"variance_decomposition_intervallength-{interval_length}_cutoff-{interval_cutoff}.csv", index_col=0)
	df = df[df.index > 1960]

	# -----------------------------------------------------
	# Plot mean fitness of each category over time, 
	# both as stacked plots and line plots.
	# For stacked plots, use log values so when stacked it
	# sums to total (Must transform y axis labels back)
	# -----------------------------------------------------
	c_palette = sns.color_palette("Set2")[0:-3] +  [sns.color_palette('Blues')[0], sns.color_palette("Set3")[10]]

	sns.set_style('whitegrid')
	fig, axs = plt.subplots(1, 2, figsize=(12, 5))
	
	mult_fit = df[[c for c in df.columns if 'MULT_FITNESS' in c]]
	mult_fit.rename(columns={c: c.split("_")[0] for c in mult_fit.columns}, inplace=True)
	mult_fit.drop(columns=['Time', 'Background', 'Random'], inplace=True)

	category_order = [c for c in mult_fit.columns]
	sns.lineplot(mult_fit, palette=c_palette, hue_order=category_order, ax=axs[1])
	mult_total = mult_fit.product(axis=1).to_list()
	axs[1].plot(mult_fit.index, mult_total, color="black", label="Total")
	axs[1].scatter(mult_fit.index, mult_total, color="black", s=4)

	# Separate positive and negative values to accurately plot
	# https://stackoverflow.com/questions/65859200/how-to-display-negative-values-in-matplotlibs-stackplot
	add_fit = np.log(mult_fit)
	pos_fit = add_fit[add_fit >= 0].fillna(0)
	neg_fit = add_fit[add_fit < 0].fillna(0)

	total = add_fit.sum(axis=1).to_list()

	for_csv = add_fit.copy()
	for_csv['total'] = total
	for_csv.to_csv(out_folder / f"Time_Mean_Fitness_Decomp_intervallength-{interval_length}_cutoff-{interval_cutoff}.png")

	axs[0].stackplot(pos_fit.index, pos_fit.values.T, labels=[cat_display(c) for c in pos_fit.columns], colors=c_palette)
	axs[0].stackplot(neg_fit.index, neg_fit.values.T, colors=c_palette)
	axs[0].plot(add_fit.index, total, color="black", label="Total")
	axs[0].scatter(add_fit.index, total, color="black", s=4)
	axs[0].axhline(y=0, linewidth=.75, color="black", linestyle='--')

	# axs[0].stackplot(add_fit.index, add_fit.values.T, labels=add_fit.columns, colors=sns.color_palette("Set2"))
	log_vals = axs[0].get_yticks().tolist()
	axs[0].set_yticklabels([np.round(np.exp(l), 2) for l in log_vals])
	axs[0].set_xlabel("Year")
	axs[0].set_ylabel("Fitness Contribution")
	axs[0].legend(loc='upper left')
	
	if decomp_name:
		fig.suptitle(f"Clade {decomp_name}")
		plt.tight_layout()
	else:
		plt.tight_layout()

	plt.savefig(out_folder / f"Time_Mean_Fitness_Decomp_Stacked-Line_intervallength-{interval_length}_cutoff-{interval_cutoff}.png", dpi=300)
	plt.close("all")

	# -----------------------------------------------------
	# Plot variance decomposition, stacked plots
	# Absolute + proportional side-by-side
	# -----------------------------------------------------
	fig, axs = plt.subplots(1, 2, figsize=(12, 5))
	
	var_decomp_abs = df[[c for c in df.columns if '_' not in c and c not in ['TOTAL', 'Time', 'n']]]
	axs[0].stackplot(var_decomp_abs.index, var_decomp_abs.values.T, labels=[cat_display(c) for c in var_decomp_abs.columns], colors=c_palette)
	axs[0].legend(loc='upper left')
	axs[0].set_xlabel("Year")
	axs[0].set_ylabel("Total Fitness Variance")

	var_decomp = df[[c for c in df.columns if 'FRAC' in c and 'TIME' not in c]]
	var_decomp.columns = [c.split("_")[1].capitalize() for c in var_decomp.columns]
	axs[1].stackplot(var_decomp.index, var_decomp.values.T, labels=[cat_display(c) for c in var_decomp.columns], colors=c_palette)
	axs[1].set_xlabel("Year")
	axs[1].set_ylabel("Proportion of Variance Explained")
	
	if decomp_name:
		fig.suptitle(decomp_name)
		plt.tight_layout()
	else:
		plt.tight_layout()
		
	plt.tight_layout()
	plt.savefig(out_folder / f"Time_Variance_Decomp_Stacked_Abs-Prop_intervallength-{interval_length}_cutoff-{interval_cutoff}.png", dpi=300)
	plt.close("all")

	# -----------------------------------------------------
	# Plot variance decomposition proportions with
	# Mean category fitness lines overlaid
	# -----------------------------------------------------
	# line_colors = [sns.dark_palette(c, reverse=True, as_cmap=False)[1] for c in sns.color_palette("Set2")]
	# bg_colors = sns.color_palette("Set2")

	# fig, ax1 = plt.subplots()
	# ax2 = ax1.twinx()

	# ax1.stackplot(var_decomp.index, var_decomp.values.T, labels=var_decomp.columns, colors=bg_colors)
	# sns.lineplot(data=mult_fit, palette=line_colors, ax=ax2)

	# ax1.set_ylabel("Proportion of Variance Explained by Feature Category")
	# ax2.set_ylabel("Combined Fitness Effect of Feature Category")
	# plt.tight_layout()
	# plt.savefig(out_folder / "Frac_Variance_Stacked_With_Lines.png", dpi=300)
	# plt.close("all")

def do_stack_plot(component_df, axs, pos_features, neg_features, pos_colors, neg_colors):
	dnames = yaml.load((dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset" / "final" / "features" / "group_short_name_to_display_manual.yml").read_text())

	# Separate positive and negative values to accurately plot
	# https://stackoverflow.com/questions/65859200/how-to-display-negative-values-in-matplotlibs-stackplot
	pos_fit = component_df[[c for c in pos_features if c in component_df.columns.to_list()]].fillna(0)
	neg_fit = component_df[[c for c in neg_features if c in component_df.columns.to_list()]].fillna(0)

	total = component_df.sum(axis=1).to_list()

	pos_labels = [dnames[n.split("_")[0]]['display_name'] if n.split("_")[0] in dnames else n.split("_")[0] for n in pos_fit.columns]
	neg_labels = [dnames[n.split("_")[0]]['display_name'] if n.split("_")[0] in dnames else n.split("_")[0] for n in neg_fit.columns]

	if len(pos_fit.columns) > 0:
		axs.stackplot(pos_fit.index, pos_fit.values.T, labels=pos_labels, colors=pos_colors)

	if len(neg_fit.columns) > 0:
		axs.stackplot(neg_fit.index, neg_fit.values.T, labels=neg_labels, colors=neg_colors)

	axs.plot(component_df.index, total, color="black", label="Total")
	axs.scatter(component_df.index, total, color="black", s=4)
	axs.axhline(y=0, linewidth=.75, color="black", linestyle='--')

	log_vals = axs.get_yticks().tolist()
	axs.set_yticklabels([np.round(np.exp(l), 2) for l in log_vals])
	
def components_by_clade(analysis_name, random_name, categories, out_folder, interval_length=5, interval_cutoff=None, kind=None, json_out=False, return_fig=False):
	in_folder = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset" / "final" / "analysis" / analysis_name / random_name

	out_folder.mkdir(exist_ok=True, parents=True)

	all_edges = pd.read_csv(in_folder / "all_edges.csv", index_col=0)
	# edge_fitness_components = pd.read_csv(in_folder / "edge_fitness_components.csv", index_col=0)
	intervals = np.load(in_folder / f"interval_times_intervallength-{interval_length}_cutoff-{interval_cutoff}.npy", allow_pickle=True)

	edge_fitness_components = edge_fitness_components.loc[:, edge_fitness_components.sum(axis=0) != 0]

	# Get clade type of each edge
	clades = pd.read_csv(in_folder.parent.parent.parent / "ST131_Typer" / "combined_ancestral_states.csv", index_col=0).squeeze()
	
	if kind == "alt":
		clades = clades.replace('C1-M27', 'C1')
		wanted_clades = ['A', 'B1', 'C1', 'C2']

	elif kind == "all":
		wanted_clades = natsorted(list(clades.unique()))

	elif kind == "combined":
		clades = clades.apply(lambda x: "All")
		wanted_clades = ["All"]

	elif kind == "combined-alt":
		wanted_clades = ["All",'A', 'B1', 'C1', 'C2']

		clades = clades.replace('C1-M27', 'C1')
		all_edges['clade'] = [clades[i.split("_")[0]] for i in all_edges.index]
		
		combined_edges = all_edges.rename(mapper=lambda i: i + "_all")
		combined_edges['clade'] = "All"
		all_edges = pd.concat([all_edges, combined_edges], axis=0)

		edge_fitness_components = pd.concat([edge_fitness_components, edge_fitness_components.rename(mapper=lambda i: i + "_all")], axis=0)
		
	else:
		wanted_clades = ['A', 'B1', 'C1', 'C1-M27', 'C2']

	if kind != "combined-alt":
		all_edges['clade'] = [clades[i.split("_")[0]] for i in all_edges.index]

	wanted_edges = all_edges[all_edges['clade'].isin(wanted_clades)].index

	component_list = []
	for time in intervals:
		alive = all_edges[(all_edges['birth_time'] <= time) & (all_edges['event_time'] >= time)]

		for clade in wanted_clades:
			clade_alive = alive[alive['clade'] == clade].index
			interval_df = edge_fitness_components.loc[clade_alive, :]
			component_list.append([time, clade] + interval_df.mean().to_list())

	component_df = pd.DataFrame(component_list, columns=['interval', 'clade'] + edge_fitness_components.columns.to_list())
	
	# Display only those intervals after 1965
	component_df = component_df.loc[component_df['interval'] > 1965]
	component_df.set_index(['clade', 'interval'], inplace=True)

	# Display only features with mean fitness of greater than 1.01 at some time
	component_df = component_df.loc[:, component_df.abs().max(axis=0) > np.log(1.01)]

	n_rows = len(categories)
	n_cols = len(wanted_clades)

	pos_fit = component_df.loc[:, component_df.sum(axis=0) > 0].columns
	neg_fit = component_df.loc[:, component_df.sum(axis=0) < 0].columns

	pos_colors = sns.color_palette("Spectral", n_colors=max([len([f for f in pos_fit if category in f]) for category in categories]))
	neg_colors = sns.color_palette("blend:#7AB,#EDA", n_colors=max([len([f for f in neg_fit if category in f]) for category in categories]))


	# ------- ACTUAL PLOTTING -------
	sns.set_style('whitegrid')
	fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5.75 * n_rows), sharex=True, sharey=True, )

	if n_cols == 1:
		axs = axs.reshape(-1, 1)
	
	elif n_rows == 1:
		axs = axs.reshape(1, -1)

	for row, cat in enumerate(categories):
		for col, clade in enumerate(wanted_clades):
			cdf = component_df.xs(clade, level="clade")
			do_stack_plot(cdf[[c for c in component_df.columns if cat in c]], axs[row][col], pos_fit, neg_fit, pos_colors, neg_colors)
			
	# Set column labels
	[ax.set_title(f"Clade {clade}", fontsize=20, pad=18) for clade, ax in zip(wanted_clades, axs[0])]

	# Put legend to right of plots
	[row[-1].legend(title=cat_display(category), loc='upper left', bbox_to_anchor = (1.02, 1.02), fontsize=18, title_fontsize=18, frameon=False) for category, row in zip(categories, axs)]

	axs[-1][2].set_xlabel("Year", fontsize=20, labelpad=18)
	[row[0].set_ylabel(f"{cat_display(category)} Fitness Contribution", fontsize=18) for category, row in zip(categories, axs)]
	[ax.xaxis.set_tick_params(labelsize=15) for ax in axs[-1]]

	# Default for shared y axis is to only have ticks on first column, shared x on last row
	# for row in axs:
	# 	[ax.yaxis.set_tick_params(labelleft=True) for ax in row[1:]]
	# 	[ax.xaxis.set_tick_params(labelbottom=True) for ax in row]

	plt.tight_layout()

	kind_str = f"_{kind}" if kind else ""
	fname = f"Fitness-Components_ByClade_{'-'.join([c for c in categories])}-{interval_length}_cutoff-{interval_cutoff}{kind_str}.png"

	if json_out:
		json_dict = {}
		for category in categories:
			json_dict[category] = {
				"positive": [f for f in pos_fit if category in f],
				"negative": [f for f in neg_fit if category in f],
			}

		(out_folder / fname.replace("png", "json")).write_text(json.dumps(json_dict, indent=4))

	if not return_fig:
		plt.savefig(out_folder / fname, dpi=300)
		plt.close("all")

	else:
		return fig, axs, fname

def failed_gridspec_lines_layout():
	sns.set_style('whitegrid')

	fig, axs = plt.subplots(n_rows + 1, n_cols, sharex=True, squeeze=True, gridspec_kw={'height_ratios': [.01] + [1] * n_rows}, figsize=(5 * n_cols, 5.75 * n_rows))

	gs = axs[0, 0].get_gridspec()
	for ax in axs[0, 1:]:
		ax.remove()

	label1 = fig.add_subplot(gs[0, 0])
	label2 = fig.add_subplot(gs[0, 1:])

	label1.set_xlabel("")
	label2.set_xlabel("")

	for ax in [label1, label2]:
		ax.tick_params(size=0)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_facecolor("none")
		for pos in ["right", "top", "left"]:
			ax.spines[pos].set_visible(False)
		ax.spines["bottom"].set_linewidth(3)
		ax.spines["bottom"].set_color("crimson")

	for row, cat in enumerate(categories):
		for col, clade in enumerate(wanted_clades):
			cdf = component_df.xs(clade, level="clade")
			do_stack_plot(cdf[[c for c in component_df.columns if cat in c]], axs[row + 1, col], pos_fit, neg_fit, pos_colors, neg_colors)
	
	# Set column labels
	[ax.set_title(f"Clade {clade}", fontsize=20, pad=18) for clade, ax in zip(wanted_clades, axs[1])]

	# breakpoint()

	# Put legend to right of plots
	[row[-1].legend(title=cat_display(category), loc='upper left', bbox_to_anchor = (1.02, 1.02), fontsize=18, title_fontsize=18, frameon=False) for category, row in zip(categories, axs[1:])]

	# axs[-1][2].set_xlabel("Year", fontsize=20, labelpad=18)
	# [row[0].set_ylabel("Fitness Contribution", fontsize=18) for row in axs]
	# [ax.xaxis.set_tick_params(labelsize=15) for ax in axs[-1]]

def analyze_by_clade(analysis_name, random_name, out_folder, interval_length=5, interval_cutoff=None):
	all_data, phylo_obj, RO, params, analysis_out_dir = load_data_and_RO_from_file(analysis_name)
	
	# Make dataframe from array of edges,
	all_edges = pd.DataFrame(all_data.getEventArray("edge"))
	all_edges.set_index('name', inplace=True)

	# Get clade type of each edge
	clades = pd.read_csv(analysis_out_dir.parent.parent / "ST131_Typer" / "combined_ancestral_states.csv", index_col=0).squeeze()
	all_edges['clade'] = [clades[i.split("_")[0]] for i in all_edges.index]

	for clade, clade_df in all_edges.groupby('clade'):
		do_decomp(analysis_name, random_name, decomp_name=clade, edge_set=clade_df.index.to_list(), interval_length=interval_length, interval_cutoff=interval_cutoff)
		do_decomp(analysis_name, random_name, decomp_name=clade, edge_set=clade_df.index.to_list(), total=True, interval_length=interval_length, interval_cutoff=interval_cutoff)
		do_plots(analysis_name, random_name, out_folder, decomp_name=clade, interval_length=interval_length, interval_cutoff=interval_cutoff)


def QRDR(analysis_name, random_name, out_folder):
	in_folder = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset" / "final" / "analysis" / analysis_name / random_name
	out_folder.mkdir(exist_ok=True, parents=True)

	all_edges = pd.read_csv(in_folder / "all_edges.csv", index_col=0)
	edge_fitness_components = pd.read_csv(in_folder / "edge_fitness_components.csv", index_col=0)
	edge_fitness_components = edge_fitness_components.loc[:, edge_fitness_components.sum(axis=0) != 0]

	# Get clade type of each edge
	clades = pd.read_csv(in_folder.parent.parent.parent / "ST131_Typer" / "combined_ancestral_states.csv", index_col=0).squeeze()
	all_edges['clade'] = [clades[i.split("_")[0]] for i in all_edges.index]
	
	# just look at single interval
	ef = edge_fitness_components.loc[[e for e in edge_fitness_components.index if '_interval' not in e]]
	qr = ef[[c for c in ef.columns if ('gyr' in c or 'par' in c)]]

	qrdict = {
		'both': len(qr[(qr['gyr_AMR'] > 0) & (qr['par_AMR'] < 0)]),
		'gyr_only': len(qr[(qr['gyr_AMR'] > 0) & (qr['par_AMR'] == 0)]),
		'par_only': len(qr[(qr['gyr_AMR'] == 0) & (qr['par_AMR'] < 0)]),
		'neither': len(qr[(qr['gyr_AMR'] == 0) & (qr['par_AMR'] == 0)]),
		}

	qr = qr.loc[[e for e in qr.index if 'SAMN' in e]]
	qrdict2 = {
		'both': len(qr[(qr['gyr_AMR'] > 0) & (qr['par_AMR'] < 0)]),
		'gyr_only': len(qr[(qr['gyr_AMR'] > 0) & (qr['par_AMR'] == 0)]),
		'par_only': len(qr[(qr['gyr_AMR'] == 0) & (qr['par_AMR'] < 0)]),
		'neither': len(qr[(qr['gyr_AMR'] == 0) & (qr['par_AMR'] == 0)]),
		}

	po = qr[(qr['gyr_AMR'] == 0) & (qr['par_AMR'] < 0)]

	clades.loc[po.index]
	breakpoint()

if __name__ == "__main__":
	from transmission_sim.analysis.PhyloRegressionTree import PhyloBoost
	import matplotlib as mpl
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.model_selection import TimeSeriesSplit
	import transmission_sim.utils.plot_phylo_standalone as pp

	analysis_name = "3-interval_constrained-sampling"
	random_name = "Est-Random_Fixed-BetaSite"

	# do_decomp(analysis_name, random_name, interval_length=interval_length, interval_cutoff=interval_cutoff)
	# do_decomp(analysis_name, random_name, total=True, interval_length=1, interval_cutoff=2021)
	do_plots(analysis_name, random_name, out_folder=dropbox_dir / "NCSU/Lab/Writing/st131_git" / "all_figures", interval_length=1, interval_cutoff=2021)
	# analyze_by_clade(analysis_name, random_name, interval_length=interval_length, interval_cutoff=interval_cutoff)

	# for interval_cutoff in [None, 2021]:
	# 	for interval_length in [5]:
	# 		do_decomp(analysis_name, random_name, total=True, interval_length=interval_length, interval_cutoff=interval_cutoff, ot='samples')
	# 		do_decomp(analysis_name, random_name, total=True, interval_length=interval_length, interval_cutoff=interval_cutoff, ot='nodes')

	# # 		for kind in ["combined-alt", "alt"]:
	# 			for catgories in [	
	# 				['AMR', 'VIR'], 
	# 				['AMR', 'PLASMID', 'VIR', 'STRESS'], 
	# 				['AMR', 'PLASMID', 'VIR', 'STRESS', 'fitness']
	# 			]:
	# 				components_by_clade(
	# 					analysis_name, 
	# 					random_name, 
	# 					out_folder=dropbox_dir / "NCSU/Lab/Writing/st131_git" / "all_figures",
	# 					categories=catgories,
	# 					interval_length=interval_length, 
	# 					interval_cutoff=interval_cutoff, 
	# 					kind=kind
	# 				)

	# # # ==================
	# # # For pub
	# # # ==================
	# interval_length = 1
	# interval_cutoff = 2021

	# out_folder = dropbox_dir / "NCSU/Lab/Writing/st131_git" / "figures"
	# do_plots(analysis_name, random_name, out_folder, interval_length=interval_length, interval_cutoff=interval_cutoff)

	# kind = "combined-alt"
	# categories = ['AMR', 'VIR']
	# fig, axs, fname = components_by_clade(
	# 				analysis_name, 
	# 				random_name, 
	# 				categories=categories,
	# 				out_folder=out_folder,
	# 				interval_length=interval_length, 
	# 				interval_cutoff=interval_cutoff, 
	# 				kind=kind,
	# 				return_fig=True,
	# 			)

	# plt.subplots_adjust(hspace=0.05)

	# for row in [0]:
	# 	box = axs[row][0].get_position()
	# 	box.x0 = box.x0 - 0.05
	# 	box.x1 = box.x1 - 0.05
	# 	axs[row][0].set_position(box)

	# plt.tight_layout()

	# w, h = fig.get_size_inches()
	# fig.set_size_inches(w + .05, h)

	# for row in range(axs.shape[0]):
	# 	box = axs[row][0].get_position()
	# 	box.x0 = box.x0 - 0.01
	# 	box.x1 = box.x1 - 0.01
	# 	axs[row][0].set_position(box)

	# axs[0][0].set_title("All Clades", fontsize=20, pad=18)

	# sp = axs[0][0].get_position()

	# y = sp.y1 + 0.01
	# xd = 0.0

	# X, Y = np.array([[sp.x0 - xd, sp.x1 + xd], [y, y]])
	# line = Line2D(X, Y, lw=1.5, color='black', alpha=1)
	# fig.add_artist(line)

	# X, Y = np.array([[axs[0][1].get_position().x0 - xd, axs[0][-1].get_position().x1 + xd], [y, y]])
	# line = Line2D(X, Y, lw=1.5, color='black', alpha=1)
	# fig.add_artist(line)

	# plt.savefig(out_folder / fname, dpi=300)

	# QRDR(analysis_name, random_name, dropbox_dir / "NCSU/Lab/Writing/st131_git" / "figures")
