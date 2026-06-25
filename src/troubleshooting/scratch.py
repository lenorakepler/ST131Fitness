import pandas as pd
import numpy as np
from pathlib import Path
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
from model_fit.arrayer import PhyloDataFile
from multiprocessing import freeze_support, set_start_method, Pool
from collections import ChainMap
import json
import copy
import matplotlib.pyplot as plt
from model_fit.utils import make_intervals
from model_fit.results_obj import ResultsObj
from model_fit.random_effects import prep_data_for_hyperparam_search, prep_data_for_fitting
import analysis.plot_phylo_standalone as pp
import seaborn as sns
from yte import process_yaml

def lj(file):
	return json.loads(Path(file).read_text())

def set_shade(color, train):
	shades = sns.light_palette(color, n_colors=12, as_cmap=False)
	if train:
		return shades[-1]
	else:
		return shades[2]

def plot_tree(tt, df, out_file, title=""):
	fig, axs = plt.subplots(1, n_folds, figsize=(12, 25))

	colors, c_func = pp.categoricalFunc(df.set_index('name')['branch_name'].to_dict(), 'name', legend=True, color_list=["blue", "red", "orange", "yellow", "purple", "lime", "pink", "green", "bisque", "aqua"])

	ax = pp.plotTraitAx(
		ax,
		tt,
		edge_c_func=c_func,
		node_c_func=c_func,
		s_func=lambda x: 50,
		tip_names=False,
		zoom=False,
		title=title,
		width=10,
	)
	ax = pp.plotInternalNodes(tt, ax, lambda k: "black", lambda k: "black", size_func=lambda k: 20)
	ax = pp.plotBirthNodes(tt, ax, c_func, c_func, c_func_birth_size=lambda k: 50)
	ax = pp.set_vertical(ax, line_style="solid", color="lightgray", line_width=3)

	pp.add_legend(colors, ax, lloc="lower left")
	plt.tight_layout()
	plt.savefig(out_file, dpi=300)
	plt.close("all")

def phylo_plot_train_test(tt, folds, df, out_file, title=""):
	n_folds = len(folds)
	fig, axs = plt.subplots(1, n_folds, figsize=(12 * n_folds, 25))
	axs = axs.ravel()

	all_types = []
	for fold_dict in folds.values():
		all_types += [v["type_int"] for v in fold_dict["train"].values()]

	trait_value_order = sorted(list(set(all_types)))
	color_list = [
		"black", "blue", "red", "orange", "yellow", "purple", "lime", 
		"pink", "green", "bisque", "aqua", "dimgray", 
		"mediumturquoise", "blueviolet", "deeppink", "peru", 
		"navy", "olive", "yellowgreen", "indigo", "orangered"
		]
	trait_colors = {t: color_list[i] for i, t in enumerate(trait_value_order)}

	for fold_num, fold_dict in folds.items():
		ax = axs[int(fold_num)]

		test_start = fold_dict['params']['test']['start_time']
		test_end = fold_dict['params']['test']['end_time']

		segment_colors = {}
		for i, subset in enumerate(["train", "test"]):
			fold_df = df.iloc[[int(k) for k in fold_dict[subset].keys()]].copy()
			type_key = 'type_int' if subset == "train" else 'parent_type_int'
			fold_df["type_int"] = [v[type_key] for v in fold_dict[subset].values()]
			segment_colors.update({n: set_shade(trait_colors[t], subset=="train") for n, t in fold_df["type_int"].items()})
		
		c_func = lambda k: segment_colors[k.traits["name"]] if k.traits["name"] in segment_colors else "white"

		ax = pp.plotTraitAx(
			ax,
			tt,
			edge_c_func=c_func,
			node_c_func=c_func,
			s_func=lambda x: 50,
			tip_names=False,
			zoom=False,
			title=title,
			width=10,
		)
		ax = pp.plotInternalNodes(tt, ax, lambda k: "black", lambda k: "black", size_func=lambda k: 20)
		ax = pp.plotBirthNodes(tt, ax, c_func, c_func, c_func_birth_size=lambda k: 50)
		ax = pp.set_vertical(ax, line_style="solid", color="lightgray", line_width=3)

		ax.axvline(x=test_start, linestyle="--", color="red")
		ax.axvline(x=test_end, linestyle="--", color="blue")

	pp.add_legend(trait_colors, ax, lloc="lower left")
	plt.tight_layout()
	plt.savefig(out_file, dpi=300)
	plt.close("all")

def scratch0():
	res_dir = Path("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/analysis/3-interval_constrained-sampling")
	
	for sub_dir in [res_dir / "random_only", res_dir / "Est-Random_Fixed-BetaSite"]:
		ro_res = json.loads((sub_dir / "results.json").read_text())

		sigma = [float(k) for k in ro_res.keys()]
		test_loss = [v['test_mean'] for v in ro_res.values()]
		train_loss = [np.mean(v["train"]) for v in ro_res.values()]

		plt.plot(sigma, test_loss, label="test loss")
		plt.plot(sigma, train_loss, label="train loss")
		plt.savefig(sub_dir / "sigma_search.png", dpi=300)
		plt.close("all")

		df = pd.DataFrame({"sigma": sigma, "loss": test_loss})
		print(df.sort_values(by="loss"))

	# ================================================================================================

	sub_dir = res_dir / "Est-Random_Fixed-BetaSite"
	more_info = lj(sub_dir / "full_random_info.json")
	fp = lj(sub_dir / "fold_params.json")
	data = pd.read_csv(res_dir / "data.csv", index_col=0)
	data = data.drop(columns="ft")
	data = data[['idx', 'name', 'birth_time', 'event_time', 'back_time', 'event', 'parent_idx']]
	data["all_type_int"] = more_info["all"]["type_int"]
	data["all_parent_type_int"] = more_info["all"]["parent_type_int"]
	data["all_parent_time_delta"] = more_info["all"]["parent_time_delta"]
	data = data[data["event"] == 4]
	data = data.set_index("name")

	# data2 = data.set_index("idx")

	new = lj("data_new/analysis/three_sampling_intervals/brownian_search_setup.json")["folds"]

	df_folds = []
	for i in [0, 1, 2]:
		tt = []
		for subset in ["train", "test"]:
			new_df = pd.DataFrame(new[str(i)][subset]).T
			new_df = new_df.drop(columns=["parent_name", "parent_time"])

			fold = fp["folds"][str(i)][subset]
			fold_df = data.iloc[fold["data_idx"], :].copy()
			fold_df["type_int"] = fold["type_int"]
			fold_df["parent_time_delta"] = fold["parent_time_delta"]
			fold_df = fold_df.rename(columns={"parent_time_delta": "delta"})

			if subset == "train":
				fold_df["parent_type_int"] = fold["parent_type_int"]

			tt.append(fold_df)
			print(f"======== fold {i}, {subset} ========")
			print(f"**************  OLD   **************")
			print(fold_df[[c for c in new_df.columns if c in fold_df.columns]].sort_values(by="idx"))
			print(f"**************  NEW   **************")
			print(new_df.sort_values(by="idx"))
			print("")
		df_folds.append(tt)

def scratch1(RO, analysis_dir):
	if False:
		interval_times, interval_tree = make_intervals(
			"data_new/test_interval_tree",
			"data_new/test.nwk",
			2023,
			[2003, 2012],
			[2009, 2012, 2015, 2018, 2022],
			)

	analysis_dir = "data_new/test_random_analysis"
	RO = ResultsObj(analysis_dir)

	if RO.success["data"] == False:
		RO.set_data(
			tree_file="data_new/test_interval_tree/phylo.nwk",
			interval_times_file="data_new/test_interval_tree/interval_times.txt",
			last_sample_date=2023,
			)

	if RO.success["index"] == False:
		RO.set_folds(test_size=0.2, n_splits=2, stratify=None, random_state=8)

	df = pd.DataFrame(RO.data.array)
	df['branch_name'] = df['name'].apply(lambda x: x.split("_interval")[0])
	df = df.set_index("name")

	tt = pp.loadTree(
		"data_new/test_interval_tree/phylo.nwk",
		internal=True,
		abs_time=2023
	)

def scratch2(analysis_dir):
	prep_data_for_hyperparam_search(analysis_dir, n_folds=3, test_proportion=(1/2), folds_start=2006, plot=False, alt=False)
	prep_data_for_fitting(analysis_dir, plot=True, alt=False)

	prep_data_for_hyperparam_search(analysis_dir, n_folds=3, test_proportion=(1/2), folds_start=2006, plot=False, alt=True)
	prep_data_for_fitting(analysis_dir, plot=True, alt=True)

def scratch3(RO, analysis_dir):
	df = pd.DataFrame(RO.data.array)
	df['branch_name'] = df['name'].apply(lambda x: x.split("_interval")[0])
	df = df.set_index("name")
	
	folds = lj(f"{analysis_dir}/brownian_search_setup.json")["folds"]
	phylo_plot_train_test(tt, folds, df, RO.folder / "brownian_search_setup_types.png")

	folds = lj(f"{analysis_dir}/brownian_search_setup_alt.json")["folds"]
	phylo_plot_train_test(tt, folds, df, RO.folder / "brownian_search_setup_types_alt.png")

	setup = lj(f"{analysis_dir}/brownian_fit_setup.json")
	folds = {
		'0': {**setup['0'], 'params': {'test': {'start_time': RO.data.root_time, 'end_time': RO.data.present_time}}}, 
		'1': {**setup['1'], 'params': {'test': {'start_time': RO.data.root_time, 'end_time': RO.data.present_time}}},
		}
	phylo_plot_train_test(tt, folds, df, RO.folder / "brownian_fit_setup_types.png")

	setup = lj(f"{analysis_dir}/brownian_fit_setup_alt.json")
	folds = {
		'0': {**setup['0'], 'params': {'test': {'start_time': RO.data.root_time, 'end_time': RO.data.present_time}}}, 
		'1': {**setup['1'], 'params': {'test': {'start_time': RO.data.root_time, 'end_time': RO.data.present_time}}},
		}
	phylo_plot_train_test(tt, folds, df, RO.folder / "brownian_fit_setup_types_alt.png")

def scratch4():
	prep_data_for_fitting("data_new/analysis/three_sampling_intervals", plot=True, alt=False)

def scratch5(RO):
	for result_key in ["full_model_birth_features+brownian_motion+sampling_background_TV+sampling_features"]:
		RO.wrangle_stragglers(result_key)
		RO.summarize_search(result_key)
		RO.plot_hyperparams(RO.folder / result_key)

	RO.summarize_searches()
	RO.plot_hyperparams(RO.folder)

def scratch5(RO):
	RO.rename_move_delete(result_key="sigma-opt_default_brownian_motion", rename="sigma-opt_whole-branch_brownian_motion")

def scratch6():
	df = pd.read_csv("data_new/analysis/three_sampling_intervals/hyperparam_search.csv")
	df = df[df['fold_test_losses'].str.count(",") == 2]
	df['fold_test_losses'] = df['fold_test_losses'].apply(lambda k: eval(k))

	temp_dir = Path("data_new/analysis/three_sampling_intervals/2-fold-test/")
	temp_dir.mkdir(exist_ok=True, parents=True)
	df['mean_test_loss'] = df['fold_test_losses'].apply(lambda k: np.mean(k[0:1]))
	df.to_csv(temp_dir / "hyperparam_search.csv")
	RO.plot_hyperparams(temp_dir)

	temp_dir = Path("data_new/analysis/three_sampling_intervals/last-fold-test/")
	temp_dir.mkdir(exist_ok=True, parents=True)
	df['mean_test_loss'] = df['fold_test_losses'].apply(lambda k: k[2])
	df.to_csv(temp_dir / "hyperparam_search.csv")
	RO.plot_hyperparams(temp_dir)

def scratch7(RO):
	RO.rename_move_delete(result_key="sigma-opt_sigmaopt-interval-branches_brownian_motion", move="old/sigma-opt_sigmaopt-interval-branches_brownian_motion")

	def convert_epoch_estimates(json_file):
		train_estimates = lj(json_file)

		new_dict = {}
		for var, var_estimates in train_estimates.items():
			names = list(var_estimates[0].keys())
			values = [list(ve.values()) for ve in var_estimates]
			new_dict[var] = {'names': names, 'values': values}

		json_file.write_text(json.dumps(new_dict))

		# train_estimates = {i: {var: list(var_estimates[i].values()) for var, var_estimates in train_estimates.items()} for i in epoch_subset}

	for json_file in (RO.folder / result_key / "epoch_estimates").glob("*.json"):
		convert_epoch_estimates(json_file)

def scratch8(RO):
	# Color tree by bioproject
	del RO.results_dict

	tt = pp.loadTree(
		RO.params['tree_file'],
		internal=True,
		abs_time=2023
	)

	estimates = pd.read_csv(RO.folder / result_key / "sampling_features_profile_CIs.csv", index_col=0)
	branch_df = pd.read_csv(RO.results_dict[result_key]["fit_model_params"]["sampling_features"]["states"], index_col=0)

	def bin_estimate(e):
		if e == 1:
			return 0
		elif e > 1:
			return 

	faux_estimates = {v: 0 if e == 1 else 1 if e > 1 else -1 for v, e in estimates['initial_mle'].items()}
	multiplier = [faux_estimates[v] for v in sampling_df.columns]
	bg_fitness = (sampling_df * multiplier).sum(axis=1)
	fit_dict =  {n: 0 if e == 1 else 1 if e > 1 else -1 for n, e in bg_fitness.items()}
	branch_dict = {n.traits['name']: fit_dict[n.traits['name'].split("_interval")[0]] for n in tt.Objects}

	palette = sns.color_palette("coolwarm", as_cmap=True).with_extremes(bad='white', over='black')
	c_func, cmap, norm = pp.continuousFunc(trait_dict=branch_dict, trait="name", cmap=palette, vmin=-1, vmax=1, norm="norm")

	fig, ax = plt.subplots(1, 1, figsize=(12, 25))
	ax = pp.plotTraitAx(
		ax,
		tt,
		edge_c_func=c_func,
		node_c_func=c_func,
		s_func=lambda x: 0,
		tip_names=False,
		tips=False,
		zoom=False,
		title="Branch Estimate Categories",
	)
	plt.tight_layout()
	plt.savefig(RO.folder / result_key / "figures" / "branch_estimate_categories.png", dpi=300)
	plt.close("all")

def scratch9(RO):
	# Color tree by bioproject + brownian

	tt = pp.loadTree(
		RO.params['tree_file'],
		internal=True,
		abs_time=2023
	)

	# Get bioproject fitness
	bioproj_estimates = pd.read_csv(RO.folder / result_key / "sampling_features_profile_CIs.csv", index_col=0)
	bioproj_df = pd.read_csv(RO.results_dict[result_key]["fit_model_params"]["sampling_features"]["states"], index_col=0)
	bioproj_fit_dict = (bioproj_df * [bioproj_estimates.loc[v, "initial_mle"] for v in bioproj_df.columns]).sum(axis=1).to_dict()
	bioproj_branch_dict = {n.traits['name']: bioproj_fit_dict[n.traits['name'].split("_interval")[0]] for n in tt.Objects}

	# Get brownian fitness
	brownian_estimates = lj(RO.folder / result_key / "brownian_motion_estimates.json")
	brownian_branch_dict = {n.traits['name']: brownian_estimates[n.traits['name'].split("_interval")[0]] for n in tt.Objects}

	# Get combined fitness

	# Get min, max of both fitness types
	abs_min = min(bioproj_estimates["initial_mle"].min(), min(brownian_estimates.values()))
	abs_max = max(bioproj_estimates["initial_mle"].max(), max(brownian_estimates.values()))

	# Set up plot
	plt.rcParams.update({'font.size': 20})
	fig, axs = plt.subplots(1, 2, figsize=(12, 25))

	# Plot Bioproject
	c_func, cmap, norm = pp.continuousFunc(trait_dict=bioproj_branch_dict, trait="name", cmap=sns.color_palette("coolwarm", as_cmap=True), center=1, vmin=abs_min, vmax=abs_max, norm="norm")
	axs[0] = pp.plotTraitAx(
		axs[0],
		tt,
		edge_c_func=c_func,
		node_c_func=c_func,
		s_func=lambda x: 0,
		tip_names=False,
		tips=False,
		zoom=False,
		title="Bioproject Estimates",
	)

	# Plot brownian
	c_func, cmap, norm = pp.continuousFunc(trait_dict=brownian_branch_dict, trait="name", cmap=sns.color_palette("coolwarm", as_cmap=True), center=1, vmin=abs_min, vmax=abs_max, norm="norm", null_color="limegreen")
	axs[1] = pp.plotTraitAx(
		axs[1],
		tt,
		edge_c_func=c_func,
		node_c_func=c_func,
		s_func=lambda x: 0,
		tip_names=False,
		tips=False,
		zoom=False,
		title="Brownian Estimates",
	)
	axs[1] = pp.add_cmap_colorbar(axs[1], cmap, norm=norm)

	plt.tight_layout()
	plt.savefig(RO.folder / result_key / "figures" / "bioproject_brownian_estimates.png", dpi=300)
	plt.close("all")

def Jan29():
	RO = ResultsObj("data_new/analysis/three_sampling_intervals")
	for result_key, results_dict in RO.results_dict.items():
		fit_model_params = results_dict["fit_model_params"]

		results_dir = RO.folder / result_key
		results_dir.mkdir(exist_ok=True, parents=True)

		(results_dir / "fit_model_params.json").write_text(json.dumps(fit_model_params, indent=4))

def Jan29_2():
	result_key = "full_model_birth_features+brownian_motion+sampling_background_TV+sampling_features"
	dir = Path("data_new/analysis/three_sampling_intervals") / result_key

	df = pd.read_csv(dir / "hyperparam_search.csv")

	print(df[df["sigma"] == 2].sort_values("mean_test_loss")[["lamb", "sigma", "lr", "n_epochs", "mean_test_loss", "mean_train_loss"]])

def Jan29_3():
	RO = ResultsObj("data_new/analysis/three_sampling_intervals")
	RO.summarize_searches()

	df = pd.read_csv(RO.folder / "hyperparam_search.csv", index_col=0)
	print(df.sort_values(by="mean_train_loss").head())

def Sep21():
	config_file = Path().resolve().parent / "configs" / "test_config.yaml"
	config = process_yaml(config_file.read_text())

	

if __name__ == "__main__":
	Sep21()
