import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import tensorflow as tf
import yaml
from yaml import CLoader as Loader, CDumper as Dumper
from natsort import natsorted
import json
import click
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from analysis_new.do_model_fit import ResultsObj
from analysis_new.fitness_model import BirthSamplingSite
from ecoli_analysis.branch_fitness import color_tree
from ecoli_analysis.feature_matrix import load_info
from analysis.arrayer import PhyloArrayer, PhyloDataFile
from analysis.phylo_loss import PhyloLoss, PhyloLossIterative

def lj(file):
	return json.loads(Path(file).read_text())

def plot_fitness_totals(results_dir):
	results_dir = Path(results_dir)
	analysis_dir = results_dir.parent
	
	file = results_dir / "edge_components_fitness.csv"
	RO = ResultsObj(analysis_dir)

	if not file.exists():
		calc_fitness_totals(results_dir)

	edge_fitness = pd.read_csv(file, index_col=0)

	color_tree(RO.params["tree_file"], edge_fitness["total"].to_dict(), results_dir / "figures" / "total_fitness.png", center=True)
	color_tree(RO.params["tree_file"], edge_fitness["brownian"].to_dict(), results_dir / "figures" / "brownian_fitness.png", center=True)
	color_tree(RO.params["tree_file"], edge_fitness["site"].to_dict(), results_dir / "figures" / "genetic_fitness.png", center=True)

def phylo_brownian_old(results_dir):
	fd = Path("/Users/lenorakepler/Documents/GitHub/ST131Fitness/data/analysis/3-interval_constrained-sampling/")
	orand = pd.read_csv(fd / "Est-Random_Fixed-BetaSite" / "edge_random_effects_all.csv", index_col=0)
	data, params, sample_features, est, tt = load_info(fd)
	color_tree(params["tree_file"], orand["random_fitness"].to_dict(), results_dir.parent / f"old_phylo_fitness_random.png", center=True, null_color="limegreen")

def calc_fitness_totals(results_dir):
	results_dir = Path(results_dir)
	analysis_dir = results_dir.parent
	results_key = results_dir.name

	# Load analysis and estimates
	# =============================================
	RO = ResultsObj(analysis_dir)
	fit_model_params = lj(results_dir / "estimated_fit_model_params.json")
	fit_model_params["rho"] = 0
	fit_model_params["gamma"] = 0
	
	full = lj(results_dir / "validation.json")

	iterative_pE = full["iterative_pE"]

	# Extrapolate random branch effects to held-out
	# and update fit_model_params
	# =============================================
	# Make dataframe from array
	all_segs = pd.DataFrame(RO.data.array)
	all_segs["branch_name"] = all_segs["name"].str.split("_", expand=True)[0]
	all_segs = all_segs.set_index(["abs_index", "event"])

	# Get "full" training data set and append held-out values
	bm = pd.DataFrame.from_dict(fit_model_params["brownian_motion"]["full"]["train"], orient="index")
	bm["abs_index"] = bm["abs_index"].astype(int)
	bm = bm.set_index(["abs_index", "event"])

	all_segs = pd.concat([all_segs, bm[["type_int", "parent_type_int", "parent_time_delta"]]], axis=1)

	# Add type int values for test pieces, dummy values for parent info since not used for fitness calculation
	name_to_int = {n: i for i, n in fit_model_params["brownian_motion"]["int_to_name"].items()}
	all_segs.loc[all_segs["type_int"].isna(), "type_int"] = all_segs.loc[all_segs["type_int"].isna()].apply(lambda row: name_to_int[row["branch_name"]], axis=1)
	all_segs["type_int"] = all_segs["type_int"].astype(int)
	all_segs.loc[all_segs["parent_type_int"].isna(), "parent_type_int"] = 999999
	all_segs.loc[all_segs["parent_time_delta"].isna(), "parent_time_delta"] = np.inf
	all_segs = all_segs.reset_index()

	assert (all_segs.index == all_segs["abs_index"]).all()

	fit_model_params["brownian_motion"]["info"] = all_segs.to_dict(orient="index")

	# Load analysis and estimates
	# =============================================

	fitness_model = BirthSamplingSite(
		data=RO.data, 
		fit_model_params=fit_model_params,
		iterative_pE=iterative_pE,
		save_intermediate=True,
	)

	m = fitness_model.call()

	# -----------------------------------------------------
	# Make / load dataframe of total fitness for each edge
	# in the tree for each category of fitness
	# -----------------------------------------------------
	edge_fitness = pd.DataFrame(index=fitness_model.edge_arr["name"], columns=["site", "background", "brownian"])

	edge_fitness["site"] = fitness_model.edge_site_b.numpy()
	edge_fitness["background"] = fitness_model.b_edge_background.numpy()
	edge_fitness["brownian"] = fitness_model.edge_brown.numpy()
	edge_fitness["total"] = m.edge_b

	edge_fitness.to_csv(results_dir / "edge_components_fitness.csv")

	edges = all_segs[all_segs["event"] == 4].set_index("name")
	edges = pd.concat([edges, edge_fitness], axis=1)
	edges.index.name = "name"

	edges.to_csv(results_dir / "full_brownian_info.csv")

def simple_scatter_with_xy(x_vals, y_vals, title, out_file):
	plt.scatter(x_vals, y_vals)

	mins, maxes = list(zip(plt.xlim(), plt.ylim()))
	min_val = min(mins)
	max_val = max(maxes)
	xyvals = np.linspace(min_val, max_val)
	plt.plot(xyvals, xyvals, color="gray")
	plt.title(title)

	plt.savefig(out_file, dpi=300)
	plt.close("all")

def compare_old_vs_new(results_dir):
	results_dir = Path(results_dir)
	
	file = results_dir / "edge_components_fitness.csv"

	if not file.exists():
		calc_fitness_totals(results_dir)

	edge_fitness = pd.read_csv(file, index_col=0)

	fd = Path("/Users/lenorakepler/Documents/GitHub/ST131Fitness/data/analysis/3-interval_constrained-sampling/")
	data, params, sample_features, est, tt = load_info(fd)

	# order data by old estimated fitness
	data = data.sort_values(by="site_fitness")
	edge_fitness = edge_fitness.loc[data.index]
	
	# scatter site fitness
	simple_scatter_with_xy(data["site_fitness"], edge_fitness["site"], results_dir.name, results_dir / "old_v_new.png")

	orand = pd.read_csv(fd / "Est-Random_Fixed-BetaSite" / "edge_random_effects_all.csv", index_col=0)
	orand = orand.sort_values("random_fitness")
	edge_fitness = edge_fitness.loc[orand.index]

	simple_scatter_with_xy(orand["random_fitness"], edge_fitness["brownian"], results_dir.name, results_dir / "old_v_new_random.png")

def examine_reg_effect(results_dir):
	results_dir = Path(results_dir)

	# Lamb
	# ===============
	result_files = results_dir.glob("lamb=*_reg_type=l1_sigma=1e-08_lr=5e-05_n_epochs=80000_fold-0.json")
	
	res_dict = {}
	for file in result_files:
		match = re.search(r"lamb=(.*)_reg_type=(.*)_sigma=(.*)_lr=(.*)_n_epochs=(.*)_fold-(.*)", file.stem)
		lamb, reg_type, sigma, lr, n_epochs, fold = match.groups()
		res = lj(file)
		res_dict[lamb] = res["estimates"]["brownian_motion"]
	
	df = pd.DataFrame(res_dict).T.sort_index()
	diff = df.iloc[0] - df.iloc[-1]

	print(diff.sort_values())

	df = df.sort_values(by=df.index[0], axis=1)
	df = df.loc[:, (df < 4).all(axis=0)]

	for lamb, row in df.iterrows():
		plt.scatter(list(range(len(row))), row, label=lamb)

	plt.legend()
	plt.tight_layout()
	plt.savefig(results_dir / "brownian_eff_changing_lamb.png", dpi=300)
	plt.close("all")

	# Sigma
	# ===============
	result_files = results_dir.glob("lamb=2_reg_type=l1_sigma=*_lr=5e-05_n_epochs=80000_fold-0.json")

	res_dict = {}
	for file in result_files:
		match = re.search(r"lamb=(.*)_reg_type=(.*)_sigma=(.*)_lr=(.*)_n_epochs=(.*)_fold-(.*)", file.stem)
		lamb, reg_type, sigma, lr, n_epochs, fold = match.groups()
		res = lj(file)
		res_dict[sigma] = res["estimates"]["brownian_motion"]
	
	df = pd.DataFrame(res_dict).T.sort_index()
	diff = df.iloc[0] - df.iloc[-1]

	print(diff.sort_values())

	df = df.sort_values(by=df.index[0], axis=1)
	df = df.loc[:, (df < 4).all(axis=0)]

	for sigma, row in df.iterrows():
		plt.scatter(list(range(len(row))), row, label=sigma)

	plt.legend()
	plt.tight_layout()
	plt.savefig(results_dir / "brownian_eff_changing_sigma.png", dpi=300)
	plt.close("all")

def old_vs_new_single(results_dir, combo_key):
	res = lj(results_dir / f"{combo_key}.json")
	fmp = lj(results_dir / "fit_model_params.json")

	params = lj(results_dir.parent / "params.json")

	df = pd.DataFrame(index=fmp["brownian_motion"]["names"], columns=["new"])
	df["new"] = res["estimates"]["brownian_motion"]

	fd = Path("/Users/lenorakepler/Documents/GitHub/ST131Fitness/data/analysis/3-interval_constrained-sampling/")
	orand = pd.read_csv(fd / "Est-Random_Fixed-BetaSite" / "edge_random_effects_all.csv", index_col=0)
	
	orand = orand.sort_values("random_fitness")
	orand.index = [i.split("_interval")[0] for i in orand.index]
	orand = orand.drop_duplicates()

	df = pd.concat([df, orand], axis=1).sort_values(by="random_fitness")
	df = df.dropna()
	print(df)

	simple_scatter_with_xy(df["random_fitness"], df["new"], combo_key, results_dir / f"old_v_new_random_{combo_key}.png")

	print(df.sort_values(by="new"))

	# Phylogeny colored by random fitness
	# ------------------------------------
	data = PhyloDataFile(array_file=results_dir.parent / "data.npy", data_params_file=results_dir.parent / "data_dict.pkl")
	data = pd.DataFrame(data.array)
	data["branch_name"] = data["name"].str.split("_interval", expand=True)[0]
	data = data.set_index("branch_name")
	data = data.loc[df.index, :]
	data["random_fitness"] = df.loc[data.index, "new"]
	data = data.set_index("name")
	color_tree(params["tree_file"], data["random_fitness"].to_dict(), results_dir / f"phylo_fitness_random__{combo_key}.png", center=True, null_color="limegreen")

def examine_distribution(results_dir):
	file = results_dir / "full_brownian_info.csv"
	if not file.exists():
		calc_fitness_totals(results_dir)
	df = pd.read_csv(file)

	outl = df[['brownian', 'time_step']].quantile(0.9985)
	qdf = df[(df['brownian'] < outl['brownian']) & (df['time_step'] < outl['time_step'])]

	extreme = df[df['brownian'] >= outl['brownian']]
	print(extreme)

	# for kind in ['kde']:

	# 	args = dict(data=qdf, x="time_step", y="brownian", kind=kind, fill=True)
	# 	if kind != 'kde':
	# 		del args['fill']

	# 	sns.jointplot(**args)
	# 	plt.savefig(results_dir / f"time_step_v_brownian_{kind}.png")
	# 	plt.close("all")


	# 	args = dict(data=extreme, x="time_step", y="brownian", kind=kind, fill=True)
	# 	if kind != 'kde':
	# 		del args['fill']

	# 	sns.jointplot(**args)
	# 	plt.savefig(results_dir / f"time_step_v_brownian_extreme_{kind}.png")
	# 	plt.close("all")

	# What are descendants
	lg  = df[df["brownian"] > 3]
	
	# Any associated birth nodes
	results_dir = Path(results_dir)
	analysis_dir = results_dir.parent
	results_key = results_dir.name

	RO = ResultsObj(analysis_dir)
	all_segs = pd.DataFrame(RO.data.array)
	all_segs["branch_name"] = all_segs["name"].str.split("_", expand=True)[0]
	all_segs = all_segs.set_index("branch_name")

	all_segs.loc[lg["branch_name"], :]

	brown_dict = df.set_index("branch_name")["brownian"].to_dict()

	all_segs["brownian"] = [brown_dict[i] for i in all_segs.index]

	# seems like we get these crazy estimates when have a zero-length branch that ends in a birth event
	# But is this always the case

	for s, sdf in all_segs[all_segs['time_step']==0].groupby("branch_name"):
		ev = sdf["event"].to_list()
		if 1 in ev and 4 in ev: 
			print(sdf)

	# No, plenty meet these criteria but have 1-ish brownian
	# Then is it because they do not have anything inheriting from them in the test set?
	bm_params = RO.results_dict[results_key]["fit_model_params"]["brownian_motion"]
	
	# Get "full" training data set and append held-out values
	bm = pd.DataFrame.from_dict(bm_params["full"]["train"], orient="index")

	bm["brownian"] = [brown_dict[i] for i in bm["branch_name"]]

	bmex = bm[bm["brownian"] > 4]

	for s, sdf in bmex.groupby("branch_name"):
		print(sdf)

	bm.loc[bm["parent_branch_name"].isin(lg["branch_name"]), :]
	breakpoint()

def check_gradients(results_dir):
	results_dir = Path(results_dir)
	analysis_dir = results_dir.parent
	results_key = results_dir.name

	RO = ResultsObj(analysis_dir)
	fit_model_params = RO.results_dict[results_key]["fit_model_params"]
	fit_model_params["rho"] = 0
	fit_model_params["gamma"] = 0

	hyperparams = RO.results_dict[results_key]["full"]["h_combo"]
	
	full = lj(results_dir / "validation.json")

	# add "full" to results dict, wasnt doing this before somehow
	if not RO.results_dict[results_key].get("full", False):
		RO.results_dict[results_key]["full"] = full
		RO.save()

	iterative_pE = RO.results_dict[results_key]["full"]["iterative_pE"]

	# Make dataframe from array
	all_segs = pd.DataFrame(RO.data.array)
	all_segs["branch_name"] = all_segs["name"].str.split("_", expand=True)[0]
	all_segs = all_segs.set_index(["abs_index", "event"])

	# Get "full" training data set and append held-out values
	bm = pd.DataFrame.from_dict(fit_model_params["brownian_motion"]["full"]["train"], orient="index")
	bm["abs_index"] = bm["abs_index"].astype(int)
	bm = bm.set_index(["abs_index", "event"])

	all_segs = pd.concat([all_segs, bm[["type_int", "parent_type_int", "parent_time_delta"]]], axis=1)

	# Add type int values for test pieces, dummy values for parent info since not used for fitness calculation
	name_to_int = {n: i for i, n in fit_model_params["brownian_motion"]["int_to_name"].items()}
	all_segs.loc[all_segs["type_int"].isna(), "type_int"] = all_segs.loc[all_segs["type_int"].isna()].apply(lambda row: name_to_int[row["branch_name"]], axis=1)
	all_segs["type_int"] = all_segs["type_int"].astype(int)
	all_segs.loc[all_segs["parent_type_int"].isna(), "parent_type_int"] = 999999
	all_segs.loc[all_segs["parent_time_delta"].isna(), "parent_time_delta"] = np.inf
	all_segs = all_segs.reset_index()

	# Assign variables to MLE estimates
	# ---------------------------------

	fit_model_params["brownian_motion"]["info"] = fit_model_params["brownian_motion"]["full"]["train"]

	variables = [f for f, fdict in fit_model_params.items() if isinstance(fdict, dict)]
	for variable in variables:
		fit_model_params[variable]['value'] = full["estimates"][variable]

	# Calc fitness model values,
	# check loss, and get gradients
	# ---------------------------------
	if iterative_pE:
		phylo_loss = PhyloLossIterative(graph=False, offset=1, **hyperparams)
	else:
		phylo_loss = PhyloLoss(graph=False, offset=1, **hyperparams)

	fit_model = BirthSamplingSite(
		data=RO.loadDataByIdx(RO.train_idx), 
		fit_model_params=fit_model_params,
		iterative_pE=iterative_pE,
		save_intermediate=True,
	)

	with tf.GradientTape() as tape:
		m = fit_model.call()
		poss_weights = [tf.reshape(v, [-1]) for v in fit_model.trainable_variables if v.name in fit_model.penalize]
		weights = tf.cond(len(poss_weights) > 0, lambda: tf.concat(poss_weights, axis=-1), lambda: np.array([1.01]))
		loss = phylo_loss.call(m.__dict__, weights=weights)

	gradients = tape.gradient(loss, fit_model.trainable_variables)
	gradients = [tf.convert_to_tensor(g) if isinstance(g, tf.IndexedSlices) else g for g in gradients]

	# Examine gradients
	# ---------------------------------	

	# Put into dictionary
	hg = {v.name.split(":")[0]: g.numpy() for v, g in zip(fit_model.trainable_variables, gradients)}

	# Make dataframe so easier to view
	df = pd.DataFrame(zip(hg["brownian_motion"], fit_model.brownian_eff.numpy()), columns=["gradient", "estimate"])

	# Look at penalty calculation
	fdf = pd.DataFrame(
			zip(
				tf.gather(m.brownian_eff, m.edge_type_int).numpy(),
				tf.gather(m.brownian_eff, m.edge_parent_type_int).numpy(),
				m.edge_type_int.numpy(), 
				m.edge_parent_type_int.numpy(), 
				m.edge_parent_time_delta.numpy(),
				),
			columns=["eff", "parent_eff", "type_int", "parent_type_int", "time_delta"]
			)
	
	fdf["fit_shift"] = fdf["eff"] - fdf["parent_eff"]
	fdf["num"] = -0.5 * fdf["fit_shift"]**2
	fdf["denom"] = hyperparams["sigma"] * fdf["time_delta"]
	fdf["penalty"] = fdf["num"] / fdf["denom"]

	epsilon = 0.0000005
	fdf["denom_alt"] = hyperparams["sigma"] * fdf["time_delta"] + epsilon
	fdf["penalty_alt"] = fdf["num"] / fdf["denom_alt"]
	
	probs = tf.clip_by_value(tf.math.exp(-0.5 * fit_shifts**2 / (sigma * times + epsilon)), epsilon, np.inf) # variance is proportional to time * sigma

	breakpoint()

@click.command()
@click.argument("command")
@click.argument("model")
@click.option("--combo_key", "-k")
def main(command, model, combo_key=""):
	if model == "full":
		result_key = "full_model_birth_features+brownian_motion+sampling_background_TV+sampling_features"
	elif model == "intercept":
		result_key = "intercept_birth_background+sampling_background_TV"
	elif model == "intercept-fixed":
		result_key = "intercept_birth_features+brownian_motion+sampling_features"
	elif model == "full-intercept":
		result_key = "full_model_birth_background+birth_features+brownian_motion+sampling_background_TV+sampling_features"
	elif model == "full-intercept-tvb":
		result_key = "full_model_birth_background_TV+birth_features+brownian_motion+sampling_background_TV+sampling_features"
	elif model == "nonidentifiable":
		result_key = "full_model_birth_features+brownian_motion+sampling_background_TV+sampling_features"
	elif model == "combined":
		result_key = "no_random_birth_background_TV+birth_features+sampling_background_TV+sampling_features+nso-random-only_brownian_motion"
	elif model == "combined-so":
		result_key = "no_random_birth_background_TV+birth_features+sampling_background_TV+sampling_features+sigma-opt_random-only_brownian_motion"
	else:
		result_key = model

	results_dir = Path("data_new/analysis/three_sampling_intervals") / result_key

	if command == "plot":
		plot_fitness_totals(results_dir)
	elif command == "plot_old":
		phylo_brownian_old(results_dir)
	elif command == "compare_old":
		compare_old_vs_new(results_dir)
	elif command == "reg_effect":
		examine_reg_effect(results_dir)
	elif command == "single":
		old_vs_new_single(results_dir, combo_key)
	elif command == "distribution":
		examine_distribution(results_dir)
	elif command == "gradients":
		check_gradients(results_dir)
	else:
		print("Unknown command")

if __name__ == "__main__":
	main()

