import shutil
import copy
import json
import itertools
import re
from pathlib import Path
from multiprocessing import freeze_support, set_start_method, Pool
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
from model_fit.results_obj import ResultsObj
from matplotlib.colors import PowerNorm
import plotly.express as px

def lj(file):
	return json.loads(Path(file).read_text())

def plot_single_epochs(train_losses, test_losses, out_file):
	# ==========================================
	# Plot loss over epochs
	# ==========================================
	best_train_idx = np.argmin(train_losses)
	best_test_idx = np.nanargmin(test_losses)

	test_idxs = [i for i, l in enumerate(test_losses) if not np.isnan(l)]
	test_losses_nonan = [l for i, l in enumerate(test_losses) if not np.isnan(l)]

	fig, ax = plt.subplots(1, 1)

	ax.plot(train_losses, label="train", color="blue")
	ax.set_ylabel("Train Loss")

	ax2 = ax.twinx()
	ax2.plot(test_idxs, test_losses_nonan, label="test", color="orange")
	ax2.set_ylabel("Test Loss")

	plt.axvline(best_train_idx, color='blue')
	plt.axvline(best_test_idx, color='orange')
	plt.legend()

	plt.tight_layout()
	plt.savefig(out_file, dpi=300)
	plt.close("all")

def plot_variable_epochs_single():
	sns.set_style("whitegrid")
	sns.set_context("paper")
	ax = sns.lineplot(data=df)

	if best_idx:
		plt.axvline(best_idx, color='red')

	if df.shape[1] > 20:
		# plt.setp(ax.get_legend().get_texts(), fontsize='5')
		sns.move_legend(ax, "upper right", ncol=3, fontsize=5)

	plt.tight_layout()
	plt.savefig(save_name, dpi=300)

	plt.close("all")

def plot_variable_epochs_alt(results_dir, out_dir, lamb, reg_type, sigma, lr):
	out_dir.mkdir(exist_ok=True, parents=True)

	combo_key = f"lamb={lamb}_reg_type={reg_type}_sigma={sigma}_lr={lr}"
	csv_out = results_dir / f"epoch_estimates__{combo_key}.csv"
	json_out = results_dir / f"epoch_estimates_min-loss-locs__{combo_key}.json"

	if not csv_out.exists():
		glob_str = f"lamb={lamb}_reg_type={reg_type}_sigma={sigma:g}_lr={lr}_n_epochs=*_fold-*.json"
		jsons = sorted(list((results_dir / "epoch_estimates").glob(glob_str)))

		print(glob_str)

		var_epochs = []
		min_loss_locs = {}
		for file in jsons:
			print(f"\n{file.stem}")

			fold = int(re.search(r"fold-(\d*)", file.stem).groups()[0])
			result = lj(results_dir / file.name)
			resume_from = result["resume_from"]

			train_estimates = lj(file)
			print("... loaded estimates")

			test_ls = lj(results_dir / "test_losses" / file.name)
			min_loss_locs[fold] = np.nanargmin(test_ls)

			epoch_subset = np.where(~np.isnan(test_ls))[0]

			for var, var_dict in train_estimates.items():
				print(f"\t ==> {var}")

				epoch_df = pd.DataFrame(var_dict["values"], columns=var_dict["names"]).iloc[epoch_subset, :]
				print(f"\t ... made DF")

				epoch_df["epoch"] = epoch_df.index + resume_from

				if var == "brownian_motion":
					epoch_df = epoch_df.loc[:, ~np.isclose(epoch_df.iloc[-1], 1, atol=0.01)]

				epoch_df = epoch_df.melt(var_name="feature", value_name="estimate", id_vars=["epoch"])
				epoch_df["fold"] = fold
				epoch_df["variable_type"] = var
				
				var_epochs.append(epoch_df)
				print(f"\t ... appended DF")

		print(f"\n BEGINNING CONCAT")
		df = pd.concat(var_epochs, axis=0)
		df.to_csv(csv_out, index=False)

		(json_out).write_text(json.dumps({int(i): int(l) for i, l in min_loss_locs.items()}))

	else:
		combo_key = csv_out.stem.split("__")[1]
		df = pd.read_csv(csv_out)
		min_loss_locs = {int(i): l for i, l in lj(json_out).items()}
	
	print(f"PLOTTING")
	for var, vdf in df.groupby("variable_type"):
		print(var)

		if var == "brownian_motion":
			brownian_dir = out_dir / f"{combo_key}__brownian_motion"
			brownian_dir.mkdir(exist_ok=True, parents=True)

			max_epoch = df.epoch.max()
			all_features = vdf[vdf["epoch"] == max_epoch].sort_values("estimate")["feature"].unique()
			
			n_chunks = max(1, len(all_features) // 30)
			f_chunks = np.array_split(all_features, n_chunks)
			for i, features in enumerate(f_chunks):
				chunk_df = vdf[vdf["feature"].isin(features)]
				single_variable_epoch_fg(var, chunk_df, min_loss_locs, brownian_dir / f"{combo_key}__{var}_{i}.png")

		elif var == "birth_features":
			for f_type in ['AMR', 'VIR', 'STRESS', 'PLASMID']:
				print(f"\t {f_type}")

				fdf = vdf[vdf["feature"].str.contains(f_type) == True]
				all_features = fdf['feature'].sort_values().unique()
				
				if f_type in ["AMR", "VIR"]:
					f_chunks = np.array_split(all_features, 2)
					for i, features in enumerate(f_chunks):
						chunk_df = fdf[fdf["feature"].isin(features)]
						single_variable_epoch_fg(var, chunk_df, min_loss_locs, out_dir / f"{combo_key}__{var}_{f_type}_{i}.png")

		else:
			single_variable_epoch_fg(var, vdf, min_loss_locs, out_dir / f"{combo_key}__{var}.png")
		
def single_variable_epoch_fg(var, df, min_loss_locs, out_file):
	sns.set_style("whitegrid")
	sns.set_context("paper")

	g = sns.relplot(data=df, kind="line", x="epoch", y="estimate", col="fold", hue="feature", facet_kws=dict(legend_out=False))
	sns.move_legend(g, bbox_to_anchor=(.5, 1), loc="center", borderaxespad=0, ncol=6)

	for i, ax in enumerate(g.axes[0]):
		ax.axvline(min_loss_locs[i], color='red', ls="--")

	plt.savefig(out_file, dpi=300, bbox_inches="tight")
	plt.close("all")

def plot_fold_variable_epochs(RO, dir, out_dir, lamb, reg_type, sigma, lr, fold):
	out_dir.mkdir(exist_ok=True, parents=True)

	glob_str = f"lamb={lamb}_reg_type={reg_type}_sigma={sigma}_lr={lr}_n_epochs=*_fold-{fold}.json"
	jsons = sorted(list((dir / "epoch_estimates").glob(glob_str)))

	estimates = {var: {'names': var_dict['names'], 'values': []} for var, var_dict in lj(jsons[0]).items()}
	epochs = []
	train_losses = []
	test_losses = []
	for file in jsons:
		n_epochs = re.search(r"n_epochs=(.*)_", file.stem).groups()[0]

		test_ls = lj(dir / "test_losses" / file.name)
		train_ls = lj(dir / "train_losses" / file.name)
		train_estimates = lj(file)
		result = lj(dir / file.name)

		resume_from = result["resume_from"]

		epochs += [i + resume_from for i, _ in enumerate(train_ls)]
		train_losses += train_ls
		test_losses += test_ls
		
		for var, var_dict in train_estimates.items():
			estimates[var]["values"] += var_dict["values"]

	combo_key = f"lamb={lamb}_reg_type={reg_type}_sigma={sigma}_lr={lr}_n_epochs={n_epochs}_fold-{fold}"

	brownian_dir = out_dir / "brownian_motion"
	brownian_dir.mkdir(exist_ok=True, parents=True)

	for var, var_estimates in estimates.items():
		df = pd.DataFrame(var_estimates["values"], columns=var_estimates["names"])

		if var == "brownian_motion":
			plot_brownian_epochs(RO, df, brownian_dir, f"{combo_key}__{var}")

		elif var == "birth_features":
			for f_type in ['AMR', 'VIR', 'STRESS', 'PLASMID']:
				fdf = df[sorted([c for c in df.columns if f_type in c])]
				if f_type in ["AMR", "VIR"]:
					chunks = np.array_split(fdf, 2, axis=1)
					for i, cdf in enumerate(chunks):
						RO.plot_variable_epochs(cdf, None, out_dir / f"{combo_key}__{var}_{f_type}_{i}.png")

		else:
			RO.plot_variable_epochs(df, None, out_dir / f"{combo_key}__{var}.png")

# def plot_multiple_loss_curves(result_dir):

def plot_brownian_epochs(RO, df, out_dir, out_file_base):
	# Narrow down to values substantially different than 1
	df = df.loc[:, ~np.isclose(df.iloc[-1], 1, atol=0.01)]

	# Sort columns by ascending value at last epoch
	df = df.sort_values(df.last_valid_index(), axis=1)

	# Plot all low together, all high together, 
	# for subset, sdf in [["low", df.loc[:, df.iloc[-1] < 1]], ["high", df.loc[:, df.iloc[-1] > 1]]]:
	n_chunks = max(1, df.shape[1] // 30)
	chunks = np.array_split(df, n_chunks, axis=1)
	for i, cdf in enumerate(chunks):
		RO.plot_variable_epochs(cdf, None, Path(out_dir) / f"{out_file_base}_{i}.png")

def plot_validation_brownian_epochs(RO, result_key):
	dir = RO.folder / result_key
	
	losses = pd.read_csv(dir / "losses.csv", index_col=0)['loss'].values
	best_idx = np.argmin(losses)

	fig_dir = dir / "figures" / "epochs"
	
	df = pd.read_csv(dir / "variable_brownian_motion_epochs.csv", index_col=0)
	plot_brownian_epochs(RO, df, fig_dir / f"variable_brownian_motion_epochs_{i}.png")

def get_test_losses(ej, RO, data, fit_model_params, iterative_pE):
	result_key = ej.parent.parent.name

	result_dir = RO.folder / result_key
	
	test_losses_dir = result_dir / "test_losses"
	test_losses_dir.mkdir(exist_ok=True, parents=True)

	fig_out_dir = result_dir / "figures" / "train_test_losses"
	fig_out_dir.mkdir(exist_ok=True, parents=True)

	combo_key, fold = ej.stem.split("_fold-")

	test_losses_file = test_losses_dir / ej.name
	train_loss_file = result_dir / "train_losses" / ej.name
	plot_out_file = fig_out_dir / ej.name.replace("json", "png")

	if test_losses_file.exists():
		if not plot_out_file.exists():
				train_losses = json.loads(train_loss_file.read_text())
				test_losses = json.loads(test_losses_file.read_text())
				plot_single_epochs(train_losses, test_losses, plot_out_file)
		return


	if not train_loss_file.exists():
		print(f"!! No training losses found for {ej.stem}")
		return

	print(f"\n======== {combo_key}, fold {fold} ========")

	train_losses = json.loads((result_dir / "train_losses" / ej.name).read_text())
	test_losses = [float("nan")] * len(train_losses)
	epoch_subset = list(range(0, len(train_losses), 100))

	train_estimates = json.loads(ej.read_text())
	train_estimates = {i: {var: var_estimates['values'][i] for var, var_estimates in train_estimates.items()} for i in epoch_subset}
	
	for epoch in epoch_subset:
		fit_model_params["brownian_motion"]["info"] = fit_model_params["brownian_motion"][str(fold)]["test"]

		epoch_estimates = train_estimates.pop(epoch)

		for var, var_estimate in epoch_estimates.items():
			fit_model_params[var]["value"] = var_estimate

		estimates, test_loss = RO.fit_score(
			data=data, fit_model_params=fit_model_params,
			iterative_pE=iterative_pE, reg_type=None, lamb=0, 
			sigma=0, n_epochs=1, lr=1,
			graph=False, return_opt=False, offset=1, 
			verbose=True, debug=False,
		)
		print(f"{epoch}: {test_loss}")
		test_losses[epoch] = test_loss

	test_losses_out = json.dumps(test_losses)
	test_losses_file.write_text(test_losses_out)
	plot_single_epochs(train_losses, test_losses, plot_out_file)

	return test_losses_out

def train_test_agg(test_losses_dir, result_dir):
	test_loss_jsons = list(test_losses_dir.glob("*.json"))

	losses_dict = {}
	for ej in test_loss_jsons:
		match = re.search(r"lamb=(.*)_reg_type=(.*)_sigma=(.*)_lr=(.*)_n_epochs=(.*)_fold-(.*)", ej.stem)
		lamb, reg_type, sigma, lr, n_epochs, fold = match.groups()

		fold_result = lj(result_dir / ej.name)
		resume_from = fold_result["resume_from"]
		for subset in ['train', 'test']:
			id_tuple = (lamb, reg_type, sigma, lr, fold, subset)

			fold_loss_json = result_dir / f"{subset}_losses" / ej.name
			fold_losses = json.loads(fold_loss_json.read_text())

			fold_dict = dict(
				lamb=float(lamb),
				sigma=float(sigma),
				lr=float(lr),
				fold=int(fold),
				subset=subset,
				)
			fold_dict = {**fold_dict, **{i + resume_from: l for i, l in enumerate(fold_losses)}}

			if not losses_dict.get(id_tuple, False):
				losses_dict[id_tuple] = fold_dict
			else:
				losses_dict[id_tuple].update(fold_dict)

	info_cols = ["lr", "sigma", "lamb", "fold", "subset"]
	df = pd.DataFrame.from_dict(losses_dict, orient="index").set_index(info_cols).sort_index()
	df.to_csv(result_dir / "all_epoch_losses.csv")

	train_df = df.xs("train", level="subset")
	test_df = df.xs("test", level="subset")

	info_df_cols = ["best_loss", "best_epoch", "last_epoch", "loss_at_last_epoch", "train_best_loss", "train_best_epoch", "loss_at_best_train"]
	info_df = pd.DataFrame(index=test_df.index, columns=info_df_cols)

	info_df.loc[:, 'last_epoch'] = test_df.apply(pd.Series.last_valid_index, axis=1)
	info_df.loc[:, 'loss_at_last_epoch'] = test_df.ffill(axis=1).iloc[:, -1]

	# Mask train df with nan where we don't have test values
	# and get best train loss, location and test loss at best
	# train epoch
	train_df = train_df.mask(test_df.isna())
	for idx, idf in train_df.iterrows():
		min_epoch = idf.idxmin()
		info_df.loc[idx, ["train_best_loss", "train_best_epoch", "loss_at_best_train"]] = [
			idf.min(),
			min_epoch,
			test_df.loc[idx, min_epoch]
		]

	# Get mean of the stats
	df_mean = info_df.groupby(['lamb', 'sigma', 'lr']).mean().sort_values(by="loss_at_best_train")

	# Actually, we need to get the mean of the 
	# "best epoch" differently...
	# ================================================
	test_df = test_df.reset_index().drop(columns="fold")
	
	for key, kdf in test_df.groupby(["lamb", "sigma", "lr"]):
		kdf = kdf.dropna(how="all", axis=1)

		print(kdf)

		# We stop when things converge, so need to 
		# fill rest of epochs with last value, otherwise
		# our means get way off
		kdf = kdf.ffill(axis=1)

		means = kdf.mean(axis=0, skipna=True)
		means = means.dropna().drop(["lamb", "sigma", "lr"])
		
		mean_min_loss = means.min()
		mean_min_loss_loc = means.idxmin()

		df_mean.loc[key, "best_loss"] = mean_min_loss
		df_mean.loc[key, "best_epoch"] = mean_min_loss_loc

		# info = df_mean.loc[key]
		# if info["best_loss"] < info["loss_at_last_epoch"] or info["best_loss"] < info["loss_at_best_train"]:
		# 	breakpoint()

		# print("")

	# Notate where all folds not completed, 
	# as these shouldn't be counted as "best"
	# ================================================
	df_mean["incomplete"] = False
	incomplete_count = 3 if "sigma-opt" in result_dir.name else 4
	complete = info_df.groupby(['lamb', 'sigma', 'lr'])['last_epoch'].count() != incomplete_count
	df_mean.loc[complete, "incomplete"] = True

	# Save
	# ================================================
	df_mean.to_csv(result_dir / "mean_stats.csv")

def plot_stats(result_dir, query, diff_epochs=False):
	df_mean = pd.read_csv(result_dir / "mean_stats.csv")
	df_mean = df_mean.set_index(['lamb', 'sigma', 'lr', 'last_epoch'])

	['sigma', 'lr', 'best_loss', 'best_epoch', 'last_epoch', 'loss_at_last_epoch', 'train_best_loss', 'train_best_epoch', 'loss_at_best_train', 'incomplete']

	melted = pd.melt(
		df_mean, 
		id_vars=["incomplete"], 
		value_vars=['best_loss', 'loss_at_last_epoch', 'loss_at_best_train'], 
		value_name='loss', var_name="loss_type", 
		ignore_index=False).reset_index()
	
	melted['epochs_lr'] = melted[['last_epoch', 'lr']].astype(str).agg('_'.join, axis=1)

	colors_a = ["maroon", "red", "darkorange", "gold", "yellowgreen", "forestgreen", "teal", "blue", "darkviolet"]
	colors_b = ["firebrick", "darkorange", "xkcd:gold", "yellowgreen", "darkturquoise", "blue", "xkcd:muted blue", "darkviolet", "deeppink"]

	melted = melted[melted["incomplete"] == False]

	if query:
		melted = melted.query(query)

	print(melted)

	if diff_epochs:
		marker_kwargs=dict(
			style = "epochs_lr",
			markers = True
			)
		
	else:
		marker_kwargs=dict(
			style = None,
			marker = "o",
			)

	# sns.relplot(data=melted, kind="line", x="lamb", col="sigma", y="loss", hue="loss_type", style="epochs_lr", markers=True)
	# plt.yscale('log')
	# plt.tight_layout()
	# plt.savefig(result_dir / "hyperparam_facet-plot.png", dpi=300)

	# sns.relplot(data=melted, kind="line", x="sigma", col="lamb", y="loss", hue="loss_type", style="epochs_lr", markers=True)
	# plt.yscale('log')
	# plt.xscale('log')
	# plt.tight_layout()
	# plt.savefig(result_dir / "hyperparam_facet-plot_sigma-x.png", dpi=300)

	if len(melted) > 0:

		g = sns.relplot(
			data=melted,
			kind="line", 
			x="sigma", 
			y="loss", 
			hue="lamb", 
			col="loss_type",
			palette=colors_b, 
			**marker_kwargs,
			)
		plt.yscale('log')
		plt.tight_layout()
		plt.savefig(result_dir / f"loss-sigma-x{'_' + query if query else ''}.png", dpi=300)

		g = sns.relplot(
			data=melted,
			kind="line", 
			x="lamb", 
			y="loss", 
			hue="sigma", 
			col="loss_type",
			palette=colors_b, 
			**marker_kwargs,
			)
		plt.yscale('log')
		plt.tight_layout()
		plt.savefig(result_dir / f"loss-lambda-x{'_' + query if query else ''}.png", dpi=300)

def get_key(row):
	return f"lamb={row['lamb']}_sigma={row['sigma']}_lr={row['lr']}"

def load_for_loss_plotting(result_dir):
	wide = pd.read_csv(result_dir / "all_epoch_losses.csv")
	wide["combo_key"] = wide.apply(lambda row: get_key(row), axis=1)

	df = pd.melt(wide, id_vars=["combo_key", "lr", "sigma", "lamb", "fold", "subset"], value_name='loss', var_name="epoch")
	df = df.dropna()
	df["epoch"] = df["epoch"].astype(float)

	stats = pd.read_csv(result_dir / "mean_stats.csv")

	return df, stats

def plot_some_epochs(result_dir, query_str):
	df, stats = load_for_loss_plotting(result_dir)

	ddf = df.query(query_str)
	ddf = ddf[ddf.epoch % 50 == 0]

	print(len(ddf))

	ddf = ddf[ddf.fold == 1]

	fig = px.line(ddf, x="epoch", y="loss", color='lr', line_dash='subset')
	fig.update_traces(line=dict(width=1), opacity=1)
	fig.write_html(result_dir / f"epoch_losses_{query_str}.html")

	# display_df = df[(df["sigma"] == 1e-08) & (df["lamb"] == 1)]
	# display_df = display_df[display_df["fold"] == 1]

	# fig = px.line(display_df, x="epoch", y="loss", color='combo_key', line_dash='subset')
	# fig.write_html(result_dir / "epoch_losses.html")

	# display_df = df[df["lr"] == 5e-05]
	# display_df = display_df[display_df["lamb"] == 1]
	# display_df = display_df[display_df["fold"] == 1]

def plot_some_epochs2(result_dir, query_str):
	df, stats = load_for_loss_plotting(result_dir)

	ddf = df.query(query_str)
	ddf = ddf[ddf.epoch % 50 == 0]

	print(len(ddf))

	ddf = ddf[ddf.fold == 1]

	test = ddf[ddf.subset == "test"]
	train = ddf[ddf.subset == "train"]

	# Plot test data
	print("Plotting ax1")
	ax = sns.lineplot(data=test, x="epoch", y="loss", hue="lr", palette=["green", "blue", "orange"], style="subset", style_order=["test", "train"])
	print("Plotted ax1")
	# 	ax.set_yscale('log')

	ax2 = ax.twinx()
	print("Plotting ax2")
	ax2 = sns.lineplot(data=train, x="epoch", y="loss", hue="lr", palette=["green", "blue", "orange"], style="subset", style_order=["test", "train"], legend=False)
	print("Plotted ax2")
	# 	ax2.set_yscale('log')

	# 	for idx in test[[c for c in test.columns if isinstance(c, int)]].idxmin(axis=1).values:
	# 		plt.axvline(idx, color='orange', ls="--")

	# 	for idx in train[[c for c in train.columns if isinstance(c, int)]].idxmin(axis=1).values:
	# 		plt.axvline(idx, color='blue', ls="--")
		
	# 	lamb, sigma, n_epochs, lr = i
	# 	title_str = f"{lamb=}, {sigma=}, {n_epochs=}, {lr=}"

	# 	plt.title(title_str)
	plt.tight_layout()
	plt.savefig(result_dir / f"train-test-losses_{query_str}.png", dpi=300)
	plt.close("all")

def check_stopping(results_dir):
	wide = pd.read_csv(results_dir / "all_epoch_losses.csv")
	wide["combo_key"] = wide.apply(lambda row: get_key(row), axis=1)

	tests = wide[wide["subset"]=="test"]

	print("dropping")
	tests = tests.drop(columns=["subset", "fold", "lamb", "lr", "sigma"])

	print("for loop")
	for key, df in tests.groupby(["lamb", "lr", "sigma"]):
		means = df.mean(axis=0, skipna=True)
		means = means.dropna()
		print(means)

		breakpoint()

@click.command()
@click.argument("command")
@click.argument("model")
@click.option("--query", "-q", default="")
@click.option("--n_threads", "-n", default=6)
@click.option("--lamb", "-l", default=1)
@click.option("--sigma", "-s", default=1e-08)
@click.option("--lr", "-r", default=5e-05)
@click.option("--fold", "-f", default=1)
def main(command, model, query, lamb, sigma, lr, fold, n_threads):
	if model == "full-intercept-tvb":
		result_key = "full_model_birth_background_TV+birth_features+brownian_motion+sampling_background_TV+sampling_features"
	elif model == "no-random":
		result_key = "no_random_birth_background_TV+birth_features+sampling_background_TV+sampling_features"
	elif model == "so-full-intercept-tvb":
		result_key = "sigma-opt_full_model_birth_background_TV+birth_features+brownian_motion+sampling_background_TV+sampling_features"
	elif model == "random-only":
		result_key = "sigma-opt_random-only_brownian_motion"
	elif model == "nso-random-only":
		result_key = "nso-random-only_brownian_motion"

	analysis_dir = "data_new/analysis/three_sampling_intervals"
	RO = ResultsObj(analysis_dir)

	results_dir = RO.folder / result_key

	if command == "update":
		RO.wrangle_stragglers(result_key)
		RO.summarize_search(result_key)
		RO.plot_hyperparams(results_dir)

	if command == "find":
		RO.wrangle_stragglers(result_key)
		RO.summarize_search(result_key)
		RO.plot_hyperparams(results_dir)

		fit_model_params = copy.deepcopy(RO.results_dict[result_key]["fit_model_params"])
		
		if 'sigma-opt' in result_key:
			bm = fit_model_params["brownian_motion"]
			cv_idxs = [bm[str(i)]['idxs'] for i in range(bm["n_folds"])]
			RO.cv_idxs = cv_idxs

		fold_test_data = [RO.loadDataByIdx(cvidxs[1]) for cvidxs in RO.cv_idxs]

		estimate_jsons = list((results_dir / "epoch_estimates").glob("*.json"))
		iterative_pE = list(RO.results_dict[result_key]["results_list"].items())[0][1]['iterative_pE']

		del RO.results_dict
		
		pool_args = []
		for ej in estimate_jsons:
			fold = int(ej.stem.split("fold-")[1])
			pool_args.append([ej, RO, fold_test_data[fold], fit_model_params, iterative_pE])

		if n_threads > 0:
			with Pool(n_threads) as pool:
				pool.starmap(get_test_losses, pool_args)
		else:
			for pool_arg in pool_args:
				get_test_losses(*pool_arg)

		del RO

		train_test_agg(results_dir / "test_losses", results_dir)
		plot_stats(results_dir, query="")

	if command == "agg":
		test_losses_dir = results_dir / "test_losses"
		del RO

		train_test_agg(test_losses_dir, results_dir)
		plot_stats(results_dir, query="")

	if command == "plotsumm":
		plot_stats(results_dir, query)

	if command == "brownian_epochs":
		plot_validation_brownian_epochs(RO, result_key)

	if command == "fold_epochs":
		out_dir = results_dir / "fold_epoch_variable_plots"
		# plot_variable_epochs(RO, RO.folder / result_key, out_dir=out_dir, lamb=lamb, reg_type="l1", sigma=sigma, lr=lr, fold=fold)
		plot_variable_epochs_alt(results_dir, out_dir, lamb=lamb, sigma=sigma, lr=lr, reg_type="l1")

	if command == "plot_epochs":
		plot_some_epochs(results_dir, query)
		plot_some_epochs2(results_dir, query)

	if command == "all_summ":
		RO.summarize_searches()

	if command == "stopping":
		check_stopping(results_dir)

if __name__ == "__main__":
	main()
	

