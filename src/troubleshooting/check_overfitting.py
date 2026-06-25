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
from troubleshooting.plot_variable_densities import plot_densities
import plotly.express as px

def lj(file):
	return json.loads(Path(file).read_text())

def plot_single_epochs(train_losses, test_losses, out_file, title=""):
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

	plt.axvline(best_train_idx, color='blue', ls='dashed')
	plt.axvline(best_test_idx, color='orange', ls='dashed')
	plt.legend()

	if title:
		plt.title(title)

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
				plot_single_epochs(train_losses, test_losses, plot_out_file, plot_out_file.name.replace(".png", ""))
		return

	if not train_loss_file.exists():
		print(f"!! No training losses found for {ej.stem}")
		return

	print(f"\n======== {combo_key}, fold {fold} ========")

	train_losses = json.loads((result_dir / "train_losses" / ej.name).read_text())
	test_losses = [float("nan")] * len(train_losses)
	epoch_subset = list(range(0, len(train_losses), 100)) + [len(train_losses) - 1]

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
	plot_single_epochs(train_losses, test_losses, plot_out_file, plot_out_file.name.replace(".png", ""))

	return test_losses_out

def train_test_agg(RO, test_losses_dir, result_dir):
	test_loss_jsons = list(test_losses_dir.glob("*.json"))

	if not test_loss_jsons:
		print(f"No loss files in {test_losses_dir}")
		return

	losses_dict = {}
	# id_vars = ["lr", "sigma", "lamb", "fold", "subset"]
	id_vars = ["lr", "sigma", "lamb", "n_epochs", "fold", "subset"]
	for ej in test_loss_jsons:
		match = re.search(r"lamb=(.*)_reg_type=(.*)_sigma=(.*)_lr=(.*)_n_epochs=(.*)_fold-(.*)", ej.stem)
		lamb, reg_type, sigma, lr, n_epochs, fold = match.groups()

		fold_result = lj(result_dir / ej.name)
		resume_from = fold_result["resume_from"]
		for subset in ['train', 'test']:
			name = ej.stem
			
			fold_loss_json = result_dir / f"{subset}_losses" / ej.name
			fold_losses = json.loads(fold_loss_json.read_text())

			fold_dict = dict(
				lamb=float(lamb),
				sigma=float(sigma),
				n_epochs=int(n_epochs),
				lr=float(lr),
				fold=int(fold),
				name=name,
				subset=subset,
				)
			id_tuple = tuple([fold_dict[k] for k in id_vars])
			fold_dict = {**fold_dict, **{i + resume_from: l for i, l in enumerate(fold_losses)}}

			if not losses_dict.get(id_tuple, False):
				losses_dict[id_tuple] = fold_dict
			else:
				print(f"\nUPDATING RESUMED RUN -- prev epochs: {losses_dict[id_tuple]['n_epochs']}, new: {fold_dict['n_epochs']}")
				losses_dict[id_tuple].update(fold_dict)

	info_cols = ["lr", "sigma", "lamb", "n_epochs", "fold", "name", "subset"]
	df = pd.DataFrame.from_dict(losses_dict, orient="index").set_index(info_cols).sort_index()
	df.to_csv(result_dir / "all_epoch_losses.csv")

	train_df = df.xs("train", level="subset")
	test_df = df.xs("test", level="subset")

	# 2025-12-19 
	# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# commenting out the below and adding here, because I want 
	# to be able to know for each fold the min loss / location. I'm not quite
	# sure why I did it the other way in the first place.
	test_df = test_df.dropna(how="all", axis=1)
	train_df = train_df.loc[test_df.index, test_df.columns]

	info_df_cols = [
		"test_best_loss", "test_best_epoch", "train_loss_at_best_test", 
		"train_best_loss", "train_best_epoch", "test_loss_at_best_train", 
		"last_epoch", "test_loss_at_last_epoch", "train_loss_at_last_epoch",
		"test_has_nan", "train_has_nan"
		]
	info_df = pd.DataFrame(index=test_df.index, columns=info_df_cols) 
	
	# Oh, maybe because there is no take_along_axis for pandas dataframes so I have to do this...
	test_np = test_df.to_numpy()
	train_np = train_df.to_numpy()

	last_epoch = train_df.apply(pd.Series.last_valid_index, axis=1)
	last_epoch_col_idxs = test_df.columns.get_indexer(last_epoch).reshape(-1, 1)
	info_df.loc[:, 'last_epoch'] = last_epoch
	info_df.loc[:, 'test_loss_at_last_epoch'] = np.take_along_axis(test_np, last_epoch_col_idxs, axis=1).ravel()
	info_df.loc[:, 'train_loss_at_last_epoch'] = np.take_along_axis(train_np, last_epoch_col_idxs, axis=1).ravel()

	test_best_epoch = test_df.idxmin(axis=1)
	test_best_epoch_col_idxs = test_df.columns.get_indexer(test_best_epoch).reshape(-1, 1)
	info_df.loc[:, 'test_best_epoch'] = test_best_epoch
	info_df.loc[:, 'test_best_loss'] = test_df.min(axis=1)
	info_df.loc[:, 'train_loss_at_best_test'] = np.take_along_axis(train_np, test_best_epoch_col_idxs, axis=1).ravel()

	train_best_epoch = train_df.idxmin(axis=1)
	train_best_epoch_col_idxs = train_df.columns.get_indexer(train_best_epoch).reshape(-1, 1)
	info_df.loc[:, 'train_best_epoch'] = train_best_epoch
	info_df.loc[:, 'train_best_loss'] = train_df.min(axis=1)
	info_df.loc[:, 'test_loss_at_best_train'] = np.take_along_axis(test_np, train_best_epoch_col_idxs, axis=1).ravel()

	info_df["test_has_nan"] = False
	info_df.loc[info_df['test_loss_at_best_train'].isna() == True, "test_has_nan"] = True
	info_df["train_has_nan"] = False
	info_df.loc[info_df['train_best_loss'].isna() == True, "train_has_nan"] = True

	info_df = info_df.sort_values(by="test_best_loss")
	info_df.to_csv(result_dir / "all_fold_stats.csv")

	df_mean = info_df.groupby(['lamb', 'sigma', 'lr']).mean().sort_values(by="test_loss_at_best_train")
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# Commented out 2025-12-19
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# Mask train df with nan where we don't have test values
	# and get best train loss, location and test loss at best
	# train epoch
	
	# for idx, idf in train_df.iterrows():
	# 	min_epoch = idf.idxmin()
	# 	info_df.loc[idx, ["train_best_loss", "train_best_epoch", "loss_at_best_train"]] = [
	# 		idf.min(),
	# 		min_epoch,
	# 		test_df.loc[idx, min_epoch]
	# 	]

	# # Get mean of the stats
	# df_mean = info_df.groupby(['lamb', 'sigma', 'lr', 'n_epochs']).mean().sort_values(by="loss_at_best_train")
	
	# # https://stackoverflow.com/questions/54307300/what-causes-indexing-past-lexsort-depth-warning-in-pandas
	# # completely overrides sorting the values above but I don't feel like debugging removing it
	# df_mean = df_mean.sort_index()

	# # Actually, we need to get the mean of the 
	# # "best epoch" differently...
	# # ================================================
	# test_df = test_df.reset_index().drop(columns="fold")
	
	# for key, kdf in test_df.groupby(["lamb", "sigma", "lr"]):
	# 	kdf = kdf.dropna(how="all", axis=1)

	# 	# We stop when things converge, so need to 
	# 	# fill rest of epochs with last value, otherwise
	# 	# our means get way off
	# 	kdf = kdf.ffill(axis=1)

	# 	means = kdf.mean(axis=0, skipna=True)
	# 	means = means.dropna().drop(["lamb", "sigma", "lr"])
		
	# 	mean_min_loss = means.min()
	# 	mean_min_loss_loc = means.idxmin()

	# 	df_mean.loc[key, "best_loss"] = mean_min_loss
	# 	df_mean.loc[key, "best_epoch"] = mean_min_loss_loc
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

	melted = pd.melt(
		df_mean, 
		id_vars=["incomplete"], 
		value_vars=['test_best_loss', 'test_loss_at_last_epoch', 'test_loss_at_best_train'], 
		value_name='loss', var_name="loss_type", 
		ignore_index=False).reset_index()
	
	melted['epochs_lr'] = melted[['last_epoch', 'lr']].astype(str).agg('_'.join, axis=1)

	colors_a = ["maroon", "red", "darkorange", "gold", "yellowgreen", "forestgreen", "teal", "blue", "darkviolet"]
	colors_b = ["firebrick", "darkorange", "xkcd:gold", "yellowgreen", "darkturquoise", "blue", "xkcd:muted blue", "darkviolet", "deeppink"]

	melted = melted[melted["incomplete"] == False]

	if query:
		melted = melted.query(query)

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
			palette=colors_b[0:len(melted["lamb"].unique())], 
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
			palette=colors_b[0:len(melted["sigma"].unique())], 
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

# def check_stopping(results_dir):
# 	wide = pd.read_csv(results_dir / "all_epoch_losses.csv")
# 	wide["combo_key"] = wide.apply(lambda row: get_key(row), axis=1)

# 	tests = wide[wide["subset"]=="test"]

# 	print("dropping")
# 	tests = tests.drop(columns=["subset", "fold", "lamb", "lr", "sigma"])

# 	print("for loop")
# 	for key, df in tests.groupby(["lamb", "lr", "sigma"]):
# 		means = df.mean(axis=0, skipna=True)
# 		means = means.dropna()
# 		print(means)

# 		breakpoint()

def find_test_losses(RO, result_key, del_RO=True, n_threads=4):
	results_dir = RO.folder / result_key

	fit_model_params = copy.deepcopy(RO.results_dict[result_key]["fit_model_params"])
	
	if 'sigma-opt' in result_key:
		bm = fit_model_params["brownian_motion"]
		cv_idxs = [bm[str(i)]['idxs'] for i in range(bm["n_folds"])]
		RO.cv_idxs = cv_idxs

	fold_test_data = [RO.loadDataByIdx(cvidxs[1]) for cvidxs in RO.cv_idxs]

	estimate_jsons = list((results_dir / "epoch_estimates").glob("*.json"))
	iterative_pE = list(RO.results_dict[result_key]["results_list"].items())[0][1]['iterative_pE']

	# del RO.results_dict
	
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

	if del_RO:
		del RO

def update(RO, result_key):
	"""
	wrangle stragglers, summarize search, 
	and plot hyperparameters
	"""
	results_dir = RO.folder / result_key

	RO.wrangle_stragglers(result_key)
	RO.summarize_search(result_key)
	RO.plot_hyperparams(results_dir)

def find_plot_test_losses(RO, result_key, n_threads=4):
	results_dir = RO.folder / result_key
	results_dir.mkdir(exist_ok=True, parents=True)

	update(RO, result_key)

	find_test_losses(RO, result_key, del_RO=True, n_threads=n_threads)

	train_test_agg(RO, results_dir / "test_losses", results_dir)

	if (results_dir / "mean_stats.csv").exists():
		plot_stats(results_dir, query="")

def update_var_names(RO):
	"""
	I should have done this when naming them initially, but ah well.
	"""
	for model_key, mdict in RO.results_dict.items():
		for param, param_dict in mdict['fit_model_params'].items():
			if isinstance(param_dict, dict) and 'names' in param_dict:
				if 'background' in param:
					param_name = param.replace("_background", "")
					param_dict['names'] = [f"{param_name}_{n}" for n in param_dict['names']]

	return RO

def get_variable_ests(RO):
	id_vars = ["model_key", "lamb", "reg_type", "sigma", "lr", "n_epochs", "fold"]
	
	RO = update_var_names(RO)

	model_dfs = []
	est_vars = {}
	for model_key, mdict in RO.results_dict.items():
		if 'test' not in model_key:
			var_name_dict = {fmp: fmpdict['names'] for (fmp, fmpdict) in mdict['fit_model_params'].items() if isinstance(fmpdict, dict) and fmpdict.get('estimate', False)}
			est_vars.update(var_name_dict)

			# make list so that order preserved
			var_types = list(var_name_dict.keys())
			var_names = id_vars + [item for sublist in [var_name_dict[k] for k in var_types] for item in sublist]

			model_dict = []
			for hp_key, hpdict in mdict['results_list'].items():
				match = re.search(r"lamb=(.*)_reg_type=(.*)_sigma=(.*)_lr=(.*)_n_epochs=(.*)", hp_key)
				lamb, reg_type, sigma, lr, n_epochs = match.groups()
				
				for fold, fold_dict in enumerate(hpdict['fold_estimates']):
					id_vals = [model_key, float(lamb), reg_type, float(sigma), float(lr), int(n_epochs), int(fold)]
					est_values = [list(np.array(fold_dict[k]).reshape(-1)) for k in var_types]
					ests = id_vals + [item for sublist in est_values for item in sublist]
					model_dict.append(ests)
			
			model_df = pd.DataFrame(model_dict, columns=var_names)
			model_dfs.append(model_df)

	df = pd.concat(model_dfs)
	df.to_csv(RO.folder / "all_models_fold_estimates.csv")

	(RO.folder / "var_names.json").write_text(json.dumps(est_vars))

	return df, est_vars

def save_ests_at_best_test(RO, result_key):
	df = pd.read_csv(RO.folder / result_key / "mean_stats.csv")
	RO = update_var_names(RO)
	pass

def model_hyperparam_heat_map(RO, result_key, query=""):
	dir = RO.folder / result_key
	df = pd.read_csv(dir / "all_fold_stats.csv")
	df = df[['lamb', 'sigma', 'lr', 'n_epochs', 'test_best_loss']]
	df = df.groupby(['lamb', 'sigma', 'lr', 'n_epochs']).mean().reset_index()
	df = df[df["sigma"] != 2]

	df = df.query(query)
	
	# want to compare lambda and sigma, ensuring that lr, n_epochs are the same
	for idx, dfg in df.groupby(["lr", "n_epochs"]): 
		# gdf = dfg[['sigma', 'lamb', 'test_best_loss']]

		if len(dfg) > 2:
			x = dfg['sigma']
			y = dfg['lamb']
			z = dfg['test_best_loss']

			fig, ax = plt.subplots(figsize=(10, 8))
			tcf = ax.tricontourf(x, y, z, levels=15, cmap='viridis')
			ax.tricontour(x, y, z, levels=5, colors='black', alpha=0.3, linewidths=0.5)
			ax.scatter(x, y, c="red")
			ax.set_xlabel('sigma')
			ax.set_ylabel('lambda')
			plt.colorbar(tcf, label='-LL')
			plt.savefig(dir / f"{idx}_lamb-vs-sigma-loss_tricountour{'_' + query if query else ''}.png", dpi=300)

def agg_mean_stats(RO):
	model_names = json.loads((RO.folder / "model_display_names.json").read_text())

	mean_stats = []
	for csv_file in RO.folder.glob("*/mean_stats.csv"):
		model_long = csv_file.parent.name
		model = model_names.get(model_long, None)

		if model:
			mdf = pd.read_csv(csv_file)
			mdf['model'] = model
			mean_stats.append(mdf)

		else:
			print(f"No display name found for {model_long}")

	if mean_stats:
		df = pd.concat(mean_stats).set_index("model")
		df = df.sort_values(by="test_best_loss")
		df.to_csv(RO.folder / "mean_stats.csv")

	else:
		print(f"No mean_stats files found in {RO.folder}")
		

@click.command()
@click.argument("command")
@click.option("--result_key", "-k", default="")
@click.option("--query", "-q", default="")
@click.option("--n_threads", "-n", default=4)
@click.option("--lamb", "-l", default=1)
@click.option("--sigma", "-s", default=1e-08)
@click.option("--lr", "-r", default=5e-05)
@click.option("--fold", "-f", default=1)
@click.option("--update", "-u", is_flag=True, default=False)
def main(command, result_key, query, lamb, sigma, lr, fold, update, n_threads):
	analysis_dir = "results"
	RO = ResultsObj(analysis_dir)

	results_dir = RO.folder / result_key

	if command == "update":
		update(RO, result_key)

	if command == "test_loss":
		if result_key:
			find_plot_test_losses(RO, result_key, n_threads=n_threads)
		else:
			for result_key in RO.results_dict.keys():
				if list((RO.folder / result_key / "epoch_estimates").glob("*.json")):
					print(f"Finding test losses for {result_key}")
					find_plot_test_losses(RO, result_key, n_threads=n_threads)
				else:
					print(f"No epoch estimates found for {result_key}")

		agg_mean_stats(RO)

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

	if command == "var_dists":
		if update:
			df, est_vars = get_variable_ests(RO)
		else:
			df = pd.read_csv(RO.folder / "all_models_fold_estimates.csv", index_col=0)
			est_vars = json.loads((RO.folder / "var_names.json").read_text())
		
		if (model_disp_file := RO.folder / "model_display_names.json").exists():
			model_display_names = json.loads((model_disp_file).read_text())
		else:
			model_display_names = ""

		# plot_densities(df, est_vars, RO.folder, group_vars=["model", "lamb"], compare_var="sigma", model_display_names=model_display_names, pre_avg=False)
		# plot_densities(df, est_vars, RO.folder, group_vars=["model", "sigma"], compare_var="lamb", model_display_names=model_display_names, pre_avg=False)

		plot_densities(
			df, 
			est_vars, 
			RO.folder, 
			group_vars=["model", "lamb"], 
			compare_var="sigma", 
			model_display_names=model_display_names, 
			pre_avg=True,
			query=query,
			)

		plot_densities(
			df, 
			est_vars, 
			RO.folder, 
			group_vars=["model", "sigma"], 
			compare_var="lamb", 
			model_display_names=model_display_names, 
			pre_avg=True,
			query=query,
			)

	if command == "heat_map":
		model_hyperparam_heat_map(RO, result_key, query)

	# if command == "stopping":
	# 	check_stopping(results_dir)

if __name__ == "__main__":
	main()
	

