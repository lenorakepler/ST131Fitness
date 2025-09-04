import shutil
import copy
import json
import itertools
from pathlib import Path
from multiprocessing import freeze_support, set_start_method, Pool
from collections import ChainMap
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
from sklearn.model_selection import KFold, train_test_split
from analysis.phylo_obj import PhyloObjPlain
from analysis.arrayer import PhyloArrayer, PhyloDataFile
from analysis.phylo_loss import PhyloLoss, PhyloLossIterative
from analysis.optimizer import Optimizer
from analysis.fitness_model import BirthSamplingSite

# https://stackoverflow.com/questions/69924881/userwarning-starting-a-matplotlib-gui-outside-of-the-main-thread-will-likely-fa
import matplotlib
matplotlib.use('agg')

class ResultsObj():
	def __init__(self, folder, verbose=True):
		self.folder = Path(folder)
		self.verbose = verbose

		self.folder.mkdir(exist_ok=True, parents=True)

		self.data = None
		self.train_idx = []
		self.validate_idx = []
		self.cv_idxs = []
		self.results_dict = {}
		self.params = {}

		# Try loading data object, fold indices, results
		self.success = self.load()
		
	def load(self):
		data_success = False
		params_success = False
		index_success = False
		analysis_success = False

		# -----------------------------------------------------
		# Load data
		# -----------------------------------------------------
		if (self.folder / "data.npy").exists() and (self.folder / "data_dict.pkl").exists():
			self.data = PhyloDataFile(array_file=self.folder / "data.npy", data_params_file=self.folder / "data_dict.pkl")
			self.data.addColumn("abs_index", list(range(len(self.data.array))), "int64")
			data_success = True

		# -----------------------------------------------------
		# Load params
		# -----------------------------------------------------
		if (params_file := (self.folder / "params.json")).exists():
			with open(params_file, "r+") as f:
				self.params = json.load(f)
			params_success = True

		# -----------------------------------------------------
		# Load indices for cross-validation
		# -----------------------------------------------------
		if (index_file := (self.folder / "idxs.json")).exists():
			with open(index_file, "r+") as f:
				i = json.load(f)
				for k, v in i.items():
					setattr(self, k, v)
			index_success = True

		# -----------------------------------------------------
		# Load results
		# -----------------------------------------------------
		if (analysis_file := (self.folder / "analysis.json")).exists():
			with open(analysis_file, "r+") as f:
				a = json.load(f)
				self.results_dict = a
			analysis_success = True

		success_dict=dict(
						data=data_success, 
						params=params_success, 
						index=index_success, 
						results=analysis_success
						)

		if self.verbose:
			for k, v in success_dict.items():
				print(f"Loaded {k}? {v}")

		return success_dict
	
	def set_data(self, tree_file, interval_times_file, last_sample_date):
		"""
		Create and save the phylo object and data object that will be associated
		with this analysis
		"""

		# -----------------------------------------------------
		# Create
		# -----------------------------------------------------
		interval_times = [float(t) for t in Path(interval_times_file).read_text().splitlines()]

		self.phylo_obj = PhyloObjPlain(
			tree_file=Path(tree_file),
			tree_schema="newick",
			last_sample_date=last_sample_date,
		)

		self.data = PhyloArrayer(
			phylo_obj=self.phylo_obj,
			param_interval_times=interval_times,
		).toData()

		# -----------------------------------------------------
		# Save
		# -----------------------------------------------------
		self.data.save("data", self.folder)
		self.phylo_obj.save(self.folder)

		self.params.update(dict(
			tree_file=str(tree_file),
			interval_times=interval_times,
			last_sample_date=last_sample_date,
			))

		self.save()

	def set_folds(self, test_size=0.2, n_splits=4, stratify=None, random_state=8):
		"""
		# Get and set indices corresponding to
		# train and validation data sets
		# + Split train for cross-validation
		"""

		self.train_idx, self.validate_idx, _, _ = train_test_split(
			list(range(len(self.data.array))),
			[0] * len(self.data.array),
			random_state=random_state,
			test_size=test_size,
			stratify=stratify,
		)

		kf = KFold(
			n_splits=n_splits,
			shuffle=True,
			random_state=random_state,
		)
		cv_idxs = list(kf.split(self.train_idx))
		self.cv_idxs = [[np.take(self.train_idx, f).tolist() for f in cv] for cv in cv_idxs]

		idx_dict = dict(
			train_idx = self.train_idx,
			validate_idx = self.validate_idx,
			cv_idxs = self.cv_idxs,
			)

		with open(self.folder / "idxs.json", "w+") as f:
			json.dump(idx_dict, f, indent=4)

		fold_params_dict = dict(
			test_size=test_size, 
			n_splits=n_splits, 
			stratify=stratify, random_state=random_state
			)
		self.params.update(fold_params_dict)
		self.save()

	def loadDataByIdx(self, idx):
		return self.data.getSubArraySpecific(idx)

	def save(self):
		if self.results_dict:
			analysis_json = Path(self.folder) / "analysis.json"
			
			try:
				results_str = json.dumps(self.results_dict, indent=4)
				analysis_json.write_text(results_str)

			except Exception as e:
				print("Issue saving json file; aborted")
				print(e)
				breakpoint()

		if self.params:
			with open(Path(self.folder) / "params.json", "w+") as f:
				json.dump(self.params, f, indent=4)

	# * to make all parameters but self keyword only so can pass **fit_model_params
	def fit_score(self, *, data, fit_model_params,
			iterative_pE, reg_type, lamb, sigma, n_epochs, lr, graph,
			return_opt, offset=1, verbose=True, debug=False, **kwargs,
			):

		fit_model_params["rho"] = 0
		fit_model_params["gamma"] = 0

		fitness_model = BirthSamplingSite(
			data=data, 
			fit_model_params=fit_model_params,
			iterative_pE=iterative_pE,
		)

		if iterative_pE:
			phylo_loss = PhyloLossIterative(graph=graph, reg_type=reg_type, lamb=lamb, offset=offset, sigma=sigma)
		else:
			phylo_loss = PhyloLoss(graph=graph, reg_type=reg_type, lamb=lamb, offset=offset, sigma=sigma)

		opt = Optimizer(n_epochs=n_epochs, lr=lr)

		opt.verbose = verbose
		opt.debug = debug

		if return_opt:
			opt.save_values = True
		else:
			opt.save_values = False

		# Do training, save loss
		estimates, train_loss = opt.doOpt(fit_model=fitness_model, phylo_loss=phylo_loss)

		if return_opt:
			return estimates, train_loss, opt
		else:
			return estimates, train_loss

	def do_train_test(
		self, fit_model_params, train_data, test_data, reg_type, lamb, sigma,
		iterative_pE, n_epochs, lr, graph, debug,offset=1, verbose=True,
		):

			# Fit to training data
			train_fit_model_params = copy.deepcopy(fit_model_params)
			train_fit_model_params["brownian_motion"]["info"] = train_fit_model_params["brownian_motion"]["info"]["train"]
			estimates, train_loss, train_opt = self.fit_score(
				data=train_data, fit_model_params=train_fit_model_params,
				iterative_pE=iterative_pE, reg_type=reg_type, lamb=lamb, sigma=sigma, n_epochs=n_epochs, lr=lr, graph=graph,
				return_opt=True, offset=offset, verbose=verbose, debug=debug
				)

			# Get negative log likelihood of test data set given training estimates
			test_fit_model_params = copy.deepcopy(fit_model_params)
			test_fit_model_params["brownian_motion"]["info"] = test_fit_model_params["brownian_motion"]["info"]["test"]
			for variable in train_opt.fit_model.model_variables:
				test_fit_model_params[variable]['value'] = estimates[variable]
	
			_, test_loss, test_opt = self.fit_score(
				data=test_data, fit_model_params=test_fit_model_params,
				iterative_pE=iterative_pE, reg_type=None, lamb=0, sigma=False, n_epochs=1, lr=lr, graph=graph,
				return_opt=True, offset=offset, verbose=verbose, debug=debug,
				)

			return estimates, train_loss, test_loss, train_opt, test_opt

	def plot_hyperparams(self, folder):
		folder = Path(folder)

		df = pd.read_csv(folder / "hyperparam_search.csv")
		df = df.dropna(subset=["mean_test_loss"])

		# Account for fact that sigmaopt has 3 folds, regular fit has 4
		# This should have been accounted for from the beginning
		def is_complete(row, folder):
			if 'result_key' in row.index:
				res_key = row["result_key"]
			else:
				res_key = folder.name

			if "sigma-opt" in res_key:
				n_folds = 3
			else:
				n_folds = 4

			is_complete = row["fold_test_losses"].count(",") == n_folds - 1
			return is_complete

		df = df[df.apply(lambda row: is_complete(row, folder), axis=1)]

		plot_train = True

		if 'result_key' in df.columns:
			df = df[df['result_key'].str.contains('test') == False]
			
		if len(df) > 0:
			df["epoch_lr"] = df.apply(lambda row: f"{row['n_epochs']}_{row['lr']}", axis=1)
			
			if 'result_key' in df.columns:
				hue = "result_key"
				style = "epoch_lr"
				plot_train = False
				style_order = None

			else:
				hue = "epoch_lr"
				style = "subset"
				style_order=["test", "train"]
				
			xs = []
			if len(df["lamb"].unique()) > 1:
				xs.append("lamb")
			if len(df["sigma"].unique()) > 1:
				xs.append("sigma")

			for x in xs:
				sns.set_style("whitegrid")
				sns.set_context("paper")

				df["subset"] = "test"
				fig, ax = plt.subplots()
				ax = sns.lineplot(data=df, x=x, y="mean_test_loss", hue=hue, markers=True, alpha=0.8, style=style, style_order=style_order, ax=ax)

				if plot_train:
					df["subset"] = "train"
					ax2 = ax.twinx()
					sns.lineplot(data=df, x=x, y="mean_train_loss", hue=hue, markers=True, alpha=0.8, style=style, ax=ax2, style_order=style_order, legend=False)

				sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), fontsize=5)
				plt.tight_layout()
				plt.savefig(folder / f"hyperparam_search_{x}.png", dpi=300)
				plt.close("all")

	def summarize_searches(self):
		df_list = []
		for result_key, results_dict in self.results_dict.items():
			for param_key, param_dict in results_dict['results_list'].items():
				fdf = {k: v for k, v in param_dict.items() if 'estimates' not in k}
				fdf = {'result_key': result_key, **fdf, **param_dict['h_combo']}
				df_list.append(fdf)
		df = pd.DataFrame(df_list)
		df = df.sort_values(by="mean_test_loss", ascending=True)
		df.to_csv(self.folder / "hyperparam_search.csv")

		self.plot_hyperparams(self.folder)

	def summarize_search(self, result_key):
		results_dict =  self.results_dict[result_key]
		
		df_list = []
		for param_key, param_dict in results_dict['results_list'].items():
			fdf = {k: v for k, v in param_dict.items() if 'estimates' not in k}
			fdf = {**fdf, **param_dict['h_combo']}
			df_list.append(fdf)

		df = pd.DataFrame(df_list)
		df = df.sort_values(by="mean_test_loss", ascending=True)
		df = df.set_index("lamb")

		df.to_csv(self.folder / result_key / "hyperparam_search.csv")

	def epoch_estimates_to_dict(self, opt_obj, fit_model_params):
		estimates = opt_obj.values

		epoch_estimates_dict = {}
		for var in opt_obj.fit_model.model_variables:
			if (n_dims := estimates[0][var].ndim) == 1:
				var_epoch_estimates = [e[var].tolist() for e in estimates]
				
			elif (n_dims := estimates[0][var].ndim) == 2:
				var_epoch_estimates = [e[var].tolist()[0] for e in estimates]

			epoch_estimates_dict[var] = {'names': fit_model_params[var]['names'], 'values': var_epoch_estimates}

		return epoch_estimates_dict

	def do_validation(self, result_key, iterative_pE, graph, debug, best_type="best_overall"):
		# Run on full dataset
		# ------------------------------------
		results_with_overfitting = self.folder / result_key / "mean_stats.csv"

		if results_with_overfitting.exists():
			df = pd.read_csv(results_with_overfitting)
			df = df.sort_values(by="best_loss")

			best_iter = df.iloc[0]

			best_params = best_iter[['h_combo', 'lr', 'n_epochs', 'iterative_pE']].to_dict()

			breakpoint()

		else:
			df = pd.read_csv(self.folder / result_key / "hyperparam_search.csv")
			best_iter = df.iloc[0]
			best_params = best_iter[['h_combo', 'lr', 'n_epochs', 'iterative_pE']].to_dict()


		if isinstance(best_params["h_combo"], str):
			best_params["h_combo"] = eval(best_params["h_combo"])

		if (full_dict := self.results_dict[result_key].get("full", False)):
			curr_best_params = {k: full_dict.get(k, None) for k in ['h_combo', 'lr', 'n_epochs', 'iterative_pE']}
			if isinstance(curr_best_params["h_combo"], str):
				curr_best_params["h_combo"] = eval(curr_best_params["h_combo"])
			if curr_best_params == best_params:
				print(f"******* Already ran validation on best hyperparams {best_params} *******")
				return
			else:
				full_dict = best_params
		else:
			self.results_dict[result_key]["full"] = best_params
			full_dict = self.results_dict[result_key]["full"]

		print(f"******* Running validation on best hyperparams {best_params} *******")

		train = self.loadDataByIdx(self.train_idx)
		test = self.loadDataByIdx(self.validate_idx)

		fit_model_params = self.results_dict[result_key]["fit_model_params"]
		fit_model_params["brownian_motion"]["info"] = fit_model_params["brownian_motion"]["full"]

		validation_estimates, validation_train_loss, validation_test_loss, validation_train_opt, validation_test_opt = self.do_train_test(
			fit_model_params, train, test, full_dict["h_combo"]["reg_type"],
			full_dict["h_combo"]["lamb"], full_dict["h_combo"]["sigma"], iterative_pE, full_dict["n_epochs"], full_dict["lr"], graph, 
			offset=1, verbose=True, debug=debug,
		)

		self.results_dict[result_key]["full"][f"train_loss"] = float(validation_train_loss)
		self.results_dict[result_key]["full"][f"test_loss"] = float(validation_test_loss)
		self.results_dict[result_key]["full"]["train_n_epochs"] = len(validation_train_opt.losses)
		

		losses_df = pd.DataFrame([{"epoch": i, "loss": l} for i, l in enumerate(validation_train_opt.losses)])
		losses_df.to_csv(self.folder / result_key / "losses.csv", index=False)

		best_epoch = validation_train_opt.min_loss_loc

		epoch_estimates_dict = self.epoch_estimates_to_dict(validation_train_opt, self.results_dict[result_key]['fit_model_params'])

		self.results_dict[result_key]["full"]["estimates"] = {}
		self.results_dict[result_key]["full"]["named_estimates"] = {}
		for var, var_dict in epoch_estimates_dict.items():
			var_df = pd.DataFrame(var_dict['values'], columns=var_dict['names'])
			var_df.index.name = "epoch"
			var_df.to_csv(self.folder / result_key / f"variable_{var}_epochs.csv")

			self.results_dict[result_key]["full"]["estimates"][var] = var_df.loc[best_epoch].to_list()
			self.results_dict[result_key]["full"]["named_estimates"][var] = var_df.loc[best_epoch].to_dict()

			var_estimates_json = json.dumps(var_df.loc[best_epoch].to_dict())
			(self.folder / result_key / f"{var}_estimates.json").write_text(var_estimates_json)

		self.save()

		validation_out = json.dumps(self.results_dict[result_key]["full"], indent=4)
		(self.folder / result_key / f"validation.json").write_text(validation_out)
		self.plot_epochs(result_key)

	def plot_variable_epochs(self, df, best_idx, save_name):
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

	def plot_epoch_losses(self, losses, dir, filename_base=""):
		# ==========================================
		# Plot loss over epochs
		# ==========================================
		best_idx = np.argmin(losses)
		y_limit = np.quantile(losses, .95)

		x_limit = np.where(losses < y_limit)[0][0]

		plt.plot(list(range(len(losses)))[x_limit:], losses[x_limit:])
		plt.axvline(best_idx, color='red')
		plt.tight_layout()
		plt.savefig(dir / f"{filename_base}epochs_95.png", dpi=300)
		plt.close("all")

		# ==========================================
		# Plot loss over epochs, zoomed in
		# ==========================================
		y_limit = np.quantile(losses, .5)
		x_limit = np.where(losses < y_limit)[0][0]

		plt.plot(list(range(len(losses)))[x_limit:], losses[x_limit:])
		plt.axvline(best_idx, color='red')
		plt.tight_layout()
		plt.savefig(dir / f"{filename_base}epochs_50.png", dpi=300)
		plt.close("all")

	def plot_epochs(self, result_key):
		losses = pd.read_csv(self.folder / result_key / "losses.csv", index_col=0)['loss'].values
		best_idx = np.argmin(losses)

		save_dir = self.folder / result_key
		fig_dir = save_dir / "figures" / "epochs"
		fig_dir.mkdir(exist_ok=True, parents=True)

		self.plot_epoch_losses(losses, fig_dir, filename_base="")

		# # ==========================================
		# # Plot values over epochs
		# # ==========================================
		fit_model_params = self.results_dict[result_key]["fit_model_params"]
		for var, param_dict in fit_model_params.items():
				if isinstance(param_dict, dict) and param_dict.get('estimate', False):
					df = pd.read_csv(self.folder / result_key / f"variable_{var}_epochs.csv", index_col=0)

					if 'loss' in df.columns:
						df = df.drop(columns=["loss"])
						df.to_csv(self.folder / result_key / f"variable_{var}_epochs.csv")

					if var == 'birth_features':
						for f_type in ['AMR', 'VIR', 'STRESS', 'PLASMID']:
							cols = sorted([c for c in df.columns if f_type in c])
							if f_type in ["AMR", "VIR"]:
								half_cols = len(cols) // 2
								self.plot_variable_epochs(df[cols[0:half_cols]], best_idx, fig_dir / f"variable_{var}-{f_type}_1_epochs.png")
								self.plot_variable_epochs(df[cols[half_cols:]], best_idx, fig_dir / f"variable_{var}-{f_type}_2_epochs.png")
							else:
								self.plot_variable_epochs(df[cols], best_idx, fig_dir / f"variable_{var}-{f_type}_epochs.png")

					elif var == 'brownian_motion':
						pass

					else:
						self.plot_variable_epochs(df, best_idx, fig_dir / f"variable_{var}_epochs.png")

	def update_from_folds_list(self, result_key, fold_dicts):
			results_df = pd.DataFrame(fold_dicts)

			for combo_key, cdf in results_df.groupby('combo_key'):
				self.results_dict[result_key]["results_list"].update(
					{
					combo_key:
						dict(
							h_combo = cdf.iloc[0]['h_combo'],
							lr=float(cdf.iloc[0]['lr']),
							n_epochs=int(cdf.iloc[0]['n_epochs']),
							iterative_pE=bool(cdf.iloc[0]['iterative_pE']),
							mean_train_loss = cdf['train_loss'].mean(),
							mean_test_loss = cdf['test_loss'].mean(),
							fold_estimates = cdf['estimates'].to_list(),
							fold_train_losses = cdf['train_loss'].to_list(),
							fold_test_losses = cdf['test_loss'].to_list(),
						)
					}
				)

			self.save()

	def wrangle_stragglers(self, result_key):
		"""
		Can run this if something goes wrong and results don't get updated after hyperparameter search
		"""

		res_folder = self.folder / result_key
		results_list = [json.loads(f.read_text()) for f in res_folder.glob("lamb=*.json")]

		if results_list:
			self.update_from_folds_list(result_key, results_list)

	def rename_move_delete(self, result_key, rename=False, move=False, delete=False):
		dir = self.folder / result_key

		if rename:
			new_result_key = rename

			# Change in dict
			self.results_dict[new_result_key] = self.results_dict[result_key]
			del self.results_dict[result_key]

			# Change base folder name
			dir.rename(self.folder / new_result_key)

			self.summarize_search(new_result_key)
			self.plot_hyperparams(self.folder / new_result_key)

		if move:
			new_dir = self.folder / move
			new_dir.parent.mkdir(exist_ok=True, parents=True)
			dir.rename(new_dir)

			del self.results_dict[result_key]

		if delete:
			shutil.rmtree(dir)
			del self.results_dict[result_key]

		# Update graphs and summaries
		self.summarize_searches()
		self.plot_hyperparams(self.folder)
		self.save()

def test_param_fold(results_obj, result_key, h_combo, fold, fit_model_params, iterative_pE, n_epochs, lr, graph, debug):
	combo_key = "_".join(sorted([f"{k}={v}" for k, v in h_combo.items()])) + f"_lr={lr}_n_epochs={n_epochs}"
	
	results_dir = results_obj.folder / result_key
	result = (results_dir / f"{combo_key}_fold-{fold}.json")

	fig_dir = results_dir / "figures" / "train_losses"
	fig_dir.mkdir(exist_ok=True, parents=True)

	losses_dir = results_dir / "train_losses"
	losses_dir.mkdir(exist_ok=True, parents=True)

	estimates_dir = results_dir / "epoch_estimates"
	estimates_dir.mkdir(exist_ok=True, parents=True)

	# ---------------------------------------------------------------------
	# If we have tested this fold with this hyperparameter set,
	# return empty dictionary
	# ---------------------------------------------------------------------
	if result.exists():
		print(f"\nAlready tested {combo_key} on fold {fold}, moving on")
		return {}

	# ---------------------------------------------------------------------
	# If this hyperparameter set has been tested but with a different
	# number of epochs, check if n_epochs surpassed or if there is an
	# analysis to resume from
	# ---------------------------------------------------------------------
	else:
		resume = False

		regex_combo = "_".join(sorted([f"{k}={v}" for k, v in h_combo.items()])) + f"_lr={lr}_n_epochs=*"
		regex_json = f"{regex_combo}_fold-{fold}.json"

		results = [[f, int(f.stem.split("n_epochs=")[1].split("_fold")[0])] for f in results_dir.glob(regex_json)]
		sorted_results = sorted(results, key=lambda r: r[1], reverse=True)

		more_epochs = [r for r in sorted_results if r[1] > n_epochs]
		less_epochs = [r for r in sorted_results if r[1] < n_epochs]

		# If we have run this hyperparameter set for MORE epochs,
		# return empty dictionary
		if any(more_epochs):
			print(f"\nAlready tested {combo_key} with {more_epochs[0][1]} epochs (> than {n_epochs})")
			return {}

		if any(less_epochs):
			resume = True
			starting_json, starting_epochs = less_epochs[0]
			starting_dict = json.loads(starting_json.read_text())
	
	# ---------------------------------------------------------------------
	# If resuming, set up
	# ---------------------------------------------------------------------
	if resume:
		resume_from = starting_dict.get("min_loss_epoch", starting_dict["n_epochs"])
		remaining_n_epochs = n_epochs - resume_from
		for var, estimate in starting_dict["estimates"].items():
			fit_model_params[var]["value"] = estimate

	else:
		remaining_n_epochs = n_epochs
		resume_from = 0
			
	# ---------------------------------------------------------------------
	# Fit + score model with this hyperparameter combination
	# and plot training loss curve
	# ---------------------------------------------------------------------
	print(f"\nTesting model {result_key} using hyperparameters {combo_key} ---> fold {fold}")
	if resume: print(f"Resuming from {starting_json.stem} at epoch {resume_from}")

	results_dir.mkdir(exist_ok=True, parents=True)
	results_dict = dict(
		combo_key=combo_key,
		fold=fold,
		h_combo=h_combo,
		n_epochs=n_epochs,
		lr=lr,
		iterative_pE=iterative_pE,
		resume_from=resume_from,
		remaining_n_epochs=remaining_n_epochs,
		)

	if resume:
		results_dict["resume_from"] = resume_from

	# Get data for training and testing
	cv_train_idx, cv_test_idx = results_obj.cv_idxs[fold]
	cv_train = results_obj.loadDataByIdx(cv_train_idx)
	cv_test = results_obj.loadDataByIdx(cv_test_idx)

	# Set brownian motion info for this fold
	fit_model_params["brownian_motion"]["info"] = fit_model_params["brownian_motion"][str(fold)]

	estimates, train_loss, test_loss, train_opt, test_opt = results_obj.do_train_test(
		fit_model_params, cv_train, cv_test, h_combo['reg_type'],
		h_combo['lamb'], h_combo['sigma'], iterative_pE, remaining_n_epochs, lr, graph, 
		offset=1, verbose=True, debug=debug,
	)

	min_loss_epoch = train_opt.min_loss_loc
	results_obj.plot_epoch_losses(train_opt.losses, fig_dir, filename_base=f"{combo_key}_fold-{fold}_")

	# ---------------------------------------------------------------------
	# Write results to file and return dictionary
	# ---------------------------------------------------------------------
	results_dict.update(
		dict(
			test_loss=float(test_loss),
			train_loss=float(train_loss),
			estimates={variable: e.tolist() for variable, e in estimates.items()},
			min_loss_epoch=int(min_loss_epoch + resume_from),
			)
		)

	epoch_estimates_dict = results_obj.epoch_estimates_to_dict(train_opt, fit_model_params)
	epoch_estimates_out = json.dumps(epoch_estimates_dict)
	(estimates_dir / f"{combo_key}_fold-{fold}.json").write_text(epoch_estimates_out)

	folds_dict_out = json.dumps(results_dict, indent=4)
	(results_dir / f"{combo_key}_fold-{fold}.json").write_text(folds_dict_out)

	losses_out = json.dumps([float(l) for l in train_opt.losses])
	(losses_dir / f"{combo_key}_fold-{fold}.json").write_text(losses_out)

	return results_dict

def init_model_params(results_obj, config, sigma_opt):
	# -----------------------------------------------------
	# Init fitness model parameters
	# -----------------------------------------------------
	fit_model_params = config["fit_model_params"]

	# Sampling background
	# -----------------------------------------------------
	s_bg_changepoints = fit_model_params["sampling_background"]["changepoints"]
	if (root_time := results_obj.data.root_time) not in s_bg_changepoints:
		s_bg_changepoints = [root_time] + s_bg_changepoints

	fit_model_params["sampling_background"]["changepoints"] = s_bg_changepoints
	fit_model_params["sampling_background"]["interval_mapping"] = [int(np.where(time > s_bg_changepoints)[0][-1]) if time != s_bg_changepoints[0] else 0 for time in results_obj.data.param_interval_times]
	fit_model_params["sampling_background"]["names"] = [str(c) for c in s_bg_changepoints]

	# Birth background
	# -----------------------------------------------------
	b_bg_changepoints = fit_model_params["birth_background"]["changepoints"]
	if (root_time := results_obj.data.root_time) not in b_bg_changepoints:
		b_bg_changepoints = [root_time] + b_bg_changepoints

	fit_model_params["birth_background"]["changepoints"] = b_bg_changepoints
	fit_model_params["birth_background"]["interval_mapping"] = [int(np.where(time > b_bg_changepoints)[0][-1]) if time != b_bg_changepoints[0] else 0 for time in results_obj.data.param_interval_times]
	fit_model_params["birth_background"]["names"] = [str(c) for c in b_bg_changepoints]

	# Brownian motion
	# -----------------------------------------------------
	all_brownian = json.loads(Path(fit_model_params["brownian_motion"]["states"]).read_text())
	
	if sigma_opt:
		fit_model_params["brownian_motion"].update({k: v for k, v in all_brownian.items() if k != "folds"})
		
		# If doing straight sigma optimization, we calculate the test likelihood using parent type int,
		# not the test piece's own type int
		for fold, fold_dict in all_brownian["folds"].items():
			for i, i_dict in fold_dict["test"].items():
				i_dict["type_int"] = i_dict["parent_type_int"]

		fit_model_params["brownian_motion"].update(all_brownian["folds"])
	else:
		fit_model_params["brownian_motion"].update(all_brownian)

	# Birth and sampling features
	# -----------------------------------------------------
	fit_model_params["birth_features"]["names"] = pd.read_csv(fit_model_params["birth_features"]["states"], index_col=0).columns.to_list()
	fit_model_params["sampling_features"]["names"] = pd.read_csv(fit_model_params["sampling_features"]["states"], index_col=0).columns.to_list()

	return fit_model_params

def crossvalidate(analysis_dir, hyper_param_values, config, debug, n_epochs=20000, lr=0.01, graph=True, n_threads=4):
	"""
	Do cross-validation with given variables 
	and given hyperparameter values

	If cross-validation with given variables exists,
	run new hyperparameter values
	"""

	# Hard to debug on parallel threads
	# or if tensorflow is in graph mode
	if debug:
		n_threads = 0
		graph = False

	sigma_opt = config.get('sigma_opt', False)

	# -----------------------------------------------------
	# Load results object
	# -----------------------------------------------------
	# results_obj = ResultsObj(folder=Path(config["data_dir"]) / "analysis" / config["analysis_name"])
	results_obj = ResultsObj(analysis_dir)
	
	if not results_obj.success["index"]:
		results_obj.set_data(
			tree_file=Path(config["data_dir"]) / config["interval_tree_name"] / "phylo.nwk",
			interval_times_file=Path(config["data_dir"]) / config["interval_tree_name"] / "interval_times.txt",
			last_sample_date=config["last_sample_date"]
		)
		results_obj.set_folds(test_size=0.2, n_splits=4, stratify=None, shuffle=True, random_state=8)

	# -----------------------------------------------------
	# Init fitness model parameters
	# -----------------------------------------------------
	fit_model_params = init_model_params(results_obj, config, sigma_opt)

	# -----------------------------------------------------
	# Load/create dict to store results
	# -----------------------------------------------------
	# Set result key based on what parameters we are estimating
	# and whether they are time varying ("_TV")

	def is_TV(v):
		if isinstance(v, dict):
			if isinstance(v.get('value', 1), list):
				if len(v.get('value', 1)) > 1:
					return True
		else:
			return False

	model_name = config.get('model_name', None)
	estimating = {k: v for k, v in fit_model_params.items() if isinstance(v, dict) and v.get('estimate', False)}
	estimating_str = ('+').join(sorted([f"{k}_TV" if is_TV(v) else k for k, v in estimating.items()]))

	result_key = f"{model_name}_{estimating_str}" if model_name else estimating_str

	if sigma_opt:
		result_key = f"sigma-opt_{result_key}"

	# Create/load results dict
	if not results_obj.results_dict.get(result_key, None):
		results_obj.results_dict[result_key] = {
			'fit_model_params': fit_model_params, 
			'hyper_param_values': hyper_param_values,
			'results_list': {},
			}
		results_obj.save()

	results_obj.wrangle_stragglers(result_key)

	# -----------------------------------------------------
	# Save fit model params
	# -----------------------------------------------------
	results_dir = results_obj.folder / result_key
	results_dir.mkdir(exist_ok=True, parents=True)

	(results_dir / "fit_model_params.json").write_text(json.dumps(fit_model_params, indent=4))

	# -----------------------------------------------------
	# Test hyperparameter combinations in parallel
	# -----------------------------------------------------
	hyper_param_combos = [dict(zip(hyper_param_values.keys(), values)) for values in itertools.product(*hyper_param_values.values())]
	hyperparam_args = [[results_obj, result_key, h_combo, fit_model_params, config['iterative_pE'], n_epochs, lr, graph, debug] for h_combo in hyper_param_combos]

	if sigma_opt:
		bm = fit_model_params["brownian_motion"]
		cv_idxs = [bm[str(i)]['idxs'] for i in range(bm["n_folds"])]
		results_obj.cv_idxs = cv_idxs

	# Each thread will test a hyperparameter combination on a given fold
	pool_args = []
	for hyperparam_arg_set in hyperparam_args:
		for fold in range(len(results_obj.cv_idxs)):
			new_arg_set = hyperparam_arg_set[:]
			new_arg_set[3:3] = [fold]
			pool_args.append(new_arg_set)

	if int(n_threads) > 0:
		print("Starting a pool")
		with Pool(int(n_threads)) as pool:
			results_list = pool.starmap(test_param_fold, pool_args)

	else:
		results_list = []
		for pool_arg in pool_args:
			results = test_param_fold(*pool_arg)
			results_list.append(results)

	# -----------------------------------------------------
	# Reload results object and save results of search, 
	# if we have any
	# -----------------------------------------------------
	results_obj = ResultsObj(analysis_dir)

	if any(results_list):
		results_df = pd.DataFrame(results_list)
		results_obj.update_from_folds_list(result_key, results_list)

	# -----------------------------------------------------
	# Plot hyperparameters then fit and validate on entire
	# training set using best combination
	# -----------------------------------------------------
	results_obj.wrangle_stragglers(result_key)
	results_obj.summarize_search(result_key)
	results_obj.plot_hyperparams(results_obj.folder / result_key)

	results_obj.do_validation(result_key, config["iterative_pE"], graph, debug)
	print("Did Validation")

	return results_obj

def resume_fits():
	# Load
	pass

@click.command()
@click.argument('model_config')
@click.option('--n_threads', default=8, type=int)
@click.option('--test', '-t', is_flag=True, default=False, help='Use to ensure setup works. Sets n_epochs to 3 and model name to "test"')
@click.option('--debug', '-d', is_flag=True, default=False, help='Use to debug anything done in parallel or TensorFlow. Sets n_threads to 0 to remove parallelization and changes graph execution to false')
@click.option('--graph', '-g', is_flag=True)
def main_func(model_config, lr=1e-05, n_epochs=20000, reg_type=["l1"], lamb=[0], sigma=[0], sigma_opt=False, n_threads=8, test=False, debug=False, graph=False):
	config = load(Path("config.yaml").read_text(), Loader=Loader)
	model_config = load(Path(f"configs/config_model_params_{model_config}.yaml").read_text(), Loader=Loader)
	config.update(model_config)

	if test:
		config["n_epochs"] = 100
		config["model_name"] = "test"

	analysis_dir = "data_new/analysis/three_sampling_intervals"

	RO = crossvalidate(analysis_dir, hyper_param_values=config["hyper_param_values"], config=config, debug=debug, n_epochs=config["n_epochs"], lr=config["lr"], n_threads=n_threads, graph=graph)

if __name__ == "__main__":
	main_func()
