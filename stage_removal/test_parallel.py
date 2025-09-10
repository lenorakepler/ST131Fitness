import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
from ecoli_analysis.results_obj import ResultsObj
from multiprocessing import freeze_support, set_start_method, Pool
from collections import ChainMap
import json
import itertools
import click
import matplotlib.pyplot as plt

def test_param_fold(results_obj, result_key, h_combo, fold, fit_model_params, iterative_pE, n_epochs, lr, graph, debug):
	combo_key = "_".join(sorted([f"{k}={v}" for k, v in h_combo.items()])) + f"_lr={lr}_n_epochs={n_epochs}"
	print(f"\nTesting model {result_key} using hyperparameters {combo_key} ---> fold {fold}")

	results_dir = results_obj.folder / result_key
	result = (results_dir / f"{combo_key}_fold-{fold}.json")
	results_dict = {"combo_key": combo_key, "fold": fold, "h_combo": h_combo}
	fig_dir = results_dir / "figures" / "train_losses"
	fig_dir.mkdir(exist_ok=True, parents=True)

	# ---------------------------------------------------------------------
	# If this hyperparameter set has not already been tested on this fold
	# ---------------------------------------------------------------------
	if not result.exists():
		results_dir.mkdir(exist_ok=True, parents=True)

		# Get data for training and testing
		cv_train_idx, cv_test_idx = results_obj.cv_idxs[fold]
		cv_train = results_obj.loadDataByIdx(cv_train_idx)
		cv_test = results_obj.loadDataByIdx(cv_test_idx)

		# Set brownian motion info for this fold
		fit_model_params["brownian_motion"]["info"] = fit_model_params["brownian_motion"][str(fold)]

		estimates, train_loss, test_loss, train_opt, test_opt = results_obj.do_train_test(
			fit_model_params, cv_train, cv_test, h_combo['reg_type'],
			h_combo['lamb'], h_combo['sigma'], iterative_pE, n_epochs, lr, graph, 
			offset=1, verbose=True, debug=debug,
		)

		results_obj.plot_epoch_losses(train_opt.losses, fig_dir, filename_base=f"{combo_key}_fold-{fold}_")

		results_dict["test_loss"] = float(test_loss)
		results_dict["train_loss"] = float(train_loss)
		results_dict["estimates"] = {variable: e.tolist() for variable, e in estimates.items()}
		
		# print(f"Fold {i}: Train loss={train_loss:.3f}, Test loss={test_loss:.3f}")
		# print(f"Estimate={estimates}")

		folds_dict_out = json.dumps(results_dict, indent=4)
		(results_dir / f"{combo_key}_fold-{fold}.json").write_text(folds_dict_out)

		return results_dict

	else:
		print(f"Already tested {combo_key} on fold {fold}, moving on")
		return {}

def init_model_params(results_obj, config, sigma_opt):
	# -----------------------------------------------------
	# Init fitness model parameters
	# -----------------------------------------------------
	branch_names = results_obj.data.array['name']
	branch_dict = {n: i for i, n in enumerate(list(set(branch_names)))}
	n_branches = len(branch_dict)

	fit_model_params = config["fit_model_params"]

	s_bg_changepoints = fit_model_params["sampling_background"]["changepoints"]
	if (root_time := results_obj.data.root_time) not in s_bg_changepoints:
		s_bg_changepoints = [root_time] + s_bg_changepoints

	fit_model_params["branch_effects"]["branch_dict"] = branch_dict
	fit_model_params["branch_effects"]["n_branches"] = len(branch_dict)

	fit_model_params["sampling_background"]["changepoints"] = s_bg_changepoints
	fit_model_params["sampling_background"]["interval_mapping"] = [int(np.where(time > s_bg_changepoints)[0][-1]) if time != s_bg_changepoints[0] else 0 for time in results_obj.data.param_interval_times]

	b_bg_changepoints = fit_model_params["birth_background"]["changepoints"]
	if (root_time := results_obj.data.root_time) not in b_bg_changepoints:
		b_bg_changepoints = [root_time] + b_bg_changepoints

	fit_model_params["birth_background"]["changepoints"] = b_bg_changepoints
	fit_model_params["birth_background"]["interval_mapping"] = [int(np.where(time > b_bg_changepoints)[0][-1]) if time != b_bg_changepoints[0] else 0 for time in results_obj.data.param_interval_times]

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

	results_obj.wrangle_stragglers(result_key)

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
		for combo_key, cdf in results_df.groupby('combo_key'):
			results_obj.results_dict[result_key]["results_list"].update(
				{
				combo_key: 
					dict(
						h_combo = cdf.iloc[0]['h_combo'],
						lr=lr,
						n_epochs=n_epochs,
						iterative_pE=config["iterative_pE"],
						mean_train_loss = cdf['train_loss'].mean(),
						mean_test_loss = cdf['test_loss'].mean(),
						fold_estimates = cdf['estimates'].to_list(),
						fold_train_losses = cdf['train_loss'].to_list(),
						fold_test_losses = cdf['test_loss'].to_list(),
						)
				}
				)
		results_obj.save()

	# -----------------------------------------------------
	# Plot hyperparameters then fit and validate on entire
	# training set using best combination
	# -----------------------------------------------------
	results_obj.wrangle_stragglers(result_key)
	results_obj.summarize_search(result_key)
	results_obj.plot_hyperparams(results_obj.folder / result_key)

	if not sigma_opt:
		results_obj.do_validation(result_key, config["iterative_pE"], graph, debug)

	return results_obj

@click.command()
@click.argument('model_config')
@click.option('--n_threads', default=8, type=int)
@click.option('--test', '-t', is_flag=True, default=False, help='Use to ensure setup works. Sets n_epochs to 3 and model name to "test"')
@click.option('--debug', '-d', is_flag=True, default=False, help='Use to debug anything done in parallel or TensorFlow. Sets n_threads to 0 to remove parallelization and changes graph execution to false')
def main_func(model_config, lr=1e-05, n_epochs=20000, reg_type=["l1"], lamb=[0], sigma=[0], sigma_opt=False, n_threads=8, test=False, debug=False):
	config = load(Path("config.yaml").read_text(), Loader=Loader)
	model_config = load(Path(f"config_model_params_{model_config}.yaml").read_text(), Loader=Loader)
	config.update(model_config)

	if test:
		config["n_epochs"] = 3
		config["model_name"] = "test"

	analysis_dir = "data_new/analysis/three_sampling_intervals"

	RO = crossvalidate(analysis_dir, hyper_param_values=config["hyper_param_values"], config=config, debug=debug, n_epochs=config["n_epochs"], lr=config["lr"], n_threads=n_threads)
	RO.summarize_searches()

if __name__ == "__main__":
	main_func()
