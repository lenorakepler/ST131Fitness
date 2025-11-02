import json
import itertools
from pathlib import Path
from multiprocessing import freeze_support, set_start_method, Pool
import click
import numpy as np
import pandas as pd
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
from model_fit.results_obj import ResultsObj

# https://stackoverflow.com/questions/69924881/userwarning-starting-a-matplotlib-gui-outside-of-the-main-thread-will-likely-fa
import matplotlib
matplotlib.use('agg')

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
		results_obj.set_folds(test_size=0.2, n_splits=4, shuffle=True, random_state=8)

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
def main_func(model_config, n_threads=8, test=False, debug=False, graph=False):
	config = load(Path("config.yaml").read_text(), Loader=Loader)
	model_config = load(Path(f"configs/config_model_params_{model_config}.yaml").read_text(), Loader=Loader)

	# TODO: BUG: model_config (specified on command line) is not the same as model_name (specified in configs/config_model_params_<model_config>.yaml)
	config.update(model_config)

	if test:
		config["n_epochs"] = 100
		config["model_name"] = "test"

	analysis_dir = "data_new/analysis/three_sampling_intervals"

	RO = crossvalidate(analysis_dir, hyper_param_values=config["hyper_param_values"], config=config, debug=debug, n_epochs=config["n_epochs"], lr=config["lr"], n_threads=n_threads, graph=graph)

if __name__ == "__main__":
	main_func()
