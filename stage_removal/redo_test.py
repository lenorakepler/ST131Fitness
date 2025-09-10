import pandas as pd
import numpy as np
from pathlib import Path
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
from ecoli_analysis.results_obj import ResultsObj
from multiprocessing import freeze_support, set_start_method, Pool
from collections import ChainMap
import json
import copy

def redo_single_test(h_name, fold, RO, test_data, test_fit_model_params, iterative_pE, debug):
	_, test_loss, test_opt = RO.fit_score(
		data=test_data, iterative_pE=iterative_pE, 
		reg_type=None, lamb=0, sigma=False, n_epochs=1, lr=0.1, 
		graph=False, return_opt=True, offset=1, 
		verbose=False, debug=debug, **test_fit_model_params
		)

	# Update JSON
	json_file = RO.folder / result_key / f"{h_name}_fold-{fold}.json"
	fold_dict = json.loads(json_file.read_text())
	fold_dict["test_loss"] = float(test_loss)
	json_file.write_text(json.dumps(fold_dict, indent=4))

	return {"h_name": h_name, "fold": fold, "test_loss": test_loss}

def redo_test(analysis_dir, result_key, n_threads, debug):
	RO = ResultsObj(analysis_dir)

	brown_info = json.loads((RO.folder / "brownian_fit_setup.json").read_text())

	# Update this because changed how do
	RO.results_dict[result_key]["fit_model_params"]["brownian_motion"].update(brown_info)

	rd = RO.results_dict[result_key]
	fit_model_params = rd["fit_model_params"]

	pool_args = []
	for h_name, h_dict in rd["results_list"].items():
		for i, fold_dict in enumerate(h_dict["fold_estimates"]):

			test_data = RO.loadDataByIdx(RO.cv_idxs[i][1])
			fold_model_params = copy.deepcopy(fit_model_params)
			fold_model_params["brownian_motion"]["n_types"] = fold_model_params["brownian_motion"][str(i)]["n_types"]
			fold_model_params["brownian_motion"]["info"] = fold_model_params["brownian_motion"][str(i)]["test"]

			for k, v in fold_dict.items():
				fold_model_params[k]["value"] = v
			
			pool_args.append([h_name, i, RO, test_data, fold_model_params, h_dict["iterative_pE"], debug])

	if int(n_threads) > 0:
		print("Starting a pool")
		with Pool(int(n_threads)) as pool:
			results_list = pool.starmap(redo_single_test, pool_args)

	else:
		results_list = []
		for pool_arg in pool_args:
			results = redo_single_test(*pool_arg)
			results_list.append(results)
			print(results)

	df = pd.DataFrame(results_list)
	for h_name, hdf in pd.DataFrame(results_list).groupby("h_name"):
		RO.results_dict[result_key]["results_list"][h_name]["fold_test_losses"] = [float(l) for l in hdf["test_loss"]]
		RO.results_dict[result_key]["results_list"][h_name]["mean_test_loss"] = float(hdf['test_loss'].mean())

	RO.save()

if __name__ == "__main__":
	analysis_dir = "data_new/analysis/three_sampling_intervals"
	result_key = "new_model_brownian_motion"

	# redo_test(analysis_dir, result_key, n_threads=0, debug=False)
	RO = ResultsObj(analysis_dir)
	RO.plot_hyperparams(RO.folder / result_key)