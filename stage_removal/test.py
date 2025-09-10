from pathlib import Path
from yte import process_yaml
import pandas as pd
import numpy as np
from model_fit.results_obj import ResultsObj
from yaml import CDumper as Dumper, CLoader as Loader, load, dump

config = load(Path("config.yaml").read_text(), Loader=Loader)
model_config = load(Path("config_model_params_b.yaml").read_text(), Loader=Loader)
config.update(model_config)

RO = ResultsObj(folder=Path(config["data_dir"]) / "analysis" / config["analysis_name"])

# -----------------------------------------------------
# Set up additional fitness model parameters
# -----------------------------------------------------
branch_names = RO.data.array['name']
branch_dict = {n: i for i, n in enumerate(list(set(branch_names)))}
n_branches = len(branch_dict)

fit_model_params = config["fit_model_params"]

s_bg_changepoints = fit_model_params["sampling_background"]["changepoints"]
if (root_time := RO.data.root_time) not in s_bg_changepoints:
	s_bg_changepoints = [root_time] + s_bg_changepoints

fit_model_params["branch_effects"]["branch_dict"] = branch_dict
fit_model_params["branch_effects"]["n_branches"] = len(branch_dict)

fit_model_params["sampling_background"]["changepoints"] = s_bg_changepoints
fit_model_params["sampling_background"]["interval_mapping"] = [int(np.where(time > s_bg_changepoints)[0][-1]) if time != s_bg_changepoints[0] else 0 for time in RO.data.param_interval_times]

b_bg_changepoints = fit_model_params["birth_background"]["changepoints"]
if (root_time := RO.data.root_time) not in b_bg_changepoints:
	b_bg_changepoints = [root_time] + b_bg_changepoints

fit_model_params["birth_background"]["changepoints"] = b_bg_changepoints
fit_model_params["birth_background"]["interval_mapping"] = [int(np.where(time > b_bg_changepoints)[0][-1]) if time != b_bg_changepoints[0] else 0 for time in RO.data.param_interval_times]


train = RO.loadDataByIdx(RO.train_idx)
estimates, loss, opt = RO.fit_score(
	data=train,
	iterative_pE=True, reg_type="l1", lamb=25, n_epochs=5, lr=0.00005, graph=False,
	return_opt=True, offset=1, verbose=True, debug=False, **fit_model_params
	)


# # Snakemake should be able to handle this, but it doesn't seem to be working
# config = process_yaml(Path("config_test.yaml").read_text())

# analysis_dir = Path("data/analysis/3-interval_constrained-sampling")
# data, phylo_obj, RO, params = load_data_and_RO_from_file(analysis_dir)


# # Do cross-validation to determine optimal sigma hyperparameter
# do_crossval(Path(analysis_dir), residual_name, 
# 	n_sigmas=config["n_sigmas"], sigma_start=config["sigma_start"], sigma_stop=config["sigma_stop"],
# 	est_site={'b0': b0, 'site': site}, 
# 	n_epochs=config["n_epochs"], lr=config["lr"])

# # Using optimal sigma, compute residual branch fitness
# analyze_fit(Path(analysis_dir), residual_name, est_site={'b0': b0, 'site': site}, n_epochs=config["n_epochs"], lr=config["lr"])

# calc_fitness_totals(analysis_dir, residual_dir)

# do_decomp(Path(analysis_dir), residual_name, total=True, interval_length=interval_length, interval_cutoff=interval_cutoff)
# do_decomp(Path(analysis_dir), residual_name, total=False, interval_length=interval_length, interval_cutoff=interval_cutoff)

# from ecoli_analysis.RO import load_data_and_RO_from_file, load_data_and_RO
# from analysis.likelihood_profile import make_profiles, get_CIs

# estimating = {k: v for k, v in config["bdm_params"].items() if v[0] == True}
# result_key = ('+').join(sorted([f"{k}_TV" if (len(v) > 1 and v[1] == True) else k for k, v in estimating.items()]))


# train = RO.loadDataByIdx(RO.train_idx)

# estimates = RO.results_dict[result_key]["full"]["estimates"]
# bdm_params = RO.results_dict[result_key]["bdm_params"]
# h_combo = RO.results_dict[result_key]["full"]["h_combo"]

# make_profiles(
# 	train, 
# 	estimates, 
# 	params["features_file"], 
# 	bdm_params, 
# 	h_combo, 
# 	params["birth_rate_idx"], 
# 	Path(analysis_dir), 
# 	Path(figures_dir),
# 	plot_effect_profiles=True
# )

# get_CIs(Path(analysis_dir))