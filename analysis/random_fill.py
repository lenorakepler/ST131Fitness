from pathlib import Path
import click
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
from analysis.do_model_fit import ResultsObj, crossvalidate
from analysis.analyze_fit import lj
import json

@click.command()
@click.argument('task')
@click.argument('model_config_name')
@click.option('--n_threads', default=8, type=int)
@click.option('--test', '-t', is_flag=True, default=False, help='Use to ensure setup works. Sets n_epochs to 3 and model name to "test"')
@click.option('--debug', '-d', is_flag=True, default=False, help='Use to debug anything done in parallel or TensorFlow. Sets n_threads to 0 to remove parallelization and changes graph execution to false')
@click.option('--graph', '-g', is_flag=True)
def main_func(task, model_config_name, lr=1e-05, n_epochs=20000, reg_type=["l1"], lamb=[0], sigma=[0], sigma_opt=False, n_threads=8, test=False, debug=False, graph=False):
	config = load(Path("config.yaml").read_text(), Loader=Loader)
	model_config = load(Path(f"configs/config_model_params_{model_config_name}.yaml").read_text(), Loader=Loader)
	
	analysis_dir = "data_new/analysis/three_sampling_intervals"

	if task == "crossvalidate":
		results_obj = ResultsObj(analysis_dir)

		# Add "full" info to brownian motion fit params.. this is so stupid
		full_info = results_obj.results_dict[model_config["effects_model"]]["fit_model_params"]["brownian_motion"]["full"]

		results_obj.results_dict["sigma-opt_random-only_brownian_motion"]["fit_model_params"]["brownian_motion"]["full"] = full_info

		results_obj.save()

		model_config["fit_model_params"]["brownian_motion"]["full"] = full_info

		for var, estimate in results_obj.results_dict[model_config["effects_model"]]["full"]["estimates"].items():
			model_config["fit_model_params"][var]["value"] = estimate
			assert model_config["fit_model_params"][var]["estimate"] == False

		Path(f"configs/config_model_params_{model_config_name}_with-estimates.yaml").write_text(dump(model_config, Dumper=Dumper))

		config.update(model_config)

		del results_obj

		if test:
			config["n_epochs"] = 100
			config["model_name"] = "test"

		RO = crossvalidate(analysis_dir, hyper_param_values=config["hyper_param_values"], config=config, debug=debug, n_epochs=config["n_epochs"], lr=config["lr"], n_threads=n_threads, graph=graph)

	elif task == "concat":
		RO = ResultsObj(analysis_dir)
		fit_params = RO.results_dict[model_config["effects_model"]]["fit_model_params"]

def concat(genetic_model, random_model):

	analysis_dir = Path("data_new/analysis/three_sampling_intervals")
	RO = ResultsObj(analysis_dir)

	vg = lj(analysis_dir / genetic_model / "validation.json")
	vr = lj(analysis_dir / random_model / "validation.json")

	fmpeg = lj(analysis_dir / genetic_model / "estimated_fit_model_params.json")
	fmper = lj(analysis_dir / random_model / "estimated_fit_model_params.json")

	concat_validation = {
		"h_combo": {'reg_type': vg["h_combo"]["reg_type"], 'lamb': vg["h_combo"]["lamb"], 'sigma': vr["h_combo"]["sigma"]},
		"lr": vg["lr"],
		"iterative_pE": vg["iterative_pE"],
		"estimates": {**vg["estimates"], **vr["estimates"]},
		"named_estimates": {**vg["named_estimates"], **vr["named_estimates"]},
		}
	
	concat_estimated_fit_model_params = {}

	for var, var_dict in fmpeg.items():
		if isinstance(var_dict, dict) and (var_dict["estimate"] == False) and (fmper[var]["estimate"]):
			concat_estimated_fit_model_params[var] = fmper[var]
		else:
			concat_estimated_fit_model_params[var] = var_dict

	new_dir = analysis_dir / f"{genetic_model}+{random_model}"
	(new_dir / "figures").mkdir(exist_ok = True, parents = True)

	(new_dir / "validation.json").write_text(json.dumps(concat_validation))
	(new_dir / "estimated_fit_model_params.json").write_text(json.dumps(concat_estimated_fit_model_params))

	print(fmpeg.keys())

def add_analysis_info():
	analysis_dir = Path("data_new/analysis/three_sampling_intervals")
	RO = ResultsObj(analysis_dir)

	for rdir in analysis_dir.iterdir():
		if rdir.is_dir():
			key = rdir.name
			fmp = RO.results_dict[key]["fit_model_params"]

			(rdir / "fit_model_params.json").write_text(json.dumps(fmp))

			if (val_file := rdir / "validation.json").exists():
				val = lj(val_file)
				 
				for var, estimate in val["estimates"].items():
					fmp[var]["value"] = estimate
				
				(rdir / "estimated_fit_model_params.json").write_text(json.dumps(fmp))

if __name__ == "__main__":
	# main_func()
	concat("no_random_birth_background_TV+birth_features+sampling_background_TV+sampling_features", "sigma-opt_random-only_brownian_motion")
	# add_analysis_info()
