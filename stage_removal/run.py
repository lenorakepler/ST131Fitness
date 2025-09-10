from yaml import CDumper as Dumper, CLoader as Loader, load, dump
from pathlib import Path
from ecoli_analysis.results_obj import ResultsObj
from _analysis.test_parallel import crossvalidate
import click
from memory_profiler import profile

# @click.command()
# @click.option('--lr', default=1e-05)
# @click.option('--n_epochs', default=20000)
# @click.option('--model_config', default="config_model_params_with_branch")
# @click.option('--reg_type', default=["l1"], multiple=True)
# @click.option('--lamb', default=[1, 5, 10, 15], multiple=True, type=float)
# @click.option('--sigma', default=[0.5], multiple=True, type=float)
# @click.option('--n_threads', default=8, type=int)
# @click.option('--test', default=False)
# @click.option('--debug', default=False)
def main_func(lr=1e-05, n_epochs=20000, model_config="with_branch", reg_type=["l1"], lamb=[1, 5, 10, 15], sigma=[0.5, 2], n_threads=8, test=False, debug=False):
	config = load(Path("config.yaml").read_text(), Loader=Loader)
	model_config = load(Path(f"config_model_params_{model_config}.yaml").read_text(), Loader=Loader)
	config.update(model_config)

	config["n_epochs"] = n_epochs
	config["lr"] = lr
	config["hyper_param_values"] = config.get("hyper_param_values", {})
	config["hyper_param_values"]["lamb"] = lamb
	config["hyper_param_values"]["sigma"] = sigma
	config["hyper_param_values"]["reg_type"] = reg_type

	RO = ResultsObj(folder=Path(config["data_dir"]) / "analysis" / config["analysis_name"])

	if not RO.success["index"]:
		RO.set_data(
			tree_file=Path(config["data_dir"]) / config["interval_tree_name"] / "phylo.nwk",
			interval_times_file=Path(config["data_dir"]) / config["interval_tree_name"] / "interval_times.txt",
			last_sample_date=config["last_sample_date"]
		)
		RO.set_folds(test_size=0.2, n_splits=4, stratify=None, shuffle=True, random_state=8)

	if test:
		config["n_epochs"] = 3
		config["model_name"] = "test"

	crossvalidate(results_obj=RO, hyper_param_values=config["hyper_param_values"], config=config, debug=debug, n_epochs=config["n_epochs"], lr=config["lr"], n_threads=n_threads)
	RO.summarize_searches()

if __name__ == "__main__":
	main_func(n_epochs=100, model_config="sigmaonly", sigma=[1.05], lamb=[0], n_threads=0)