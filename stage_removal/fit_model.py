from ecoli_analysis.results_obj import load_data_and_RO_from_file, load_data_and_RO
import numpy as np
import pandas as pd
import click
from yte import process_yaml
from pathlib import Path

def fit_model(analysis_dir, analysis_name, interval_tree_file, features_file, config):
	analysis_dir = Path(analysis_dir)

	if not isinstance(config, dict):
		try:
			config = process_yaml(Path(config).read_text())
		except Exception as e:
			print(e)
			print("Could not load config")

	# Load or create results object (RO) + formatted data
	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	if (Path(analysis_dir) / "params.json").exists():
		data, phylo_obj, RO, params = load_data_and_RO_from_file(analysis_dir)
	else:																											
		data, phylo_obj, RO, params = load_data_and_RO(
			analysis_dir.parent,
			analysis_name,
			interval_tree_file,
			features_file,
			config['bd_array_params'], 
			config['bioproject_times_file'],
			config["last_sample_date"],
			config['constrained_sampling_rate'], 
			config['birth_rate_changepoints'], 
			config['n_epochs']
			)

	# Do the cross-validation
	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	bdm_params = config["bdm_params"]

	RO.crossvalidate(
		bdm_params, 
		config["hyper_param_values"],
		analysis_dir, phylo_obj.feature_names, 
		config["n_epochs"], 
		config["lr"]
		)

	# Format and save the results of cross-validation
	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	estimating = {k: v for k, v in bdm_params.items() if v[0] == True}
	result_key = ('+').join(sorted([f"{k}_TV" if (len(v) > 1 and v[1] == True) else k for k, v in estimating.items()]))
	results = RO.results_dict[result_key]["full"]

	site_names = phylo_obj.features_df.columns
	estimate_results = []

	if bdm_params['b0'][0]:
		b0_estimates = results['estimates']['b0']

		changepoints = [phylo_obj.root_time] + config["birth_rate_changepoints"]
		interval_strings = []
		for i, time in enumerate(changepoints):
			if i == len(changepoints) - 1:
				t = time
				t1 = phylo_obj.present_time
			else:
				t = time
				t1 = changepoints[i + 1]

			interval_strings.append(f"Interval_{t:.0f}-{t1:.0f}")

		if isinstance(b0_estimates, np.ndarray) or isinstance(b0_estimates, list):
			estimate_results += [[interval, e] for interval, e in zip(interval_strings, b0_estimates)]

		else:
			estimate_results += [[f"b0", b0_estimates]]

	if bdm_params['site'][0]:
		site_estimates = results['estimates']['site'][0]
		for e, t in zip(site_names, site_estimates):
			estimate_results += [[e, t]]

	df = pd.DataFrame(estimate_results, columns=["feature", "estimate"])
	df.to_csv(analysis_dir / "estimates.csv")

	print("===================================================================")
	print(f"Best hyperparameters: {results['h_combo']}")
	print(f"Train loss: {results['train_loss']}, Test loss: {results['test_loss']}")
	print("===================================================================")

@click.command()
@click.argument('analysis_dir')
@click.argument('analysis_name')
@click.argument('interval_tree_file')
@click.argument('features_file')
@click.argument('config')
def main(analysis_dir, analysis_name, interval_tree_file, features_file, config):
	fit_model(analysis_dir, analysis_name, interval_tree_file, features_file, config)

if __name__ == "__main__":
	main()
