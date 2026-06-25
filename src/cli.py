from pathlib import Path
import click
from click.core import ParameterSource
from yaml import CDumper as Dumper, CLoader as Loader, load, dump

module_path = Path(__file__).parent.parent

# @dataclass
# class config:
# 	model_name: "full_model_tvbs"
# sigma_opt: False

# # -----------------------------------------------------
# # BIRTH-DEATH-SAMPLING MODEL PARAMETERS
# # -----------------------------------------------------
# fit_model_params:
#   # Not estimated
#   death_rate: 1
  
#   # Potentially estimated
#   birth_background:
#     estimate: True
#     penalize: False
#     value: [1.000001, 1.000001, 1.000001]
#     changepoints: [2003, 2013]
#   birth_features:
#     estimate: True
#     penalize: True
#     states: "../data/model_input/marginal_features_genetic.csv" # Binary features file for each sample + ancestral node
#     value: 1
#   sampling_features:
#     variables:
#         bioproject: 
#           type: "multi"
#           states: "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/meta_features_bp/marginal_states.csv"
#           mask: "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/model_input/sampling_mask.csv"
#           value: 1
#           estimate: True
#           penalize: True
#         specimen_type: 
#           type: "binary"
#           states: "/Users/lenorakepler/Dropbox/NCSU/Lab/iMac/ST131Fitness Full April 2025/data/meta_features_specimen/marginal_states.csv"
#           value: 1
#           estimate: True
#           penalize: True
#   sampling_background:
#     estimate: True
#     penalize: False
#     value: [0.0001, 0.0001, 0.0001]
#     changepoints: [2003, 2013]
#   brownian_motion:
#     estimate: True
#     penalize: False
#     value: 1
#     states: "../results/three_sampling_intervals/brownian_fit_setup.json"

def read_yaml(yaml_file):
	return load(Path(yaml_file).resolve().read_text(), Loader=Loader)

# def load_config(yaml_file):

decoration_length = 60
start_stop = '=' * decoration_length

@click.group()
@click.pass_context
@click.option("--decoration", default=True, help="Whether to display decorative ASCII command line elements")
def cli(ctx, decoration):
	print("Opened CLI")
	ctx.obj = {} # context object that gets passed to child commands
	ctx.obj['decoration'] = decoration

	if decoration:
		click.echo(start_stop)

@cli.command()
@click.option("--arg", default="3")
@click.pass_context
def prep_data(ctx, arg):
	"""
	Prep data for model fitting
	"""
	click.echo(arg)

@cli.result_callback()
def end_decorator(result, decoration):
	if decoration:
		click.echo(start_stop + "\n")

# TODO: add default config files used to get published results
@cli.command()
@click.pass_context
@click.option("--data", default=module_path / "configs/config.yaml", type=click.Path(exists=True), help="Path to config specifying data parameters")
@click.option("--model", default=module_path / "configs/config_model_params_test-full-model-tvbs.yaml", type=click.Path(exists=True), help="Path to config specifying fitness model parameters")
@click.option("--opt", default=module_path / "configs/opt_params.yaml", type=click.Path(exists=True), help="Path to config specifying optimization parameters")
@click.option('--n_threads', default=8, type=int)
@click.option('--test', '-t', is_flag=True, default=False, help='Use to ensure setup works. Sets n_epochs to 3 and model name to "test"')
@click.option('--debug', '-d', is_flag=True, default=False, help='To allow debugging, sets n_threads to 0 to remove parallelization and turns on eager execution')
@click.option('--eager', '-g', is_flag=True, default=False, help="Put TensorFlow into eager mode. Use if you need to debug and get tensor values or if running for very few epochs and upfront graph pre-computation is too slow")
@click.option('--interactive', '-i', is_flag=True, default=False, help="Drop into IDE after crossvalidation so you can interact with the results object")
def fit_model(ctx, data, model, opt, n_threads, test, debug, eager, interactive):
	from model_fit.do_model_fit import crossvalidate

	decoration = ctx.obj['decoration']
	if decoration:
		click.echo("Fitting model:\n--------------")

	if click.get_current_context().get_parameter_source('data') == ParameterSource.DEFAULT:
		print(f'No data config file specified, defaulting to {data}')

	if click.get_current_context().get_parameter_source('model') == ParameterSource.DEFAULT:
		print(f'No model config file specified, defaulting to {model}')

	if click.get_current_context().get_parameter_source('opt') == ParameterSource.DEFAULT:
		print(f'No opt config file specified, defaulting to {opt}')

	try:
		config = read_yaml(data)
		model_config = read_yaml(model)
		opt_config = read_yaml(opt)

	except Exception as e:
		click.echo(e)

	# TODO: BUG: model_config (specified on command line) is not the same as model_name (specified in configs/config_model_params_<model_config>.yaml)
	config.update(model_config)
	config.update(opt_config)

	if test:
		config["n_epochs"] = 3
		config["model_name"] = "test0"

	# I changed the flag to eager in the CLI because I thought it was easier to understand,
	# but I'm leaving graph everywhere else. Here, graph mode is just 'not eager'.
	graph = not eager

	# Do crossvalidation with
	RO = crossvalidate(
		analysis_dir=Path(config["analysis_dir"]).resolve(),
		hyper_param_values={k: config[k] for k in ["lamb", "sigma", "reg_type"]},
		config=config, 
		debug=debug, 
		n_epochs=config["n_epochs"], 
		lr=config["lr"], 
		n_threads=n_threads, 
		graph=graph)

@cli.command()
def analyze_results():
	pass

if __name__ == "__main__":
	print(Path(__file__).parent.parent.resolve())
	cli(["fit-model", "--debug", "--test"])