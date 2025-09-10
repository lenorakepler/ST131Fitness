import json
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.optimizer import Optimizer
from ecoli_analysis.random_effects import find_parents, split_intervals, get_parent_type_info
from _analysis.test_parallel import test_param_fold, init_model_params
from ecoli_analysis.results_obj import ResultsObj
from natsort import natsorted, ns

# from ecoli_analysis.random_effects_classes import RandomEffectSite
import analysis.plot_phylo_standalone as pp

def phylo_plot_in_train_test(tree_file, present_time, train_data, folds, out_file):
	# Load tree
	tt = pp.loadTree(
		tree_file,
		internal=True,
		abs_time=present_time
	)

	n_folds = len(folds)
	fig, axs = plt.subplots(1, n_folds, figsize=(12 * n_folds, 25))
	axs = axs.ravel()

	for fold_num, fold_dict in folds.items():
		test_start = fold_dict['params']['test']['start_time']
		test_end = fold_dict['params']['test']['end_time']

		fold_trait = {
			**{n['name']: "Train" for n in fold_dict['train'].values()},
			**{n['name']: "Test" for n in fold_dict['test'].values()},
		}

		colors, c_func = pp.categoricalFunc(fold_trait, 'name', legend=True, null_color="red")

		axs[fold_num] = pp.plotTraitAx(
			axs[fold_num],
			tt,
			edge_c_func=c_func,
			node_c_func=c_func,
			s_func=lambda x: 4,
			tip_names=False,
			zoom=False,
			title=f"Fold {fold_num}",
		)
		axs[fold_num].axvline(x=test_start, linestyle="--", color="red")
		axs[fold_num].axvline(x=test_end, linestyle="--", color="blue")

	pp.add_legend(colors, axs[fold_num], lloc="lower left")

	plt.tight_layout()
	plt.savefig(out_file, dpi=300)
	plt.close("all")

def prep_data_for_hyperparam_search(analysis_dir, n_folds=3, test_proportion=(1/2), folds_start=1960, plot=False, alt=False):
	"""
	Adds "parent_idx" to 
	"""

	RO = ResultsObj(folder=analysis_dir)
	data = RO.data

	# -----------------------------------------------------
	# Load / make time folds info
	# -----------------------------------------------------
	split_intervals(
		all_data=data, 
		train_idx=RO.train_idx, 
		out_folder=RO.folder,
		n_folds=n_folds,
		test_proportion=test_proportion, 
		root_time=data.root_time, 
		present_time=data.present_time, 
		folds_start=folds_start,
		alt=alt,
		)

	fname = "brownian_search_setup.json"
	if alt:
		fname = fname.replace(".json", "_alt.json")

	fold_params = json.loads((RO.folder / fname).read_text())
	folds = {int(i): v for i, v in fold_params['folds'].items()}

	if plot:
		phylo_plot_in_train_test(RO.params['tree_file'], data.present_time, data, folds, RO.folder / "brownian_search_setup.png")

	return RO, folds

def prep_data_for_fitting(analysis_dir, plot=True, alt=False):
	RO = ResultsObj(folder=analysis_dir)
	out_folder = RO.folder
	data = RO.data
	
	all_arr = pd.DataFrame(data.array)
	all_arr['index'] = list(range(len(all_arr)))

	if alt:
		all_arr["branch_name"] = all_arr["name"]
	else:
		all_arr["branch_name"] = all_arr["name"].apply(lambda n: n.split("_interval")[0])
	
	all_branch_names = natsorted(all_arr["branch_name"].unique(), alg=ns.GROUPLETTERS)

	brownian_info_dict = dict(
		n_types=int(len(all_branch_names)),
		names=all_branch_names,
		int_to_name={i: name for i, name in enumerate(all_branch_names)},
		)

	brownian_info_dict["full"] = get_parent_type_info(all_arr, RO.train_idx, RO.validate_idx, all_branch_names)
	for i, [train_idxs, test_idxs] in enumerate(RO.cv_idxs):
		brownian_info_dict[i] = get_parent_type_info(all_arr, train_idxs, test_idxs, all_branch_names)

	fname = "brownian_fit_setup.json"
	if alt:
		fname = fname.replace(".json", "_alt.json")

	(RO.folder / fname).write_text(json.dumps(brownian_info_dict, indent=4))

	if plot:
		plot_fname = fname.replace(".json", ".png")
		plot_dict = {i: {'params': {'test': {'start_time': data.root_time, 'end_time': data.present_time}}, **fold_dict} for i, fold_dict in brownian_info_dict.items() if isinstance(i, int)}
		phylo_plot_in_train_test(RO.params['tree_file'], data.present_time, data, plot_dict, RO.folder / plot_fname)
	
def do_crossval(analysis_dir, random_name, n_sigmas, sigma_start, sigma_stop, est_site=False, n_epochs=50000, lr=0.00005):
	# -----------------------------------------------------
	# Create and/or load fold-segmented tree file 
	# as data object
	# -----------------------------------------------------
	data, phylo_obj, fold_params, folds, out_folder, analysis_params = load_data(analysis_dir, random_name)

	json_out = out_folder / "results.json"

	# -----------------------------------------------------
	# Do sigma hyperparameter optimization
	# -----------------------------------------------------
	sigmas = np.linspace(sigma_start, sigma_stop, n_sigmas)
	results = {}
	for sigma in sigmas:
		results[sigma] = {'train': [], 'test': [], 'effs': []}

		for i, fold_dict in folds.items():
			print(f"\nsigma={sigma}: fold {i}")

			fit_model_estimates=dict(
					random_effect=[True, False],
					b0=[True, True],
					d=[False],
					s=[False],
					rho=[False],
					gamma=[False],
				)

			if est_site:
				fit_model_estimates['site'] = [True, False]
				fit_model_estimates['b0'] = [True, True]
	
			# DO TRAIN
			# -----------------------------------------------------
			opt = Optimizer(
				fit_model=RandomEffectSite,
				fit_model_kwargs={
					**fit_model_estimates,
					'loss_kwargs': {'reg_type': 'sigma', 'sigma': tf.constant(sigma, shape=[], dtype=tf.dtypes.float64)},
					'data': fold_dict['train']['data'].returnCopy(),
					'n_types': fold_dict['n_types'],
					'birth_rate_idx': analysis_params['birth_rate_idx'],
					},
				n_epochs=n_epochs, lr=lr,
			)

			if isinstance(est_site, dict):		
				opt.fit_model.b0 = tf.constant(est_site['b0'], shape=opt.fit_model.n_betas, dtype=tf.dtypes.float64)
				opt.fit_model.site = tf.constant(est_site['site'], shape=[1, opt.fit_model.edge_ft.shape[1]], dtype=tf.dtypes.float64)

			opt.debug = True
			opt.fit_model.phylo_loss.i = tf.constant(0, shape=[], dtype=tf.dtypes.int32)
			train_vals, train_loss = opt.doOpt()
			results[sigma]['train'].append(float(train_loss))

			# DO TEST
			# -----------------------------------------------------
			fit_model_kwargs = {
					**fit_model_estimates,
					'loss_kwargs': {'reg_type': 'sigma', 'sigma': tf.constant(0, shape=[], dtype=tf.dtypes.float64)},
					'data': fold_dict['test']['data'].returnCopy(),
					'n_types': fold_dict['n_types'],
					'birth_rate_idx': analysis_params['birth_rate_idx'],
				}
			model = RandomEffectSite(**fit_model_kwargs)
			model.rand_eff = tf.Variable(train_vals['rand_eff'], dtype=tf.dtypes.float64)

			if est_site:
				if isinstance(est_site, dict):
					model.b0 = tf.Variable(est_site['b0'], shape=model.n_betas, dtype=tf.dtypes.float64)
					model.site = tf.Variable(est_site['site'], shape=[1, model.edge_ft.shape[1]], dtype=tf.dtypes.float64)
				else:
					model.b0 = tf.Variable(train_vals['b0'], shape=model.n_betas, dtype=tf.dtypes.float64)
					model.site = tf.Variable(train_vals['site'], shape=[1, model.edge_ft.shape[1]], dtype=tf.dtypes.float64)
			
			phylo_loss = model.phylo_loss(**model.loss_kwargs)
			phylo_loss.i = tf.constant(0, shape=[], dtype=tf.dtypes.int32)
			
			c = model.call()
			test_loss = phylo_loss.call(c.__dict__)
			results[sigma]['test'].append(float(test_loss.numpy()))
			results[sigma]['effs'].append({eff_name: [float(i) for i in effs] for eff_name, effs in train_vals.items()})

			print(f"train_loss={train_loss:.3f}, test_loss={test_loss:.3f}")

			(out_folder / "results.json").write_text(json.dumps(results, indent=4))

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

	results_obj = ResultsObj(analysis_dir)

	# -----------------------------------------------------
	# Init fitness model parameters
	# -----------------------------------------------------
	fit_model_params = init_model_params(results_obj, config)

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

	estimating = {k: v for k, v in fit_model_params.items() if isinstance(v, dict) and v.get('estimate', False)}
	estimating_str = ('+').join(sorted([f"{k}_TV" if is_TV(v) else k for k, v in estimating.items()]))
	model_name = config.get('model_name', None)
	result_key = f"{model_name}_{estimating_str}" if model_name else estimating_str

	# Create/load results dict
	if not results_obj.results_dict.get(result_key, None):
		results_obj.results_dict[result_key] = {
			'fit_model_params': fit_model_params, 
			'hyper_param_values': hyper_param_values,
			'results_list': {},
			}

	# -----------------------------------------------------
	# Test hyperparameter combinations in parallel
	# -----------------------------------------------------
	hyper_param_combos = [dict(zip(hyper_param_values.keys(), values)) for values in itertools.product(*hyper_param_values.values())]
	hyperparam_args = [[results_obj, result_key, h_combo, fit_model_params, config['iterative_pE'], n_epochs, lr, graph, debug] for h_combo in hyper_param_combos]

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
	# Save results of search, if we have any
	# -----------------------------------------------------
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
	results_obj.summarize_search(result_key)
	results_obj.plot_hyperparams(results_obj.folder / result_key)
	results_obj.do_validation(result_key, config["iterative_pE"], graph, debug)


def analyze_fit(analysis_dir, random_name, est_site=False, est_b0=False, n_epochs=50000, lr=0.00005):
	# -----------------------------------------------------
	# Create and/or load fold-segmented tree file 
	# as data object
	# -----------------------------------------------------
	all_data, phylo_obj, RO, analysis_params = load_data_and_RO_from_file(analysis_dir)
	
	if not est_b0:
		all_data.addArrayParams(b0=(1, False))

	out_folder = analysis_dir / random_name
	out_folder.mkdir(exist_ok=True, parents=True)

	# -----------------------------------------------------
	# Data we are getting train/test folds from should be
	# only the data that is NOT in our validation set
	# -----------------------------------------------------
	train_data = all_data.getSubArraySpecific(RO.train_idx)

	# -----------------------------------------------------
	# Load results, plot, find best sigma
	# -----------------------------------------------------
	results = {float(sig): v for sig, v in json.loads((out_folder / "results.json").read_text()).items()}

	for sig, sig_dict in results.items():
		sig_dict['test_mean'] = float(np.mean(sig_dict['test']))

	best_sigma, best_mean = sorted({sig: sig_dict['test_mean'] for sig, sig_dict in results.items()}.items(), key=lambda k: k[1])[0]

	fig, ax = plt.subplots()
	for sigma, sigma_dict in results.items():
		ax.plot(list(range(len(sigma_dict['test']))), sigma_dict['test'], label=sigma, c="red" if sigma==best_sigma else "black")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_folder / "fig1.png", dpi=300)
	plt.close("all")

	means = []
	sigmas = []
	fig, ax = plt.subplots()
	for sigma, sigma_dict in results.items():
		means.append(np.mean(sigma_dict['test']))
		sigmas.append(sigma)
	plt.scatter(sigmas, means)
	plt.tight_layout()
	plt.savefig(out_folder / "fig2.png", dpi=300)
	plt.close("all")

	(out_folder / "results.json").write_text(json.dumps(results, indent=4))

	# ----------------------------------------------------------------
	# Calculate/load type int info for both full and train-only dataset
	# ----------------------------------------------------------------
	random_info_file = out_folder / "full_random_info.json"

	if not random_info_file.exists():
		random_info = {}
		for name, data_obj in [["all", all_data]]:
			random_info[name] = {}

			edge_arr = data_obj.getEventArray("edge")
			indices = np.append(-1, np.unique(data_obj.array['idx'])).tolist()

			parent_idxs, parent_deltas = find_parents(indices, data_obj, all_data)
			random_info[name]['n_types'] = len(indices)

			random_info[name]['type_int'] = [indices.index(edge_arr[edge_arr['name']==sample_name.split("_")[0]][0]['idx']) for sample_name in data_obj.array['name']]
			random_info[name]['parent_type_int'] = [indices.index(i) for i in parent_idxs]
			random_info[name]['parent_time_delta'] = [float(i) for i in parent_deltas]

		random_info_file.write_text(json.dumps(random_info, indent=4))

	info = json.loads(random_info_file.read_text())
	
	# ----------------------------------------------------------------
	# Find random effects on full dataset
	# ----------------------------------------------------------------
	data_obj = all_data
		
	data_obj.addColumn('type_int', info["all"]['type_int'], np.int64)
	data_obj.addColumn('parent_type_int', info["all"]['parent_type_int'], np.int64)
	data_obj.addColumn('parent_time_delta', info["all"]['parent_time_delta'], np.float64)

	df = pd.DataFrame(data_obj.array)
	df.sort_values(by="name", inplace=True)

	# -----------------------------------------------------
	# Find random effects using best sigma
	# -----------------------------------------------------
	fit_model_estimates=dict(
			random_effect=[True, False],
			b0=[True, True],
			d=[False],
			s=[False],
			rho=[False],
			gamma=[False],
		)

	if est_site:
		fit_model_estimates['site'] = [True, False]
		fit_model_estimates['b0'] = [True, True]

	opt = Optimizer(
		fit_model=RandomEffectSite,
		fit_model_kwargs={
			**fit_model_estimates,
			'loss_kwargs': {'reg_type': 'sigma', 'sigma': tf.constant(sigma, shape=[], dtype=tf.dtypes.float64)},
			'data': data_obj.returnCopy(),
			'n_types': info["all"]['n_types'],
			'birth_rate_idx': analysis_params['birth_rate_idx'],
			},
		n_epochs=n_epochs, lr=lr,
	)

	if isinstance(est_site, dict):		
		opt.fit_model.b0 = tf.constant(est_site['b0'], shape=opt.fit_model.n_betas, dtype=tf.dtypes.float64)
		opt.fit_model.site = tf.constant(est_site['site'], shape=[1, opt.fit_model.edge_ft.shape[1]], dtype=tf.dtypes.float64)

	opt.debug = True
	opt.fit_model.phylo_loss.i = tf.constant(0, shape=[], dtype=tf.dtypes.int32)

	train_vals, train_loss = opt.doOpt()

	random_effects = train_vals['rand_eff']
	np.savetxt(out_folder / f"type_random_effects.txt", random_effects, delimiter=',')

	df = pd.DataFrame(data_obj.getEventArray("edge"))
	df['random_fitness'] = np.take(random_effects, df['type_int'])
	df[['name', 'random_fitness']].to_csv(out_folder / f"edge_random_effects.csv", index=False)

	(out_folder / f"loss.txt").write_text(f"{train_loss}")

def plot_random_branch_fitness(analysis_dir, random_name, est_site=False):
	all_data, phylo_obj, RO, params = load_data_and_RO_from_file(analysis_dir)

	out_folder = analysis_dir / random_name
	out_folder.mkdir(exist_ok=True, parents=True)

	# -----------------------------------------------------
	# Load tree, branch random effects
	# -----------------------------------------------------
	tt = pp.loadTree(
		phylo_obj.tree_file,
		internal=True,
		abs_time=phylo_obj.present_time
	)

	df = pd.read_csv(out_folder / "edge_random_effects.csv", index_col=0)

	vmin = df.min().min()
	vmax = df.max().max()

	# -----------------------------------------------------
	# Plot trees side by side
	# -----------------------------------------------------
	fig, ax = plt.subplots(figsize=(12, 25))

	fit_dict = df["random_fitness"].to_dict()
	c_func, cmap, norm = pp.continuousFunc(trait_dict=fit_dict, trait="name", cmap=sns.color_palette("flare", as_cmap=True), vmin=vmin, vmax=vmax)

	ax = pp.plotTraitAx(
		ax,
		tt,
		edge_c_func=c_func,
		node_c_func=c_func,
		tips=False,
		tip_names=False,
		zoom=False,
		title=f"Random Effects",
	)

	pp.add_cmap_colorbar(fig, ax, cmap, norm=norm)

	plt.tight_layout()
	plt.savefig(out_folder / f"Phylo_Random_Effects.png", dpi=300)
	plt.close("all")

def test(analysis_dir, random_name):
	all_data, phylo_obj, RO, params = load_data_and_RO_from_file(analysis_dir)
	all_data.addArrayParams(b0=(1, False))

	out_folder = analysis_dir / random_name
	out_folder.mkdir(exist_ok=True, parents=True)
	
	data = all_data.getSubArraySpecific(RO.train_idx)

	split_intervals(phylo_obj, all_data, data, out_folder, n_folds=3, test_proportion=(1/2), folds_start=1960)

if __name__ == "__main__":
	from yaml import CDumper as Dumper, CLoader as Loader, load, dump
	from pathlib import Path
	from ecoli_analysis.results_obj import ResultsObj
	from _analysis.test_prob import make_intervals

	config = load(Path("config.yaml").read_text(), Loader=Loader)
	analysis_name = config["analysis_name"]
	data_dir = Path(config["data_dir"])
	analysis_dir = data_dir / "analysis" / analysis_name

	if False:
		interval_times, interval_tree = make_intervals(
			"data_new/4_interval_tree_test_a",
			"data_new/a_test.nwk",
			config['last_sample_date'],
			config['sampling_prob_changepoints'],
			config['bioproject_changepoints'],
			)

	RO = ResultsObj(folder=analysis_dir)

	if RO.success["data"] == False:
		RO.set_data(
			tree_file=data_dir / "3_interval_tree" / "phylo.nwk",
			interval_times_file=data_dir / "3_interval_tree" / "interval_times.txt",
			last_sample_date=config["last_sample_date"],
			)
	if RO.success["index"] == False:
		RO.set_folds(test_size=0.2, n_splits=4, stratify=None, random_state=8)

	df = pd.DataFrame(RO.data.array)

	prep_data_for_hyperparam_search(analysis_dir, n_folds=3, test_proportion=(1/2), folds_start=1960, plot=True)





	


