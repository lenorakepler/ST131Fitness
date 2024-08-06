import json
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.optimizer import Optimizer
from ecoli_analysis.random_effects import find_parents, split_intervals
from ecoli_analysis.random_effects_classes import RandomEffectSite
from ecoli_analysis.results_obj import load_data_and_RO_from_file

def phylo_plot_in_train_test(phylo_obj, train_data, folds, out_folder):
	# Load tree
	tt = pp.loadTree(
		phylo_obj.tree_file,
		internal=True,
		abs_time=phylo_obj.present_time
	)

	n_folds = len(folds)
	fig, axs = plt.subplots(1, n_folds, figsize=(12 * n_folds, 25))
	axs = axs.ravel()

	for fold_num, fold_dict in folds.items():
		test_start = fold_dict['test']['start_time']
		test_end = fold_dict['test']['end_time']

		fold_trait = {
			**{n: "Train" for n in train_data.array[fold_dict['train']['data_idx']]['name']},
			**{n: "Test" for n in train_data.array[fold_dict['test']['data_idx']]['name']},
		}

		colors, c_func = pp.categoricalFunc(fold_trait, 'name', legend=True)

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
		axs[fold_num].axvline(x=test_end, linestyle="--", color="red")

	pp.add_legend(colors, axs[fold_num], lloc="lower left")

	plt.tight_layout()
	plt.savefig(out_folder / f"Phylo_Folds.png", dpi=300)
	plt.close("all")


def load_data(analysis_dir, random_name):
	all_data, phylo_obj, RO, params = load_data_and_RO_from_file(analysis_dir)
	all_data.addArrayParams(b0=(1, False))

	out_folder = analysis_dir / random_name
	out_folder.mkdir(exist_ok=True, parents=True)

	data = all_data

	# -----------------------------------------------------
	# Load / make time folds info
	# -----------------------------------------------------
	fold_params_file = out_folder / "fold_params.json"
	if not fold_params_file.exists():
		split_intervals(phylo_obj, all_data, data, out_folder, n_folds=3, test_proportion=(1/2), root_time=phylo_obj.root_time, present_time=phylo_obj.present_time, folds_start=1960)

	fold_params = json.loads((fold_params_file).read_text())
	
	# -----------------------------------------------------
	# Get train and test arrays for each of the folds
	# Make/load parent and time delta info
	# -----------------------------------------------------
	folds = {int(k): v for k, v in fold_params['folds'].items()}

	# phylo_plot_in_train_test(phylo_obj, data, folds, out_folder)

	for i, folds_dict in folds.items():
		train_dict = folds_dict['train']
		test_dict = folds_dict['test']

		fold_train = data.getSubArraySpecific(train_dict['data_idx'])
		fold_test = data.getSubArraySpecific(test_dict['data_idx'])

		# Add columns specifying the index of the birth rate that should be 
		# used for each phylogeny piece (note that this is different than idx).
		# For the test set, we use the birth rate index of the parent phylogeny piece, 
		# and for the training set, we use the index of the self birth rate index.
		fold_train.addColumn('type_int', train_dict['type_int'], np.int64)
		fold_test.addColumn('type_int', test_dict['type_int'], np.int64)

		# Add columns specifying the parent birth rate index for just the
		# training set (placeholder for test because param model requires it)
		fold_train.addColumn('parent_type_int', train_dict['parent_type_int'], np.int64)
		fold_test.addColumn('parent_type_int', test_dict['type_int'], np.int64)

		# Add the parent time delta to the train and test data.
		fold_train.addColumn('parent_time_delta', train_dict['parent_time_delta'], np.float64)
		fold_test.addColumn('parent_time_delta', test_dict['parent_time_delta'], np.float64)
		
		train_dict['data'] = fold_train
		test_dict['data'] = fold_test

	return data, phylo_obj, fold_params, folds, out_folder, params

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


