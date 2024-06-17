import json
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from transmission_sim.analysis.param_model import ParamComponent, ParamModel, ComponentSite, ComponentB0, Site
from transmission_sim.analysis.phylo_loss import PhyloLossIterative
from transmission_sim.analysis.arrayer import PhyloArrayer, PhyloData
from transmission_sim.analysis.phylo_obj import PhyloObj
from transmission_sim.analysis.optimizer import Optimizer
from transmission_sim.ecoli.plot_test_reg import get_true_effs
from transmission_sim.ecoli.random_effects import dropbox_dir, RandomEffect, RandomEffectSite, PhyloLossRandomEffIterative, find_parents, find_parents_alt, get_parent_type_info
from transmission_sim.ecoli.analyze import load_data_and_RO_from_file
import matplotlib.pyplot as plt
import seaborn as sns
import transmission_sim.analysis.phylo_loss

transmission_sim.analysis.phylo_loss.use_graph_execution = False

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

def split_intervals(phylo_obj, all_data, train_data, out_folder, n_folds, test_proportion, folds_start=None):
	# -----------------------------------------------------
	# Create interval breakpoints and put into dictionary
	# -----------------------------------------------------
	if folds_start:
		start = folds_start
	else:
		start = phylo_obj.root_time

	end = phylo_obj.present_time

	# Get starting times of of folds
	fold_times = np.linspace(start, end, n_folds + 1)

	# Get length of first fold, separate that into 
	# test and train periods based on test_proportion
	fold_period = fold_times[1] - fold_times[0]
	test_period = fold_period * test_proportion
	train_period = fold_period - test_period

	# Make dictionary of fold values and interval times
	interval_times = []
	folds = {}
	for i, fold_time in enumerate(fold_times[0:-1]):
		folds[i] = {'train': {}, 'test': {}}

		folds[i]['train'] = {
			'start_time': fold_time,
			'end_time': fold_time + train_period,
			'idx': len(interval_times),
		}
		interval_times.append(fold_time)

		folds[i]['test'] = {
			'start_time': fold_time + train_period,
			'idx': len(interval_times),
		}
		interval_times.append(fold_time + train_period)

		if i != 0:
			folds[i-1]['test']['end_time'] = fold_time
	
	folds[i]['test']['end_time'] = phylo_obj.present_time

	# -----------------------------------------------------
	# Get indexes of phylogeny pieces in the train/test
	# datasets for each fold, plus get their self and/or
	# parental fitness index and time deltas
	# -----------------------------------------------------
	for i, folds_dict in folds.items():
		# Get times corresponding with the train/test
		# datasets for this interval
		train_interval = folds_dict['train']['idx']
		test_interval = folds_dict['test']['idx']
		
		# Get phylogeny pieces that are in the scope of this fold
		train_start_pass = train_data.array['birth_time'] <= folds_dict['train']['end_time']
		train_
		train_idx = np.nonzero()[0]

		breakpoint()
		##
		# we need to have some sort of branch dictionary here and get names of branches that start,
		# then select all intervals of that branch from data
		##

		test_start_pass = train_data.array['birth_time'] > folds_dict['test']['start_time']
		test_end_pass = train_data.array['birth_time'] < folds_dict['test']['end_time']
		test_idx = np.nonzero(test_start_pass * test_end_pass)[0]

		folds_dict['train']['data_idx'] = [int(i) for i in train_idx]
		folds_dict['test']['data_idx'] = [int(i) for i in test_idx]

		# Get fitness indices and time deltas
		fold_train = train_data.getSubArraySpecific(train_idx)
		fold_test = train_data.getSubArraySpecific(test_idx)
		type_info_dict = get_parent_type_info(fold_train, fold_test, all_data)

		folds_dict['train'] = {**folds[i]['train'], **type_info_dict['train']}
		folds_dict['test'] = {**folds[i]['test'], **type_info_dict['test']}
		folds_dict['n_types'] = type_info_dict['n_types']

	phylo_plot_in_train_test(phylo_obj, train_data, folds, out_folder)

	(out_folder / "fold_params.json").write_text(
		json.dumps(
			dict(
					n_folds=n_folds, 
					test_proportion=test_proportion,
					interval_times=interval_times,
					folds=folds,
				)
			)
		)

def load_data(analysis_name, random_name):
	all_data, phylo_obj, RO, params, analysis_out_dir = load_data_and_RO_from_file(analysis_name)
	all_data.addArrayParams(b0=(1, False))

	out_folder = analysis_out_dir / random_name
	out_folder.mkdir(exist_ok=True, parents=True)

	# -----------------------------------------------------
	# Data we are getting train/test folds from should be
	# only the data that is NOT in our validation set
	# -----------------------------------------------------
	# data = all_data.getSubArraySpecific(RO.train_idx)
	data = all_data

	# -----------------------------------------------------
	# Load / make time folds info
	# -----------------------------------------------------
	fold_params_file = out_folder / "fold_params.json"
	if not fold_params_file.exists():
		split_intervals(phylo_obj, all_data, data, out_folder, n_folds=3, test_proportion=(1/2), folds_start=1965)

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

	return data, phylo_obj, fold_params, folds, out_folder

def do_crossval(analysis_name, random_name, est_site=False):
	# -----------------------------------------------------
	# Create and/or load fold-segmented tree file 
	# as data object
	# -----------------------------------------------------
	data, phylo_obj, fold_params, folds, out_folder = load_data(analysis_name, random_name)

	# -----------------------------------------------------
	# Do sigma hyperparameter optimization
	# -----------------------------------------------------
	sigmas = np.linspace(0, 5, 20)
	results = {}
	for sigma in sigmas:
		results[sigma] = {'train': [], 'test': [], 'effs': []}

		for i, fold_dict in folds.items():
			print(f"\nsigma={sigma}: fold {i}")
			
			# DO TRAIN
			# -----------------------------------------------------
			opt = Optimizer(
				fit_model=RandomEffectSite if est_site else RandomEffect,
				fit_model_kwargs=dict(
					data=fold_dict['train']['data'].returnCopy(),
					n_types=fold_dict['n_types'],
					random_effect=[True, False],
					b0=[False],
					d=[False],
					s=[False],
					rho=[False],
					gamma=[False],
					loss_kwargs={'reg_type': 'sigma', 'sigma': sigma}
				),
				n_epochs=20000, lr=0.00005,
			)
			if est_site: fit_model_kwargs['site'] = [True, False]

			opt.debug = True
			opt.fit_model.phylo_loss.i = 0
			train_vals, train_loss = opt.doOpt()
			results[sigma]['train'].append(float(train_loss))

			# DO TEST
			# -----------------------------------------------------
			random_effs = train_vals[0]
			if est_site: site = train_vals[1]
			fit_model_kwargs = dict(
					data=fold_dict['test']['data'].returnCopy(),
					n_types=fold_dict['n_types'],
					random_effect=[True, False],
					b0=[False],
					d=[False],
					s=[False],
					rho=[False],
					gamma=[False],
					loss_kwargs={'reg_type': 'sigma', 'sigma': 0}
				)
			if est_site:
				fit_model_kwargs['site'] = [True, False]
				model = RandomEffectSite(**fit_model_kwargs)
			else:
				model = RandomEffect(**fit_model_kwargs)

			phylo_loss = model.phylo_loss(**model.loss_kwargs)
			phylo_loss.i = 0
			model.rand_eff = tf.Variable(random_effs, dtype=tf.dtypes.float64)
			
			if est_site:
				model.site = tf.Variable(site, dtype=tf.dtypes.float64)

			c = model.call()
			test_loss = phylo_loss.call(c.__dict__)
			results[sigma]['test'].append(float(test_loss.numpy()))
			results[sigma]['effs'].append([float(i) for i in random_effs])

			print(f"train_loss={train_loss:.3f}, test_loss={test_loss:.3f}")

	fig, ax = plt.subplots()
	for sigma, sigma_dict in results.items():
		ax.plot(list(range(len(folds))), sigma_dict['test'], label=sigma)
		# print(f"{sigma}: {sigma_dict['test']}")
		# print(sigma_dict['effs'][0])
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
	plt.savefig(out_folder / "fig2.png")
	plt.close("all")

	for sig, sig_dict in results.items():
		sig_dict['test_mean'] = float(np.mean(sig_dict['test']))

	(out_folder / "results.json").write_text(json.dumps(results, indent=4))

def analyze_fit(analysis_name, random_name, est_site=False, est_b0=False):
	# -----------------------------------------------------
	# Create and/or load fold-segmented tree file 
	# as data object
	# -----------------------------------------------------
	all_data, phylo_obj, RO, params, analysis_out_dir = load_data_and_RO_from_file(analysis_name)
	
	if not est_b0:
		all_data.addArrayParams(b0=(1, False))

	out_folder = analysis_out_dir / random_name
	out_folder.mkdir(exist_ok=True, parents=True)

	# -----------------------------------------------------
	# Data we are getting train/test folds from should be
	# only the data that is NOT in our validation set
	# -----------------------------------------------------
	train_data = all_data.getSubArraySpecific(RO.train_idx)

	# -----------------------------------------------------
	# Load results, find best sigma
	# -----------------------------------------------------
	results = {float(sig): v for sig, v in json.loads((out_folder / "results.json").read_text()).items()}

	best_sigma, best_mean = sorted({sig: sig_dict['test_mean'] for sig, sig_dict in results.items()}.items(), key=lambda k: k[1])[0]

	# for name, data_obj in [["train_only", train_data], ["all", all_data]]:
	for name, data_obj in [["all", all_data]]:
		edge_arr = data_obj.getEventArray("edge")

		indices = np.append(-1, np.unique(data_obj.array['idx'])).tolist()
		parent_idxs, parent_deltas = find_parents_alt(indices, data_obj, all_data)
		n_types=len(indices)
		type_int=[indices.index(edge_arr[edge_arr['name']==name.split("_")[0]][0]['idx']) for name in data_obj.array['name']]
		parent_type_int=[indices.index(i) for i in parent_idxs]
		parent_time_delta=[float(i) for i in parent_deltas]

		data_obj.addColumn('type_int', type_int, np.int64)
		data_obj.addColumn('parent_type_int', parent_type_int, np.int64)
		data_obj.addColumn('parent_time_delta', parent_time_delta, np.float64)

		df = pd.DataFrame(data_obj.array)
		df.sort_values(by="name", inplace=True)

		# -----------------------------------------------------
		# Find random effects using best sigma
		# -----------------------------------------------------
		opt = Optimizer(
			fit_model=RandomEffectSite if est_site else RandomEffect,
			fit_model_kwargs=dict(
				data=data_obj,
				n_types=n_types,
				birth_rate_idx=self.birth_rate_idx,
				random_effect=[True, False],
				b0=[False],
				d=[False],
				s=[False],
				rho=[False],
				gamma=[False],
				loss_kwargs={'reg_type': 'sigma', 'sigma': best_sigma}
			),
			n_epochs=50000, lr=0.0005,
		)

		if est_site: fit_model_kwargs['site'] = [True, False]
		if est_b0: fit_model_kwargs['b0'] = [True, True]

		opt.debug = True
		opt.fit_model.phylo_loss.i = 0
		train_vals, train_loss = opt.doOpt()

		random_effects = train_vals[0]
		np.savetxt(out_folder / f"type_random_effects_{name}.txt", random_effects, delimiter=',')

		df = pd.DataFrame(data_obj.getEventArray("edge"))
		df['random_fitness'] = np.take(random_effects, df['type_int'])
		df[['name', 'random_fitness']].to_csv(out_folder / f"edge_random_effects_{name}.csv", index=False)

		(out_folder / f"{name}_loss.txt").write_text(f"{train_loss}")

def plot_random_branch_fitness(analysis_name, random_name, est_site=False):
	all_data, phylo_obj, RO, params, analysis_out_dir = load_data_and_RO_from_file(analysis_name)

	out_folder = analysis_out_dir / random_name
	out_folder.mkdir(exist_ok=True, parents=True)

	# -----------------------------------------------------
	# Load tree, branch random effects
	# -----------------------------------------------------
	tt = pp.loadTree(
		phylo_obj.tree_file,
		internal=True,
		abs_time=phylo_obj.present_time
	)

	df = pd.read_csv(out_folder / "edge_random_effects_all.csv", index_col=0)

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

	# # -----------------------------------------------------
	# # Check if the same
	# # -----------------------------------------------------
	# sns.set_style("whitegrid")
	# scatter = sns.scatterplot(data=df, x="all", y="train_only")
	# plt.xlim(vmin, vmax)
	# plt.ylim(vmin, vmax)
	# plt.tight_layout()
	# plt.savefig(out_folder / f"Compare_Train_All_Random_Effs.png", dpi=300)
	# plt.close("all")

	# # -----------------------------------------------------
	# # Random effects through time
	# # -----------------------------------------------------
	# all['time'] = all_data.getEventArray("edge")['event_time']
	# scatter = sns.scatterplot(data=all, x="time", y="random_fitness")
	# plt.tight_layout()
	# plt.savefig(out_folder / f"Random_Effs_Time.png", dpi=300)
	# plt.close("all")

	# # -----------------------------------------------------
	# # Random effects by branch length
	# # -----------------------------------------------------
	# all['length'] = all_data.getEventArray("edge")['time_step']
	# scatter = sns.scatterplot(data=all, x="length", y="random_fitness")
	# plt.tight_layout()
	# plt.savefig(out_folder / f"Random_Effs_Length.png", dpi=300)
	# plt.close("all")
	
if __name__ == "__main__":
	from transmission_sim.analysis.PhyloRegressionTree import PhyloBoost
	import matplotlib as mpl
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.model_selection import TimeSeriesSplit
	import transmission_sim.utils.plot_phylo_standalone as pp

	analysis_name = "3-interval_constrained-sampling"
	random_name = "condit_test"
	
	do_crossval(analysis_name, random_name, est_site=False)
	# analyze_fit(analysis_name, random_name, est_site=False)
	# plot_random_branch_fitness(analysis_name, random_name, est_site=False)
