from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.lib import recfunctions as rfn

from sklearn.model_selection import KFold, train_test_split
from transmission_sim.analysis.optimizer import Optimizer
from transmission_sim.analysis.param_model import Site, ComponentB0, ParamModel, ComponentSite
from transmission_sim.analysis.phylo_obj import PhyloObj
from transmission_sim.analysis.arrayer import PhyloArrayer
from ecoli_analysis.param_intervals import constrain_sampling

class ComponentB0Mod(ComponentB0):
	"""
	"base" birth rate (b0) effect methods
	but where we don't estimate a b0 for each interval

	self.birth_rate_idx will need to be added to param model
	"""
	def init_b0(self):
		super().init_b0()

		self.n_betas = len(tf.unique(self.birth_rate_idx)[0])

		if list(self.bdm['b0'])[0]:
			if list(self.bdm['b0'])[1]:
				self.b0 = tf.Variable(tf.ones(shape=self.n_betas, dtype=tf.dtypes.float64) + .000001, name='b0')

	# Only time varying methods get overridden
	# ----------------------------------------
	def get_b0VarTV(self):
		expand_b0 = tf.gather(self.b0, self.birth_rate_idx)
		self.edge_b0 = tf.gather(expand_b0, self.p.edge_param_interval)
		self.birth_b0 = tf.gather(expand_b0, self.p.birth_param_interval)

	def get_b0VarTV_pE(self):
		expand_b0 = tf.gather(self.b0, self.birth_rate_idx)
		self.edge_b0 = tf.gather(expand_b0, self.p.edge_param_interval)
		self.birth_b0 = tf.gather(expand_b0, self.p.birth_param_interval)
		self.pE_b0 = self.pE_ones * expand_b0

class SiteMod(ParamModel, ComponentB0Mod, ComponentSite):
	"""
	model params: b0, site, d, s, rho
	call() calls self.edge_b0, self.birth_b0, + self.pE_b0 if pE
	"""
	def __init__(self, data, b0, site, gamma, d, s, rho, birth_rate_idx, iterative_pE=True, **kwargs):
		self.bdm = dict(b0=b0, site=site, d=d, s=s, rho=rho, gamma=gamma)
		self.private_vars = [k for k, v in self.bdm.items() if v[0] == True]
		self.vars = ['b'] + self.private_vars

		self.birth_rate_idx = tf.constant(birth_rate_idx)

		super().__init__(data=data, iterative_pE=iterative_pE, **kwargs)

		# Set parameter methods
		self.setMethods()

		# Initialize parameters
		for var in self.bdm:
			if var not in self.base_bdm_params:
				getattr(self, f"init_{var}")()

	def call_(self):
		for var in self.private_vars:
			getattr(self, f"{var}_call")()

		self.p.edge_b = self.edge_site_b * self.edge_b0
		self.p.birth_b = self.birth_site_b * self.birth_b0

		return self.p

	def call_PE(self):
		for var in self.private_vars:
			getattr(self, f"{var}_call")()

		self.p.edge_b = self.edge_site_b * self.edge_b0
		self.p.birth_b = self.birth_site_b * self.birth_b0
		self.p.pE_b = tf.transpose(tf.transpose(self.pE_b0) * self.pE_site_b)

		return self.p

class ResultsObj():
	def __init__(self, data):
		self.data = data
		self.train_idx = []
		self.validate_idx = []
		self.cv_idxs = []
		self.results_dict = {}

	def loadDataByIdx(self, idx):
		return self.data.getSubArraySpecific(idx)

	def get_folds(self):
		"""
		# Get and set indices corresponding to
		# train and validation data sets
		# + Split train for cross-validation
		"""

		self.train_idx, self.validate_idx, _, _ = train_test_split(
			list(range(len(self.data.array))),
			[0] * len(self.data.array),
			random_state=8,
			test_size=0.2,
			stratify=None,
		)

		kf = KFold(
			n_splits=4,
			shuffle=True,
			random_state=8
		)
		cv_idxs = list(kf.split(self.train_idx))
		self.cv_idxs = [[f.tolist() for f in cv] for cv in cv_idxs]

	def load_results(self, dir):
		index_success = False
		analysis_success = False

		if (index_file := (Path(dir) / "idxs.json")).exists():
			with open(index_file, "r+") as f:
				i = json.load(f)
				for k, v in i.items():
					setattr(self, k, v)
			print(f"Test/train/validate indices loaded from file")
			index_success = True
		else:
			print(f"{index_file} not found")

		if (analysis_file := (Path(dir) / "analysis.json")).exists():
			with open(analysis_file, "r+") as f:
				a = json.load(f)
				self.results_dict = a
			print(f"Analysis results loaded from file")
			analysis_success = True
		else:
			print(f"{analysis_file} not found")

		return index_success, analysis_success

	def save(self, dir):
		idx_dict = dict(
			train_idx = self.train_idx,
			validate_idx = self.validate_idx,
			cv_idxs = self.cv_idxs,
			)

		save_dict = self.results_dict

		with open(Path(dir) / "idxs.json", "w+") as f:
			json.dump(idx_dict, f, indent=4)

		with open(Path(dir) / "analysis.json", "w+") as f:
			json.dump(save_dict, f, indent=4)

	def do_train(self, h_combo, n_epochs, lr, cv_train, bdm_params, return_opt=False, **kwargs):
		# Set up optimization for training
		opt = Optimizer(
			fit_model=SiteMod,
			fit_model_kwargs=dict(
				**bdm_params,
				birth_rate_idx=self.birth_rate_idx,
				data=cv_train,
				iterative_pE=True,
				loss_kwargs={
					**h_combo,
				}),
			n_epochs=n_epochs, lr=lr,
		)
		opt.verbose = True
		opt.debug = False

		if return_opt:
			opt.save_values = True
		else:
			opt.save_values = False

		# Do training, save loss
		estimates, train_loss = opt.doOpt()

		if return_opt:
			return estimates, train_loss, opt
		else:
			return estimates, train_loss

	def do_test(self, estimates, cv_test, bdm_params):
		fit_model_kwargs = dict(
			**bdm_params,
			birth_rate_idx=self.birth_rate_idx,
			data=cv_test,
			iterative_pE=True,
			loss_kwargs={}
		)
		model = SiteMod(**fit_model_kwargs)
		phylo_loss = model.phylo_loss(**model.loss_kwargs)

		if bdm_params['b0'][0]:
			model.b0 = tf.Variable(estimates['b0'], dtype=tf.dtypes.float64)

			if bdm_params['site'][0]:
				model.site = tf.Variable(estimates['site'], dtype=tf.dtypes.float64)

		else:
			if bdm_params['site'][0]:
				model.site = tf.Variable(estimates['site'], dtype=tf.dtypes.float64)

		c = model.call()
		test_loss = phylo_loss.call(c.__dict__).numpy()
		return test_loss

	def do_validation(self, result_key, save_dir, feature_names):
		# Run on full dataset
		# ------------------------------------
		best_iter = sorted(self.results_dict[result_key]['results_list'].items(), key=lambda d: d[1]['mean_test_loss'])[0]

		best_lr = best_iter[1]["lr"]
		best_n_epochs = best_iter[1]["n_epochs"]
		best_hyperparams = best_iter[1]["h_combo"]

		if self.results_dict[result_key].get("full"):
			if self.results_dict[result_key]["full"]["h_combo"] == {k: v for k, v in best_hyperparams.items()}:
				print(f"\n******* Already ran validation on best hyperparams {best_hyperparams} *******")
				return
		else:
			self.results_dict[result_key]["full"] = {}
		
		results = self.results_dict[result_key]["full"]

		results['h_combo'] = {k: v for k, v in best_hyperparams.items()}

		train = self.loadDataByIdx(self.train_idx)
		test = self.loadDataByIdx(self.validate_idx)

		validation_estimates, validation_train_loss, opt = self.do_train(
			best_hyperparams, 
			best_n_epochs, 
			best_lr, 
			train, 
			self.results_dict[result_key]["bdm_params"], 
			return_opt=True,
			)

		results[f"train_loss"] = float(validation_train_loss)
		results[f"estimates"] = {v: e.tolist() for v, e in validation_estimates.items()}
		results["train_n_epochs"] = len(opt.losses)

		# Test
		validation_test_loss = self.do_test(validation_estimates, test, self.results_dict[result_key]["bdm_params"])
		results[f"test_loss"] = float(validation_test_loss)
		self.save(save_dir)

		self.run_and_plot(opt, feature_names, save_dir)

	def crossvalidate(self, bdm_params, hyper_param_values, save_dir, feature_names, n_epochs=20000, lr=0.01, **kwargs):
		"""
		Do cross-validation with given variables 
		and given hyperparameter values

		If cross-validation with given variables exists,
		run new hyperparameter values
		"""

		# Set result key based on what parameters we are estimating
		# and whether they are time varying ("_TV")
		estimating = {k: v for k, v in bdm_params.items() if v[0] == True}
		result_key = ('+').join(sorted([f"{k}_TV" if (len(v) > 1 and v[1] == True) else k for k, v in estimating.items()]))

		# Create/load results dict
		if not self.results_dict.get(result_key, None):
			self.results_dict[result_key] = {
				'bdm_params': bdm_params, 
				'hyper_param_values': hyper_param_values,
				'results_list': {},
				}

		hyper_param_combos = [dict(zip(hyper_param_values.keys(), values)) for values in itertools.product(*hyper_param_values.values())]
		
		# Find best hyperparameter combination
		# ------------------------------------
		# For each hyperparameter combination
		for h_combo in hyper_param_combos:
			print(f"\n------------------------------------")
			print(f"Testing hyperparameters: {h_combo}")
			print(f"------------------------------------")

			combo_key = "_".join(sorted([f"{k}={v}" for k, v in h_combo.items()])) + f"_lr={lr}_n_epochs={n_epochs}"
			do_analysis = True

			if self.results_dict[result_key]["results_list"].get(combo_key):
				if self.results_dict[result_key]["results_list"][combo_key].get(f'mean_test_loss'):
					print(f"Already tested {combo_key}, moving on")
					do_analysis = False
			else:
				self.results_dict[result_key]["results_list"][combo_key] = {"h_combo": {k: v for k, v in h_combo.items()}, 'lr': lr, 'n_epochs': n_epochs}
			
			combo_results = self.results_dict[result_key]["results_list"][combo_key]

			if do_analysis:
				# For each fold
				for i, (cv_train_idx, cv_test_idx) in enumerate(self.cv_idxs):

					cv_train = self.loadDataByIdx(cv_train_idx)
					cv_test = self.loadDataByIdx(cv_test_idx)

					# Train
					estimates, train_loss = self.do_train(h_combo, n_epochs, lr, cv_train, bdm_params, **kwargs)
					combo_results[f"fold_{i}_train_loss"] = float(train_loss)
					combo_results[f"fold_{i}_estimates"] = {variable: e.tolist() for variable, e in estimates.items()}

					# Test
					test_loss = self.do_test(estimates, cv_test, bdm_params)
					combo_results[f"fold_{i}_test_loss"] = float(test_loss)

					print(f"Fold {i}: Train loss={train_loss:.3f}, Test loss={test_loss:.3f}")
					print(f"Estimate={estimates}")
					
				combo_results['mean_train_loss'] = float(np.mean([combo_results[f"fold_{i}_train_loss"] for i in range(len(self.cv_idxs))]))
				combo_results['mean_test_loss'] = float(np.mean([combo_results[f"fold_{i}_test_loss"] for i in range(len(self.cv_idxs))]))

				print(f"Mean train loss: {combo_results['mean_train_loss']:.3f}, Mean test loss: {combo_results['mean_test_loss']:.3f}")
				self.save(save_dir)

		self.do_validation(result_key, save_dir, feature_names)

	def plot_variable_epochs(self, values, variable_slice_func, save_name, variable_names=[]):
		selected_values = variable_slice_func(values)

		df = pd.DataFrame(selected_values)
		if variable_names:
			df.columns = variable_names

		sns.set_style("whitegrid")
		sns.set_context("paper")
		sns.lineplot(data=df)
		plt.tight_layout()
		plt.savefig(save_name, dpi=300)
		plt.close("all")

	def run_and_plot(self, opt, feature_names, save_dir):
		losses = opt.losses

		save_dir.mkdir(exist_ok=True, parents=True)

		# ==========================================
		# Plot loss over epochs
		# ==========================================
		best_idx = np.argmin(losses)
		y_limit = np.quantile(losses, .95)

		for i, l in enumerate(losses):
		    if l <= y_limit:
		        x_limit = i
		        break

		plt.plot(list(range(len(losses)))[x_limit:], losses[x_limit:])
		plt.axvline(best_idx, color='red')
		plt.tight_layout()
		plt.savefig(save_dir / "epochs_95.png", dpi=300)
		plt.close("all")

		# ==========================================
		# Plot loss over epochs, zoomed in
		# ==========================================
		y_limit = np.quantile(losses, .5)
		for i, l in enumerate(losses):
		    if l <= y_limit:
		        x_limit = i
		        break

		plt.plot(list(range(len(losses)))[x_limit:], losses[x_limit:])
		plt.axvline(best_idx, color='red')
		plt.tight_layout()
		plt.savefig(save_dir / "epochs_50.png", dpi=300)
		plt.close("all")

		# ==========================================
		# Plot values over epochs
		# ==========================================

		# Time features
		self.plot_variable_epochs(opt.values, lambda x: [est[0] for est in x], save_name=save_dir / "var_time.png", variable_names=[])

		# Site features
		for feature_group in ["PLASMID", "AMR", "META", "VIR", "STRESS"]:
			locs = [i for i, n in enumerate(feature_names) if feature_group in n]
			names = [n for i, n in enumerate(feature_names) if feature_group in n]

			if feature_group != "AMR":
				self.plot_variable_epochs(opt.values, lambda x: [est[1][0][locs] for est in x], save_name=save_dir / f"var_{feature_group}.png", variable_names=names)

			else:
				self.plot_variable_epochs(opt.values, lambda x: [est[1][0][locs[0:15]] for est in x], save_name=save_dir / f"var_{feature_group}_1.png", variable_names=names[0:15])
				self.plot_variable_epochs(opt.values, lambda x: [est[1][0][locs[15:]] for est in x], save_name=save_dir / f"var_{feature_group}_2.png", variable_names=names[15:])

		try:
			pd.DataFrame([est[0] for est in opt.values]).to_csv(save_dir / "time_var_ests.csv")
			pd.DataFrame([est[1][0] for est in opt.values], columns=feature_names).to_csv(save_dir / "site_var_ests.csv")
			np.savetxt(save_dir / "losses.csv", np.array(losses), delimiter=",")

		except Exception as e:
			pass

def get_data(tree_file, features_file):
	# -----------------------------------------------------
	# Read in time intervals
	# -----------------------------------------------------
	interval_times = [float(t) for t in (tree_file.parent / "interval_times.txt").read_text().splitlines()]

	# -----------------------------------------------------
	# Load and date tree
	# -----------------------------------------------------
	phylo_obj = PhyloObj(
		tree_file=tree_file,
		tree_schema="newick",
		features_file=features_file,
		params={}
	)
	# Date the tree
	last_sample_date = 2023
	
	for n in phylo_obj.tree.nodes():
		n.age = n.age + (last_sample_date - phylo_obj.present_time)

	phylo_obj.root = phylo_obj.tree.seed_node
	phylo_obj.root_time = phylo_obj.root.age - (phylo_obj.root.edge_length if phylo_obj.root.edge_length else 0)
	phylo_obj.present_time = last_sample_date

	# -----------------------------------------------------
	# Convert input data to array, set array params
	# -----------------------------------------------------
	# Load data as PhyloData array
	arrayer = PhyloArrayer(
		phylo_obj=phylo_obj,
		param_interval_times=interval_times,
	)
	data = arrayer.toData()
	data.iterative_pE = True

	return data, phylo_obj

def load_data_and_RO_from_file(analysis_dir):
	out_dir = analysis_dir
	params = json.loads((out_dir / "params.json").read_text())

	# -----------------------------------------------------
	# Get data object, add birth-death model params
	# -----------------------------------------------------
	data, phylo_obj = get_data(Path(params["tree_file"]), Path(params["features_file"]))

	if 'constrained_sampling_rates' in params:
		print("Loading pre-calculated constrained sampling rates")
		data.array = rfn.append_fields(
			data.array, 
			params['constrained_sampling_rates']['names'],
			params['constrained_sampling_rates']['values'], 
			dtypes=[float for _ in params['constrained_sampling_rates']['values']],
			usemask=False
		)
	else:
		print("No pre-calculated constrained sampling rates")

	data.addArrayParams(**params["bd_array_params"])

	RO = ResultsObj(data)

	RO.birth_rate_idx = params["birth_rate_idx"]
	index_success, analysis_success = RO.load_results(out_dir)

	if not index_success:
		RO.get_folds()

	print("Did load_data_and_RO_from_file")

	return data, phylo_obj, RO, params, out_dir

def load_data_and_RO(analysis_dir, name, tree_file, features_file, bd_array_params, bioproject_times, constrained_sampling_rate, birth_rate_changepoints, n_epochs):
	out_dir = analysis_dir / name
	out_dir.mkdir(exist_ok=True, parents=True)

	# -----------------------------------------------------
	# Get data object, add birth-death model params
	# -----------------------------------------------------
	data, phylo_obj = get_data(tree_file, features_file)

	if constrained_sampling_rate:
		 data, constrained_sampling_rates = constrain_sampling(data, phylo_obj, constrained_sampling_rate, bioproject_times)

	data.addArrayParams(**bd_array_params)

	pd.DataFrame(data.array).to_csv(out_dir / "data.csv")

	# -----------------------------------------------------
	# Get a list of what birth rate indices each time
	# interval should use
	# -----------------------------------------------------
	# add root time to birth rate changepoints if not already there
	if phylo_obj.root_time not in birth_rate_changepoints:
		birth_rate_changepoints = [phylo_obj.root_time] + birth_rate_changepoints

	birth_rate_idx = [np.where(time > birth_rate_changepoints)[0][-1] if time != birth_rate_changepoints[0] else 0 for time in data.param_interval_times]

	# -----------------------------------------------------
	# Save analysis parameters
	# -----------------------------------------------------
	params = dict(
		name=name,
		tree_file=str(tree_file),
		features_file=str(features_file),
		n_epochs=n_epochs,
		interval_times=list(map(float, data.param_interval_times)),
		birth_rate_changepoints=birth_rate_changepoints,
		birth_rate_idx=list(map(int, birth_rate_idx)),
		bd_array_params=bd_array_params,
		constrained_sampling_rate=constrained_sampling_rate,
		constrained_sampling_rates=constrained_sampling_rates,
		)
	param_dict = json.dumps(params, indent=4)
	(out_dir / "params.json").write_text(param_dict)

	# -----------------------------------------------------
	# Instantiate results object, load previous
	# analyses if they exist, and get indices
	# -----------------------------------------------------
	RO = ResultsObj(data)

	RO.birth_rate_idx = birth_rate_idx
	index_success, analysis_success = RO.load_results(out_dir)

	if not index_success:
		RO.get_folds()

	return data, phylo_obj, RO, params, out_dir

	