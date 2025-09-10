import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import re
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
import json
import tensorflow as tf
import analysis.plot_phylo_standalone as pp
import seaborn as sns
import matplotlib.pyplot as plt
from model_fit.optimizer import Optimizer
from analysis.param_model import SiteMarginal
from model_fit.phylo_obj import PhyloObj, PhyloObjPlain
from model_fit.phylo_loss import PhyloLossIterative
from model_fit.arrayer import PhyloArrayer
from data_prep.param_intervals import make_intervals
from tensorflow.keras import Model as KerasModel
from types import SimpleNamespace

from analysis.param_model import ParamModel

class PosNonZero(tf.keras.constraints.Constraint):
	def __call__(self, w):
		return tf.where(w < 0.0001, 0.0001, w)

params_dict = dict(
	b=['edge', 'birth'],
	d=['edge', 'sample'],
	s=['edge', 'sample'],
	rho=['edge', 'csa'],
	gamma=['edge'],
	)

values_dict = dict(
	time_step=['edge'],
	back_time=['edge'],
	param_interval=['edge', 'birth', 'sample', 'csa'],
	)

def get_bioproject_times_list(bioproject_times_file, changepoints_out_file):
	"""
	Just get list of all bioproject starts and ends, output to file
	Can put in config
	"""
	bioproject_times = pd.read_csv(bioproject_times_file, index_col=0)

	# Output a list of the sampling changepoints
	sampling_changepoints = list(set(bioproject_times["min_time"].to_list() + bioproject_times["max_time"].to_list()))
	Path(changepoints_out_file).write_text(",".join([str(s) for s in sampling_changepoints]))

def make_sampling_mask(bioproject_ancestral_file, bioproject_times_file, interval_times_file, mask_out_file, uncertainty=True):
	"""
	Given data object, set sampling upon removal rate for each
	phylogeny segment based on its bioproject
	"""

	# Read in interval times, but remove last
	interval_times = [float(t) for t in Path(interval_times_file).read_text().splitlines()]

	# Read in CSV of bioproject times
	bioproject_times = pd.read_csv(bioproject_times_file, index_col=0)
	bioproject_times = bioproject_times[["min_time", "max_time"]]
	bioproject_times.index = [i.split("_")[0] for i in bioproject_times.index]
	bioprojects = bioproject_times.index.to_list()

	# Get reconstructed bioproject feature states
	bp_anc = pd.read_csv(bioproject_ancestral_file, sep="\t")[['node', 'bioproject_id_META']]
	bp_anc = bp_anc.dropna(subset="bioproject_id_META")

	n_obs = len(bp_anc['node'].unique())
	n_int = len(interval_times)

	# Initialize matrix of zeroes with rows for each sample, column for each parameter interval
	s_arr = np.zeros((n_obs, n_int), dtype=float)

	# For each sample, figure out window in which sampling could have occurred based on the
	# first and last sample from that bioproject. 

	# If uncertain=True, we also take into account ancestral uncertainty in bioproject. 
	# Otherwise, we just use the first listed ancestral state
	if not uncertainty:
		bp_anc = bp_anc.loc[bp_anc.index.drop_duplicates(keep="first"), :]

	nodes = []
	for i, (sample, sdf) in enumerate(bp_anc.groupby('node')):
		nodes.append(sample)

		# sdf is a dataframe of all potential bioprojects, which we weight as equally likely
		prob = 1 / len(sdf)

		# For each potential bioproject, add marginal probability to time intervals that occur
		# during the window in which that project's sampling was taking place
		for r, row in sdf.iterrows():

			# Determine bioproject for sample
			proj = row['bioproject_id_META']

			# Get start and end of bioproject sampling
			if proj in bioproject_times.index:
				start, end = bioproject_times.loc[proj, :].apply(lambda x: interval_times.index(x)).values
				
				# Add to the mask
				s_arr[i, start:end] += prob

	s_df = pd.DataFrame(s_arr, columns=interval_times, index=nodes)
	s_df = s_df.drop(columns=[interval_times[-1]])
	s_df.to_csv(mask_out_file)

	# print(s_df.loc['root'])
	# root_projs = bp_anc.loc[bp_anc['node'] == 'root', 'bioproject_id_META']
	# print(bioproject_times.loc[root_projs])

class BirthSamplingSite(KerasModel):
	"""
	Sampling rate is parameterized by feature-specific effects and a time-based background rate. 
	A "sampling mask" uses this rate when a sample's bioproject is actively sampling, and sets it to 0 otherwise.

	Birth rate is parameterized by feature-specific effects.

	Death rate is fixed. There are no rho sampling events or migration (gamma).
	"""

	override_vars = ['s', 'b']

	def __init__(self, 
			data, fit_model_params,
			iterative_pE, rho=0, gamma=0, **kwargs):

		super().__init__(**kwargs)

		self.data = data
		self.iterative_pE = iterative_pE

		# Stores all the tensors needed to calculate the likelihood
		self.p = SimpleNamespace()

		# Get sub-arrays corresponding to each event type
		arr = data.array
		self.edge_arr = arr[arr['event'] == 4]
		self.birth_arr = arr[arr['event'] == 1]
		self.sample_arr = arr[arr['event'] == 2]
		self.csa_arr = arr[arr['event'] == 3]
		self.pE_arr = self.edge_arr

		# Info from sub arrays and data that is useful to 
		# create the tensors we need
		self.n_edges = len(self.edge_arr)
		self.n_births = len(self.birth_arr)
		self.n_samples = len(self.sample_arr)
		self.n_csas = len(self.csa_arr)
		self.n_times = self.data.n_times
		self.n_components = len(arr)
		
		self.init_time_values()
		self.init_constant_params(d=fit_model_params["death_rate"], rho=fit_model_params["rho"], gamma=fig_model_params["gamma"])
		self.init_variable_params(
			birth_background=fit_model_params["birth_background"], 
			birth_features=fit_model_params["birth_features"], 
			sampling_features=fit_model_params["sampling_features"], 
			sampling_background=fit_model_params["sampling_background"], 
			sampling_mask=fit_model_params["sampling_mask"], 
			brownian_motion=fit_model_params["brownian_motion"]
			)
		self.init_branch_effects(fit_model_params["branch_effects"])

		self.model_variables = [v.name.split(":")[0] for v in self.trainable_variables]
		self.penalize = [v for v in self.model_variables if fit_model_kwargs[v]["penalize"] == True]

	def init_constant_params(self, d, rho, gamma):
		for param, value in dict(d=d, rho=rho, gamma=gamma).items():
			
			array_types = params_dict[param]

			if self.iterative_pE:
				array_types += ['pE']

			for array_type in array_types:
				if array_type == 'pE':
					shape = (self.n_edges, self.n_times)
				else:
					shape = (getattr(self, f"n_{array_type}s"),)
				
				setattr(self.p, f"{array_type}_{param}", tf.constant(value, shape=shape, dtype=tf.dtypes.float64))

	def init_birth_background(self, birth_background):
		self.birth_background_idx = tf.constant(birth_background["interval_mapping"], dtype=tf.dtypes.int64)
		self.birth_background = tf.constant(birth_background['value'], dtype=tf.dtypes.float64)

		if birth_background['estimate']:
			self.birth_background = tf.Variable(self.birth_background, name="birth_background", constraint=PosNonZero())

	def init_sampling_prob(self, sampling_prob):
		self.sampling_prob_idx = tf.constant(sampling_prob["interval_mapping"], dtype=tf.dtypes.int64)
		self.sampling_prob = tf.constant(sampling_prob['value'], dtype=tf.dtypes.float64)

		if sampling_prob['estimate']:
			self.sampling_prob = tf.Variable(self.sampling_prob, name="sampling_background", constraint=PosNonZero())
		
	def init_branch_effects(self, branch_effects):
		branch_names = [n.split("_interval")[0] for n in self.data.array['name']]

		# Create tensor that stores value of individual branch effects
		self.branch_effects = tf.constant(branch_effects['value'], shape=(branch_effects['n_branches'],), dtype=tf.dtypes.float64)
		if branch_effects['estimate']: 
			self.branch_effects = tf.Variable(self.branch_effects, name="branch_effects", constraint=PosNonZero())

		# Add tensors that specify which branch number each segment is
		array_types =  params_dict['b'] + ['pE'] if self.iterative_pE else params_dict['b']
		for array_type in array_types:
			arr = getattr(self, f"{array_type}_arr")
			arr_branch_types = [branch_effects['branch_dict'][n.split('_interval')[0]] for n in arr['name']]
			setattr(self, f"{array_type}_branch_int", tf.constant(arr_branch_types, tf.dtypes.int64))

	def init_features(self, super_param, feature_type, feature_dict):
		"""
		feature_dict has keys 'estimated', 'states', and 'value'
		"""

		states = pd.read_csv(Path(feature_dict['states']), index_col=0)

		# Set tensor of coeff corresponding to effect of each feature
		n_features = states.shape[1]
		coeff = tf.constant(feature_dict['value'], shape=[1, n_features], dtype=tf.dtypes.float64)

		if feature_dict['estimate']:
			coeff = tf.Variable(coeff, name=feature_type, constraint=PosNonZero())

		setattr(self, f"{feature_type}_coeff", coeff)

		array_types = params_dict[super_param]

		if self.iterative_pE:
			array_types += ['pE']

		for array_type in array_types:
			arr = getattr(self, f"{array_type}_arr")

			# Set tensor of presence/absence/marginal states
			sample_names = np.array([n.split("_interval")[0] for n in arr['name']])
			features = tf.constant(states.loc[sample_names, :].values, dtype=tf.dtypes.float64)
			setattr(self, f"{array_type}_{feature_type}", features)

	def init_mask(self, sampling_mask):
		sampling_mask = pd.read_csv(sampling_mask, index_col=0)

		for array_type in ['edge', 'sample', 'pE']:
			arr = getattr(self, f"{array_type}_arr")
			sample_names = [n.split("_interval")[0] for n in arr['name']]
			sample_mask = sampling_mask.loc[sample_names, :].values

			if array_type != 'pE':
				sample_mask = np.choose(arr['param_interval'], sample_mask.T)

			setattr(self, f"{array_type}_sampling_mask", tf.constant(sample_mask, dtype=tf.dtypes.float64))

	def init_brownian(self, brownian_motion):
		df = pd.DataFrame.from_dict(brownian_motion["info"], orient="index")
		df.index = [int(i) for i in df.index]

		n_types = brownian_motion["n_types"]
		
		edge_df = df.loc[self.edge_arr['abs_index'], :]
		birth_df = df.loc[self.birth_arr['abs_index'], :]

		self.p.edge_type_int = tf.constant(edge_df['type_int'], dtype=tf.dtypes.int64)
		self.p.birth_type_int = tf.constant(birth_df['type_int'], dtype=tf.dtypes.int64)

		# These are just for calculating the penalty
		self.p.edge_parent_type_int = tf.constant(edge_df['parent_type_int'], dtype=tf.dtypes.int64)
		self.p.edge_parent_time_delta = tf.constant(edge_df['parent_time_delta'], dtype=tf.dtypes.float64)

		self.brownian_eff = tf.constant(brownian_motion['value'], shape=[n_types], dtype=tf.dtypes.float64)
		
		if brownian_motion['estimate']:
			self.brownian_eff = tf.Variable(self.brownian_eff, name="brownian_motion")

	def init_variable_params(self, birth_background: dict, birth_features: dict, sampling_features: dict, sampling_background: dict, sampling_mask: pd.DataFrame, brownian_motion: dict):
		# Init tensor to estimate or set birth rate
		self.init_birth_background(birth_background)

		# Init tensors to estimate or set features that influence birth and sampling rates
		self.init_features(super_param="b", feature_type="birth_features", feature_dict=birth_features)
		self.init_features(super_param="s", feature_type="sampling_features", feature_dict=sampling_features)

		# Init tensor to estimate background sampling probability
		self.init_sampling_prob(sampling_background)

		# Init sampling mask
		self.init_mask(sampling_mask)

		# Init brownian motion
		self.init_brownian(brownian_motion)

	def init_time_values(self):
		# Creates self.p.edge_param_interval, self.p.birth_oaram_interval, self.p.edge_back_time, etc.
		for value, array_types in values_dict.items():
			for array_type in array_types:
				arr = getattr(self, f"{array_type}_arr")
				dtype = tf.dtypes.int64 if 'interval' in value else tf.dtypes.float64
				setattr(self.p, f"{array_type}_{value}", tf.constant(arr[value], dtype=dtype))

		# If doing iterative pE,
		# creates self.p.edge_pE_init_time, etc. which we have to do a little differently than the above
		if self.iterative_pE:
			self.init_pE_values()

	def init_pE_values(self):
		back_times = self.data.bkwd_interval_times

		# End time of each time interval, going backwards in time (farthest from present)
		self.p.pE_back_times = tf.constant(back_times, shape=self.n_times, dtype=tf.dtypes.float64)
		
		# Indices of time intervals, when they are going backwards in time
		self.p.pE_back_idxs = tf.constant(np.arange(self.n_times), shape=self.n_times, dtype=tf.dtypes.int64)

		# Add 0 to back times to account for the fact that rho is associated with the end of a time interval
		# (the beginning in backward time) ...
		back_times = np.append(back_times, 0)

		# Get beginning of time interval (closest to present, in backwards time) that each sample is in
		edge_pE_interval = self.edge_arr['pE_interval']
		self.p.edge_pE_init_time = tf.constant(np.take(back_times, edge_pE_interval), dtype=tf.dtypes.float64)

		# To use tensorflow function declaration, we can't iterate across a tensor in the way we want, 
		# so need to do this... create an n x 2 tensor where first column is index of sample and second column is pE interval
		edge_idxs = np.arange(0, self.n_edges)
		self.p.edge_pE_interval = tf.constant(np.vstack([edge_idxs, edge_pE_interval]).T, dtype=tf.dtypes.int64)

		self.pE_ones = tf.ones(shape=[self.n_edges, self.n_times], dtype=tf.dtypes.float64)

		# self.n_betas = len(tf.unique(self.birth_background_idx)[0])
		# self.b0 = tf.Variable(tf.ones(shape=self.n_betas, dtype=tf.dtypes.float64) + .000001, name='b0')

	def calc_sampling(self):
		"""
		Calculate total sampling probability for edges, birth events, and e probs
		"""

		# Expand background sampling rates to use correct rate for each time interval
		# The quicker way to do this would be to take out this expand_background middleman and just init
		# with the correct background sampling index for each component
		expand_background = tf.gather(self.sampling_prob, self.sampling_prob_idx)

		# Calc background sampling rate tensors
		edge_background = tf.gather(expand_background, self.p.edge_param_interval)
		sample_background = tf.gather(expand_background, self.p.sample_param_interval)

		# Calc site effects
		log_coeffs = tf.math.log(self.sampling_features_coeff)
		edge_site = tf.squeeze(tf.exp(tf.matmul(self.edge_sampling_features, tf.transpose(log_coeffs))))
		sample_site = tf.squeeze(tf.exp(tf.matmul(self.sample_sampling_features, tf.transpose(log_coeffs))))

		if self.iterative_pE:
			pE_site = tf.exp(tf.matmul(self.edge_sampling_features, tf.transpose(log_coeffs))) * self.pE_sampling_mask

		self.p.edge_s = edge_background * edge_site * self.edge_sampling_mask
		self.p.sample_s = sample_background * sample_site * self.sample_sampling_mask
		self.p.pE_s = tf.reshape(edge_site, [-1, 1]) * self.pE_sampling_mask * expand_background

	def calc_birth(self):
		# Calc background birth rate
		# --------------------------------------
		# Expand background sampling rates to use correct rate for each time interval
		# The quicker way to do this would be to take out this expand_background middleman and just init
		# with the correct background sampling index for each component
		expand_background = tf.gather(self.birth_background, self.birth_background_idx)
		edge_background = tf.gather(expand_background, self.p.edge_param_interval)
		birth_background = tf.gather(expand_background, self.p.birth_param_interval)

		# Calc site fitness
		# --------------------------------------
		log_coeffs = tf.math.log(self.birth_features_coeff)
		edge_site_b = tf.exp(tf.matmul(self.edge_birth_features, tf.transpose(log_coeffs)))
		birth_site_b = tf.exp(tf.matmul(self.birth_birth_features, tf.transpose(log_coeffs)))
		if self.iterative_pE:
			pE_site_b = tf.exp(tf.matmul(self.edge_birth_features, tf.transpose(log_coeffs)))

		# Calc branch effects
		# --------------------------------------
		edge_branch_effects = tf.gather(self.branch_effects, self.edge_branch_int)
		birth_branch_effects = tf.gather(self.branch_effects, self.birth_branch_int)
		if self.iterative_pE:
			pE_branch_effects = tf.reshape(tf.gather(self.branch_effects, self.pE_branch_int), shape=(-1, 1))

		# Calc brownian fitness
		# --------------------------------------
		self.p.brownian_eff = self.brownian_eff
		edge_brown = tf.gather(self.brownian_eff, self.p.edge_type_int)
		birth_brown = tf.gather(self.brownian_eff, self.p.birth_type_int)
		if self.iterative_pE:
			pE_brown = tf.reshape(edge_brown, (-1, 1))

		# Multiply to get total birth rate
		# ======================================
		self.p.edge_b = tf.squeeze(edge_site_b) * edge_branch_effects * edge_brown * edge_background
		self.p.birth_b = tf.squeeze(birth_site_b) * birth_branch_effects * birth_brown * birth_background 

		if self.iterative_pE:
			self.p.pE_b = pE_site_b * pE_branch_effects * pE_brown * expand_background

	def call(self):
		self.calc_birth()
		self.calc_sampling()
		return self.p

def make_subtree():
	clades = pd.read_csv(Path("data") / "clade_combined_ancestral_states.csv", index_col=0)
	clades = clades.loc[[i for i in clades.index if 'SAMN' in i], 'Clade']
	clades = clades[clades!="A"]

	subtree = pp.removeLeaves(Path("data_new") / "lsd-tree_grouped-features" / "tree_named.tree_lsd.date.noref.pruned_unannotated.nwk", clades.index.to_list())
	subtree.write(format=1, outfile="data_new/a_test.nwk")

def to_single_state(pastml_dir):
	clades = pd.read_csv(Path("data") / "clade_combined_ancestral_states.csv", index_col=0)
	clades = clades.loc[:, 'Clade']
	clades = clades[clades!="A"]

	pastml_dir = Path(pastml_dir)
	comb = pd.read_csv(pastml_dir / "combined_ancestral_states.csv", index_col=0)
	comb = comb.dropna(how="any", axis=0)
	
	ct = comb.sum(axis=1)
	comb = comb.loc[:, comb.sum(axis=0) > 0].astype(int)
	comb.to_csv(pastml_dir / "a_test_binary.csv")

if __name__ == "__main__":
	config = load(Path("config.yaml").read_text(), Loader=Loader)

	# Toy Example Stuff
	# =============
	if False:
		make_subtree()

	if False:
		to_single_state("data_new/lsd-tree_grouped-features/")

	if False:
		marginal_file = "data_new/lsd-tree_grouped-features/work/marginal_states.csv"
		load_data_test("data_new/a_test.nwk", "data_new/lsd-tree_grouped-features/a_test_binary.csv", marginal_file)
 
	if False:
		make_sampling_mask(
			bioproject_ancestral_file="data_new/old_combined_ancestral_states.tab", 
			bioproject_times_file="data/bioproject_times.csv",
			interval_times_file="data/interval_trees/2003-2013-bioprojsampling/interval_times.txt", 
			mask_out_file="data_new/sampling_mask_binary.csv", 
			uncertainty=False
			)

	if False:
		get_bioproject_times_list(bioproject_times_file="data/bioproject_times.csv", changepoints_out_file="data_new/bp_changepoints.txt")

	if False:
		interval_times, interval_tree = make_intervals(
			"data_new/4_interval_tree",
			"data/named.tree_lsd.date.noref.pruned_unannotated.nwk",
			config['last_sample_date'],
			config['sampling_prob_changepoints'],
			config['bioproject_changepoints'],
			)

	if False:
		config['birth_features_file'] = "data_new/functional_groups_corr_pastml/marginal_states.csv"
		config['sampling_features_file'] = "data_new/meta_features_marginal.csv"
		config['sampling_mask_file'] = "data_new/sampling_mask.csv"

		test_fitness_model(
			tree_file="data_new/4_interval_tree/phylo.nwk",
			interval_times_file="data_new/4_interval_tree/interval_times.txt",
			last_sample_date=config['last_sample_date'],
			config=config,
			)

