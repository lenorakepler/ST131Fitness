from pathlib import Path
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import Model as KerasModel
from types import SimpleNamespace

class PosNonZero(tf.keras.constraints.Constraint):
	def __call__(self, w):
		return tf.where(w < 0.000001, 0.000001, w)

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
			iterative_pE, rho=0, gamma=0, 
			save_intermediate=False,
			**kwargs):

		super().__init__(**kwargs)

		self.data = data
		self.iterative_pE = iterative_pE
		# self.fit_model_params = fit_model_params

		# Stores all the tensors needed to calculate the likelihood
		self.p = SimpleNamespace()

		# Get sub-arrays corresponding to each event type
		# TODO: Does this need to be numpy? 
		# It makes everything so much harder.
		# TODO: Would it be easier to calculate everything in one big array
		# and then extract the sub-arrays at each epoch
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

		self.params_dict = dict(
			b=['edge', 'birth'],
			d=['edge', 'sample'],
			s=['edge', 'sample'],
			rho=['edge', 'csa'],
			gamma=['edge'],
			)

		self.values_dict = dict(
			time_step=['edge'],
			back_time=['edge'],
			param_interval=['edge', 'birth', 'sample', 'csa'],
			)

		if self.iterative_pE:
			for param, param_arrays in self.params_dict.items():
				param_arrays.append('pE')
		
		self.init_time_values()
		self.init_constant_params(d=fit_model_params["death_rate"], rho=fit_model_params["rho"], gamma=fit_model_params["gamma"])
		self.init_variable_params(
			birth_background=fit_model_params["birth_background"], 
			birth_features=fit_model_params["birth_features"], 
			sampling_features=fit_model_params["sampling_features"],
			sampling_background=fit_model_params["sampling_background"],
			brownian_motion=fit_model_params["brownian_motion"]
			)

		self.model_variables = [v.name.split(":")[0] for v in self.trainable_variables]

		# List of coefficients to shrinkwith L1/L2 regularization
		self.penalize = [v.name for v in self.trainable_variables if fit_model_params[v.name.split(":")[0]]["penalize"] == True]

		# For debugging or calculating component fitness
		self.save_intermediate = save_intermediate

	def init_constant_params(self, d, rho, gamma):
		for param, value in dict(d=d, rho=rho, gamma=gamma).items():
			
			for array_type in self.params_dict[param]:
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
		
	def init_features(self, super_param, feature_type, feature_dict):
		"""
		feature_dict has keys 'estimated' and 'value', plus
		'states', which has a list of binary 
		"""

		states = pd.read_csv(Path(feature_dict['states']), index_col=0)

		# Set tensor of coeff corresponding to effect of each feature
		n_features = states.shape[1]
		coeff = tf.constant(feature_dict['value'], shape=[1, n_features], dtype=tf.dtypes.float64)

		if feature_dict['estimate']:
			coeff = tf.Variable(coeff, name=feature_type, constraint=PosNonZero())

		setattr(self, f"{feature_type}_coeff", coeff)

		for array_type in self.params_dict[super_param]:
			arr = getattr(self, f"{array_type}_arr")

			# Set tensor of presence/absence/marginal states
			sample_names = np.array([n.split("_interval")[0] for n in arr['name']])
			features = tf.constant(states.loc[sample_names, :].values, dtype=tf.dtypes.float64) 
			setattr(self, f"{array_type}_{feature_type}", features)

	def init_brownian(self, brownian_motion):
		# TODO: where do I add this "info" to brownian_motion? 
		# It's not in the actual json file. Why?
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
			self.brownian_eff = tf.Variable(self.brownian_eff, name="brownian_motion", constraint=PosNonZero())

	def init_variable_params(self, birth_background: dict, birth_features: dict, sampling_features: dict, sampling_background: dict, brownian_motion: dict):
		# Init tensor to estimate or set birth rate
		self.init_birth_background(birth_background)

		# Init tensors to estimate or set features that influence birth and sampling rates
		self.init_features(super_param="b", feature_type="birth_features", feature_dict=birth_features)
		# self.init_features(super_param="s", feature_type="sampling_features", feature_dict=sampling_features)

		# Init tensor to estimate background sampling probability
		self.init_sampling_prob(sampling_background)

		# Init sampling mask
		# self.init_mask(sampling_mask)
		self.init_sampling(sampling_features)

		# Init brownian motion
		self.init_brownian(brownian_motion)

	def init_time_values(self):
		# Creates self.p.edge_param_interval, self.p.birth_oaram_interval, self.p.edge_back_time, etc.
		for value, array_types in self.values_dict.items():
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

	def init_sampling(self, sampling_features):
		super_param = "s"

		for feature_type, feature_dict in sampling_features["variables"].items():
			
			states = pd.read_csv(feature_dict["states"], index_col=0)

			if "mask" in feature_dict:
				mask = pd.read_csv(feature_dict["mask"], index_col=0)
				setattr(self, f"{feature_type}_mask", mask)
				
				# make sure state feature order matches mask order
				assert len(mask.index) == len(states.columns)
				states = states.loc[:, mask.index]

			setattr(self, f"{feature_type}_states", states)

			for array_type in self.params_dict[super_param]:
				arr = getattr(self, f"{array_type}_arr")

				# Set tensor of presence/absence/marginal states
				sample_names = np.array([n.split("_interval")[0] for n in arr['name']])

				# TODO-ADDRESSED: with sampling, we actually want to use the _interval one? 
				# or do we? so we have sample x bp x time?
				# No, because all _interval have the same features.. maybe should have done pastML on 
				# interval tree instead, but here we are.
				features = tf.constant(states.loc[sample_names, :].values, dtype=tf.dtypes.float64)
				setattr(self, f"{array_type}_{feature_type}_prob", features)

			coeff = tf.constant(feature_dict['value'], shape=[1, len(states.columns)], dtype=tf.dtypes.float64)

			if feature_dict['estimate']:
				coeff = tf.Variable(coeff, name=feature_type, constraint=PosNonZero())

			setattr(self, f"{feature_type}_coeff", coeff)

	def print_shape(self, var_name, var):
		print(f"{var_name}: {tf.shape(var)}")

	def calc_sampling(self):
		"""
		Calculate total sampling probability for edges, birth events, and e probs
		"""

		# Background effects
		# --------------------------------------------------------------------
		# Expand background sampling rates to use correct rate for each time interval
		# The quicker way to do this would be to take out this expand_background middleman and just init
		# with the correct background sampling index for each component
		expand_background = tf.gather(self.sampling_prob, self.sampling_prob_idx)
		# self.print_shape("expand_background", expand_background)

		# Calc background sampling rate tensors
		edge_background = tf.gather(expand_background, self.p.edge_param_interval)
		sample_background = tf.gather(expand_background, self.p.sample_param_interval)

		# print(f"--------------------------")
		
		# self.print_shape("sample_background", sample_background)

		# Calc site effects

		# Bioproject effects
		# --------------------------------------------------------------------
		# masked_coeff = tf.math.pow(self.bioproject_coeff, tf.transpose(self.bioproject_mask))
		# bioproject_edge_by_time = tf.reduce_prod(tf.math.pow(tf.expand_dims(masked_coeff, 0), tf.expand_dims(self.edge_bioproject_prob, 1)), axis=-1)
		# bioproject_sample_by_time = tf.reduce_prod(tf.math.pow(tf.expand_dims(masked_coeff, 0), tf.expand_dims(self.sample_bioproject_prob, 1)), axis=-1)
		
		log_masked_coeff = tf.math.log(tf.transpose(self.bioproject_coeff)) * self.bioproject_mask
		bioproject_edge_by_time = tf.exp(tf.matmul(self.edge_bioproject_prob, log_masked_coeff))
		bioproject_sample_by_time = tf.exp(tf.matmul(self.sample_bioproject_prob, log_masked_coeff))
		# self.print_shape("log_masked_coeff", log_masked_coeff)
		# self.print_shape("bioproject_edge_by_time", bioproject_edge_by_time)

		# Select effect of interval that component occurs in
		bioproject_edge_effect = tf.gather(bioproject_edge_by_time, tf.transpose(self.p.edge_param_interval), batch_dims=1)
		bioproject_sample_effect = tf.gather(bioproject_sample_by_time, tf.transpose(self.p.sample_param_interval), batch_dims=1)

		# self.print_shape("bioproject_edge_effect", bioproject_edge_effect)

		# Do for pE matrix
		if self.iterative_pE:
			bioproject_pE_effects = bioproject_edge_by_time

		

		# Specimen type effect
		# --------------------------------------------------------------------
		# Matrix multiply so that we have, for each phylo component, 
		# the sum of the marginal probability of the component being in 
		# each bioproject times the effect of the bioproject at that time.
		specimen_edge_effect = tf.math.pow(self.specimen_type_coeff, self.edge_specimen_type_prob)
		specimen_sample_effect = tf.math.pow(self.specimen_type_coeff, self.sample_specimen_type_prob)

		# self.print_shape("specimen_edge_effect", specimen_edge_effect)
		# self.print_shape("specimen_type_coeff", self.specimen_type_coeff)
		# self.print_shape("edge_specimen_type_prob", self.edge_specimen_type_prob)

		# Do for pE matrix
		if self.iterative_pE:
			specimen_pE_effect = tf.math.pow(self.specimen_type_coeff, self.pE_specimen_type_prob)

		# self.print_shape("pE_specimen_type_prob", self.pE_specimen_type_prob)
		

		# Calculate total sampling probability
		# --------------------------------------------------------------------
		expand_background = tf.tile(tf.reshape(expand_background, [1, -1]), tf.constant([self.n_edges, 1]))

		# TODO: I should not have any one-dimensional tensors, all need to have explicit dimensions
		self.p.edge_s = edge_background * bioproject_edge_effect * tf.squeeze(specimen_edge_effect)
		self.p.sample_s = sample_background * bioproject_sample_effect * tf.squeeze(specimen_sample_effect)
		self.p.pE_s =  expand_background * bioproject_pE_effects * specimen_pE_effect

	

		# self.print_shape("expand_background", expand_background)
		# self.print_shape("bioproject_pE_effects", bioproject_pE_effects)
		# self.print_shape("specimen_pE_effect", specimen_pE_effect)
		# print("-----")
		# self.print_shape("sample_background * bioproject_sample_effect", sample_background * bioproject_sample_effect)
		# self.print_shape("bioproject_sample_effect * specimen_sample_effect", bioproject_sample_effect * specimen_sample_effect)

		# self.print_shape("expand_background * bioproject_pE_effects", expand_background * bioproject_pE_effects)
		

		# print(f"edge_s: {tf.shape(self.p.edge_s)}")
		# print(f"--------------------------")

		# if self.save_intermediate:
		# 	self.s_expand_background = expand_background
		# 	self.s_edge_background = edge_background
		# 	self.s_sample_background = sample_background
		# 	self.bioproject_edge_by_time = bioproject_edge_by_time
		# 	self.specimen_edge_effect = specimen_edge_effect
			
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

		# Calc brownian fitness
		# --------------------------------------
		self.p.brownian_eff = self.brownian_eff
		edge_brown = tf.gather(self.brownian_eff, self.p.edge_type_int)
		birth_brown = tf.gather(self.brownian_eff, self.p.birth_type_int)
		if self.iterative_pE:
			pE_brown = tf.reshape(edge_brown, (-1, 1))

		# Multiply to get total birth rate
		# ======================================
		self.p.edge_b = tf.squeeze(edge_site_b) * edge_brown * edge_background
		self.p.birth_b = tf.squeeze(birth_site_b) * birth_brown * birth_background

		# if not tf.reduce_all(self.p.birth_b > 0):
		# 	print(np.argwhere(self.p.birth_b < 0))
		# 	breakpoint()

		if self.iterative_pE:
			self.p.pE_b = pE_site_b * pE_brown * expand_background

			# print("~~~~~~~")
			# self.print_shape("pE_site_b", pE_site_b)
			# self.print_shape("pE_brown", pE_brown)
			# self.print_shape("expand_background", expand_background)
			# self.print_shape("self.p.pE_b", self.p.pE_b)
			# print("~~~~~~~")

		# if self.save_intermediate:
		# 	self.b_expand_background = expand_background
		# 	self.b_edge_background = edge_background
		# 	self.b_birth_background = birth_background
		# 	self.b_log_coeffs = log_coeffs
		# 	self.edge_site_b = edge_site_b
		# 	self.birth_site_b = birth_site_b
		# 	self.edge_brown = edge_brown
		# 	self.birth_brown = birth_brown

	def call(self):
		self.calc_birth()
		self.calc_sampling()
		return self.p

