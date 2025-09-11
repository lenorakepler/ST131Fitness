from pathlib import Path
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import Model as KerasModel
from types import SimpleNamespace

class PosNonZero(tf.keras.constraints.Constraint):
	def __call__(self, w):
		return tf.where(w < 0.0001, 0.0001, w)

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
			sampling_mask=fit_model_params["sampling_mask"], 
			brownian_motion=fit_model_params["brownian_motion"]
			)

		self.model_variables = [v.name.split(":")[0] for v in self.trainable_variables]
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
		feature_dict has keys 'estimated', 'states', and 'value'
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

		if self.save_intermediate:
			self.s_expand_background = expand_background
			self.s_edge_background = edge_background
			self.s_sample_background = sample_background
			self.s_log_coeffs = log_coeffs
			self.s_edge_site = edge_site
			self.s_sample_site = sample_site

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

		if self.iterative_pE:
			self.p.pE_b = pE_site_b * pE_brown * expand_background

		if self.save_intermediate:
			self.b_expand_background = expand_background
			self.b_edge_background = edge_background
			self.b_birth_background = birth_background
			self.b_log_coeffs = log_coeffs
			self.edge_site_b = edge_site_b
			self.birth_site_b = birth_site_b
			self.edge_brown = edge_brown
			self.birth_brown = birth_brown

	def call(self):
		self.calc_birth()
		self.calc_sampling()
		return self.p

