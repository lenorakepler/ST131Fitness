import json
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from transmission_sim.analysis.param_model import ParamComponent, ParamModel, ComponentSite
from transmission_sim.analysis.phylo_loss import PhyloLossIterative, conditional_decorator, use_graph_execution
from transmission_sim.analysis.arrayer import PhyloArrayer, PhyloData
from transmission_sim.analysis.phylo_obj import PhyloObj
from transmission_sim.analysis.optimizer import Optimizer
from transmission_sim.ecoli.plot_test_reg import get_true_effs
from transmission_sim.ecoli.results_obj import ComponentB02
import transmission_sim.analysis.phylo_loss

transmission_sim.analysis.phylo_loss.use_graph_execution = True

class ComponentRandomEffect(ParamComponent):
	def init_random_effect(self):
		# Get variables/constants for random site effects
		# Not possible to have time-specific effects
		random_effect_est, _ = list(self.bdm['random_effect'])

		# If estimating branch/node-specific random fitness effects
		if random_effect_est:
			self.p.edge_type_int = tf.constant(self.edge_arr['type_int'], dtype=tf.dtypes.int64)
			self.p.birth_type_int = tf.constant(self.birth_arr['type_int'], dtype=tf.dtypes.int64)

			# These are just for calculating the penalty
			self.p.edge_parent_type_int = tf.constant(self.edge_arr['parent_type_int'], dtype=tf.dtypes.int64)
			self.p.edge_parent_time_delta = tf.constant(self.edge_arr['parent_time_delta'], dtype=tf.dtypes.float64)

			self.rand_eff = tf.Variable(
				tf.ones(shape=[self.n_types], dtype=tf.dtypes.float64),
				name='rand_eff'
			)

		# If not estimating, need to grab
		else:
			self.rand_eff = self.rand_eff

			if self.iterative_pE:
				self.pE_rand_eff = self.rand_eff

	def get_random_effectVar(self):
		self.edge_rand_eff = tf.gather(self.rand_eff, self.p.edge_type_int)
		self.birth_rand_eff = tf.gather(self.rand_eff, self.p.birth_type_int)

	def get_random_effectVar_pE(self):
		self.edge_rand_eff = tf.gather(self.rand_eff, self.p.edge_type_int)
		self.birth_rand_eff = tf.gather(self.rand_eff, self.p.birth_type_int)
		self.pE_rand_eff = tf.reshape(self.edge_rand_eff, (-1, 1))

class RandomEffectSite(ParamModel, ComponentRandomEffect, ComponentB02, ComponentSite):
	def __init__(self, n_types, birth_rate_idx, data, random_effect, site, b0, d, s, rho, gamma, **kwargs):

		self.birth_rate_idx = tf.constant(birth_rate_idx)
		self.n_types = n_types

		self.bdm = dict(
			random_effect=random_effect, b0=b0, site=site,
		    d=d, s=s, rho=rho, gamma=gamma,
		)

		self.private_vars = [k for k, v in self.bdm.items() if v[0] == True]
		self.vars = ['b'] + self.private_vars

		super().__init__(data=data, iterative_pE=data.iterative_pE, **kwargs)

		# Set parameter methods
		self.setMethods()

		# Initialize parameters
		for var in self.bdm:
			if var not in self.base_bdm_params:
				getattr(self, f"init_{var}")()

		self.phylo_loss = PhyloLossRandomEffIterative if self.iterative_pE else PhyloLossRandomEffNonIterative

	def call_(self):
		for var in self.private_vars:
			getattr(self, f"{var}_call")()

		self.p.edge_b = self.edge_rand_eff * self.edge_b0 * self.edge_site_b
		self.p.birth_b = self.birth_rand_eff * self.birth_b0 * self.birth_site_b
		self.p.edge_rand_eff = self.edge_rand_eff
		self.p.rand_eff = self.rand_eff

		return self.p

	def call_PE(self):
		for var in self.private_vars:
			getattr(self, f"{var}_call")()

		self.p.edge_b = self.edge_rand_eff * self.edge_b0 * self.edge_site_b
		self.p.birth_b = self.birth_rand_eff * self.birth_b0 * self.birth_site_b
		self.p.edge_rand_eff = self.edge_rand_eff
		self.p.rand_eff = self.rand_eff
		self.p.pE_b = self.pE_rand_eff * self.pE_b0 * tf.reshape(self.pE_site_b, [-1, 1])

		return self.p

class SigmaMixin():
	@conditional_decorator(tf.function, lambda: transmission_sim.analysis.phylo_loss.use_graph_execution)
	def Sigma(self, c, **kwargs):

		sigma = self.sigma
		self.i += 1

		loss = self.call_(c)

		# tf.print({name: type(t) for name, t in c.items()})

		if sigma != 0:
			epsilon = 0.0000005
			fit_shifts = c["edge_rand_eff"] - tf.gather(c["rand_eff"], c["edge_parent_type_int"])
			times = c["edge_parent_time_delta"]

			probs = tf.clip_by_value(tf.math.exp(-0.5 * fit_shifts**2 / (sigma * times + epsilon)), epsilon, np.inf) # variance is proportional to time * sigma
			penalty = tf.reduce_sum(tf.math.log(probs)) # Sum log prob values

			# Penalty term will always be negative: the more negative,
			# the farther we are from optimal. Because our loss is a
			# negative log likelihood that we MINIMIZE,
			# the added penalty needs to be LARGER the farther we are from
			# optimal. So, multiply by -1
			penalty = penalty * -1

		else:
			penalty = tf.constant(0, dtype=tf.dtypes.float64)

		# if (self.i % 100 == 0):
		# 	with np.printoptions(precision=4):
		# 		print(f"{self.i}: loss={loss.numpy()}, penalty={penalty.numpy()}, total={loss.numpy() + penalty.numpy()}")
		# 		print(c["rand_eff"][0:10].numpy())

		return loss + penalty

class PhyloLossRandomEffIterative(PhyloLossIterative, SigmaMixin):
	pass
