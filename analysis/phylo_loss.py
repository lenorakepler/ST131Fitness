import tensorflow as tf
import numpy as np

class PhyloLoss(tf.keras.losses.Loss):
	def __init__(self, **kwargs):
		super().__init__()
		self.graph = getattr(kwargs, 'graph', True)

		for k, v in kwargs.items():
			setattr(self, k, v)

		if getattr(self, 'reg_type', False):

			# Use 0 offset unless otherwise specified
			offset = getattr(self, 'offset', 0)

			if self.reg_type.lower() == 'l1':
				if offset == 0:
					self.call = self.L1_0
				elif offset == 1:
					self.call = self.L1_1

			elif self.reg_type.lower() == 'l2':
				if offset == 0:
					self.call = self.L2_0
				elif offset == 1:
					self.call = self.L2_1

			elif self.reg_type.lower() == 'sigma':
					self.call = self.Sigma

			else:
				assert self.reg_type.lower() in ['l1', 'l2', 'sigma'], "please specify a regularization type of l1 or l2"

		else:
			self.call = self.call_

		if self.graph:
			self.call = tf.function(self.call)

	def L1_0(self, c, weights):
		loss = self.call_(c)
		penalty = tf.reduce_sum(self.lamb * tf.abs(weights))
		reg_loss = loss + penalty
		# print(f"loss={round(loss.numpy(), 2)}, penalty={round(penalty.numpy(), 2)}, coeff={round(coeff.numpy(), 2)}")

		return reg_loss

	def L2_0(self, c, weights):
		loss = self.call_(c)
		penalty = tf.reduce_sum(self.lamb * tf.math.square(weights))
		reg_loss = loss + penalty

		# with np.printoptions(precision=2):
		# 	print(f"loss={round(loss.numpy(), 2)}, penalty={round(penalty.numpy(), 2)}, coeff={weights.numpy()}")

		return reg_loss

	def L1_1(self, c, weights):
		loss = self.call_(c)
		penalty = tf.reduce_sum(self.lamb * tf.abs(weights - 1))
		reg_loss = loss + penalty
		# print(f"loss={round(loss.numpy(), 2)}, penalty={round(penalty.numpy(), 2)}, coeff={round(coeff.numpy(), 2)}")

		return reg_loss

	def L2_1(self, c, weights):
		loss = self.call_(c)
		penalty = tf.reduce_sum(self.lamb * tf.math.square(weights - 1))
		reg_loss = loss + penalty
		# print(f"loss={loss.numpy():.2f}, penalty={penalty.numpy():.2f}, total={reg_loss.numpy():.2f}, coeff={[round(w, 2) for w in weights.numpy()[0]]}")

		return reg_loss

	def safedivide(self, a, b):
		safe_x = tf.where(tf.not_equal(b, 0.), b, tf.ones_like(b))
		return tf.where(tf.not_equal(b, 0.), tf.math.divide(x=a, y=safe_x), tf.zeros_like(safe_x))

	def safelog(self, a):
		safe_a = tf.where(tf.not_equal(a, 0.), a, tf.ones_like(a))
		return tf.where(tf.not_equal(a, 0.), tf.math.log(safe_a), tf.zeros_like(a))

	def find_nonfinite(self, pEs):
		if (nonfinite_pE := tf.reduce_all(tf.math.is_finite(pEs))):
			where_nonfinite = tf.where(tf.math.is_finite(pEs) != True)

	def __getstate__(self):
		if self.graph:
			self.call = self.call.__original_wrapped__
		
		state = self.__dict__.copy()
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)

		if self.graph:
			self.call = tf.function(self.call)

class PhyloLossNonIterative(PhyloLoss):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call_(self, c, **kwargs):
		"""
		Calculate edge likelihood using simplification from Barido-Sottani, Vaughan, and Stadler 2020
		"""

		gbd_sum = tf.add_n([c["edge_gamma"], c["edge_b"], c["edge_d"]])
		cnst_c = tf.sqrt(tf.square(gbd_sum) - 4 * c["edge_d"] * (1 - c["edge_s"]) * c["edge_b"])
		cnst_x = (-gbd_sum - cnst_c) / 2
		cnst_y = (-gbd_sum + cnst_c) / 2

		t_e = c["edge_back_time"]
		t_s = c["edge_back_time"] + c["edge_time_step"]

		pD_num = (cnst_y + (c["edge_b"] * (1 - c["edge_rho"]))) * tf.exp(-cnst_c * t_e) - cnst_x - (c["edge_b"] * (1 - c["edge_rho"]))
		pD_denom = (cnst_y + (c["edge_b"] * (1 - c["edge_rho"]))) * tf.exp(-cnst_c * t_s) - cnst_x - (c["edge_b"] * (1 - c["edge_rho"]))
		pD = tf.exp(-cnst_c * c["edge_time_step"]) * (pD_num / pD_denom) ** 2
		log_pD = self.safelog(pD)

		line_like = tf.reduce_sum(log_pD)
		sample_like = tf.reduce_sum(tf.math.log(c["sample_s"] * c["sample_d"]))
		sample_like_csa = tf.reduce_sum(tf.math.log(c["csa_rho"]))
		birth_like = tf.reduce_sum(tf.math.log(c["birth_b"]))

		loss = -(line_like + sample_like + sample_like_csa + birth_like)

		# self.edge_beta = c["edge_b"].numpy()
		# self.log_pD = log_pD.numpy()
		# self.log_sample_like = tf.math.log(c["sample_s"] * c["sample_d"]).numpy()
		# self.log_birth_like = tf.math.log(2 * c["birth_b"]).numpy()
		# self.log_sample_like_csa = tf.math.log(c["csa_rho"]).numpy()
		
		return loss

class PhyloLossIterative(PhyloLoss):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		if self.graph:
			self.calcIterativePEs = tf.function(self.calcIterativePEs)
			self.calcPEs = tf.function(self.calcPEs)

	# https://stackoverflow.com/questions/2345944/exclude-objects-field-from-pickling-in-python
	def __getstate__(self):
		state = self.__dict__.copy()

		if self.graph:
			state['call'] = state['call'].__original_wrapped__
			state['calcIterativePEs'] = state['calcIterativePEs'].__original_wrapped__
			state['calcPEs'] = state['calcPEs'].__original_wrapped__

		return state

	def __setstate__(self, state):
		self.__dict__.update(state)

		if self.graph:
			self.call = tf.function(self.call)
			self.calcIterativePEs = tf.function(self.calcIterativePEs)
			self.calcPEs = tf.function(self.calcPEs)

	def calcIterativePEs(self, c):
		n_edges = c["edge_d"].shape[0]

		init_time = tf.constant(0, shape=[], dtype=tf.dtypes.float64)
		pE_init = tf.ones(shape=n_edges, dtype=tf.dtypes.float64)
		rho_scalar_init = (1 - c["pE_rho"][:, -1])

		pEs = tf.reshape(pE_init * rho_scalar_init, [1, -1])

		for i in c["pE_back_idxs"][::-1]:

			# This allows pEs to be concatenated to larger dimensions
			tf.autograph.experimental.set_loop_options(
       			shape_invariants=[(pEs, tf.TensorShape([None, n_edges]))]
    		)

			time = c["pE_back_times"][i]

			rho_scalar = (1 - c["pE_rho"][:, i])

			b = c["pE_b"][:, i]
			s = c["pE_s"][:, i]
			d = c["pE_d"][:, i]
			gamm = c["pE_gamma"][:, i]

			gbd_sum = tf.add_n([gamm, b, d])

			# 2023-07-12: running into issue where sqrt(gbd_sum) < 4 * d * (1 - s) * b, 
			# resulting in square root of a negative number = nan.
			# The difference is really tiny: 
			# 4 * d * (1 - s) * b = 4.000000000117102
			# tf.square(gbd_sum) = 4.000000000117101
			# so I think we can just clip this value to have a floor of 0
			# without actually affecting the quality of inference
			cnst_c = tf.sqrt(tf.clip_by_value(tf.square(gbd_sum) - 4 * d * (1 - s) * b, 0, np.inf))
			cnst_x = (-gbd_sum - cnst_c) / 2
			cnst_y = (-gbd_sum + cnst_c) / 2

			pE_num = (cnst_y + b * pE_init) * cnst_x * tf.exp(-cnst_c * time) - cnst_y * (cnst_x + b * pE_init) * tf.exp(-cnst_c * init_time)
			pE_denom = (cnst_y + b * pE_init) * tf.exp(-cnst_c * time) - (cnst_x + b * pE_init) * tf.exp(-cnst_c * init_time)

			pE = self.safedivide((-1 / b) * pE_num, pE_denom)

			# Multiply by rho
			pE *= rho_scalar

			# if not tf.reduce_all(tf.math.is_finite(pE)):
			# 	where_nf = tf.where(tf.math.is_finite(pE) != True)
			# 	breakpoint()

			# print(f"{i=}, {init_time=:.1f}, {time=:.1f}, pE_init={pE_init.numpy()[0]:.4f}, pE={pE.numpy()[0]:.4f}, s={s.numpy()[0]:.4f}")

			# Append to pE and set current time, pE as init_time, pE_init
			pE_reshape = tf.reshape(pE, [1, -1])
			pEs = tf.concat([pE_reshape, pEs], axis=0)
			pE_init = pE
			init_time = time

		all_pEs = tf.transpose(pEs)
		return all_pEs

	def calcPEs(self, c):
		all_pEs = self.calcIterativePEs(c)

		new_pE_inits = tf.gather_nd(all_pEs, c["edge_pE_interval"])

		# breakpoint()

		edge_b = c["edge_b"]

		# Starting time is edge's event time (closest to present, in bkwds time)
		time = c["edge_back_time"]
		init_time = c["edge_pE_init_time"]

		gbd_sum = tf.add_n([c["edge_gamma"], edge_b, c["edge_d"]])

		# 2023-07-12: see calcIterativePEs for clip_by_value explanation
		cnst_c = tf.sqrt(tf.clip_by_value(tf.square(gbd_sum) - 4 * c["edge_d"] * (1 - c["edge_s"]) * edge_b, 0, np.inf))
		cnst_x = (-gbd_sum - cnst_c) / 2
		cnst_y = (-gbd_sum + cnst_c) / 2

		pE_num = (cnst_y + edge_b * new_pE_inits) * cnst_x * tf.exp(-cnst_c * time) - cnst_y * (cnst_x + edge_b * new_pE_inits) * tf.exp(-cnst_c * init_time)
		pE_denom = (cnst_y + edge_b * new_pE_inits) * tf.exp(-cnst_c * time) - (cnst_x + edge_b * new_pE_inits) * tf.exp(-cnst_c * init_time)
		pEs = self.safedivide((-1 / edge_b) * pE_num, pE_denom)

		return cnst_x, cnst_y, cnst_c, pEs

	def call_(self, c, **kwargs):
		cnst_x, cnst_y, cnst_c, pEs = self.calcPEs(c)

		# Calculate edge and full tree likelihood
		# ---------------------------------------
		pD_denom = ((cnst_y + c["edge_b"] * pEs) * tf.exp(-cnst_c * c["edge_time_step"])) - (cnst_x + c["edge_b"] * pEs)
		pD_intermed = self.safedivide((cnst_y - cnst_x), pD_denom)
		pD = tf.exp(-cnst_c * c["edge_time_step"]) * tf.square(pD_intermed)

		log_pD = self.safelog(pD)
		line_like = tf.reduce_sum(log_pD)
		sample_like = tf.reduce_sum(tf.math.log(c["sample_s"] * c["sample_d"]))
		sample_like_csa = tf.reduce_sum(tf.math.log(c["csa_rho"]))
		birth_like = tf.reduce_sum(tf.math.log(c["birth_b"]))

		loss = -(line_like + sample_like + sample_like_csa + birth_like)

		# self.edge_beta = c["edge_b"].numpy()
		# self.log_pD = log_pD.numpy()
		# self.log_sample_like = tf.math.log(c["sample_s"] * c["sample_d"]).numpy()
		# self.log_birth_like = tf.math.log(c["birth_b"]).numpy()
		# self.log_sample_like_csa = tf.math.log(c["csa_rho"]).numpy()

		# if not tf.math.is_finite(loss):
		# 	breakpoint()
		
		return loss