import tensorflow as tf
import numpy as np

class PhyloLoss(tf.keras.losses.Loss):
	def __init__(self, **kwargs):
		super().__init__()

		self.epsilon = kwargs.get('epsilon', 0.0000005)

		for k, v in kwargs.items():
			setattr(self, k, v)

		self.graph = kwargs.get('graph', True)
		self.sigma = kwargs.get('sigma', 0)

		if self.graph:
			self.Sigma = tf.function(self.Sigma)

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

			else:
				assert self.reg_type.lower() in ['l1', 'l2'], "please specify a regularization type of l1 or l2"
		else:
			self.call = self.call_

		if self.graph:
			self.call = tf.function(self.call)

	def L1_0(self, m, weights):
		loss = self.Sigma(m) if self.sigma else self.call_(m)
		penalty = tf.reduce_sum(self.lamb * tf.abs(weights))
		reg_loss = loss + penalty

		self.penalty = penalty
		# print(f"loss={round(loss.numpy(), 2)}, penalty={round(penalty.numpy(), 2)}, coeff={round(coeff.numpy(), 2)}")

		return reg_loss

	def L2_0(self, m, weights):
		loss = self.Sigma(m) if self.sigma else self.call_(m)
		penalty = tf.reduce_sum(self.lamb * tf.math.square(weights))
		reg_loss = loss + penalty

		self.penalty = penalty

		# with np.printoptions(precision=2):
		# 	print(f"loss={round(loss.numpy(), 2)}, penalty={round(penalty.numpy(), 2)}, coeff={weights.numpy()}")

		return reg_loss

	def L1_1(self, m, weights):
		loss = self.Sigma(m) if self.sigma else self.call_(m)
		penalty = tf.reduce_sum(self.lamb * tf.abs(weights - 1))
		reg_loss = loss + penalty

		self.penalty = penalty

		# print(f"lambda penalty={penalty}")
		# print(f"loss={reg_loss}")
		# print(f"loss={round(loss.numpy(), 2)}, penalty={round(penalty.numpy(), 2)}, coeff={round(coeff.numpy(), 2)}")

		return reg_loss

	def L2_1(self, m, weights):
		loss = self.Sigma(m) if self.sigma else self.call_(m)
		penalty = tf.reduce_sum(self.lamb * tf.math.square(weights - 1))
		reg_loss = loss + penalty
		# print(f"loss={loss.numpy():.2f}, penalty={penalty.numpy():.2f}, total={reg_loss.numpy():.2f}, coeff={[round(w, 2) for w in weights.numpy()[0]]}")

		self.penalty = penalty

		return reg_loss

	def Sigma(self, m, **kwargs):
		sigma = self.sigma
		loss = self.call_(m)

		fit_shifts = tf.gather(m["brownian_eff"], m["edge_type_int"]) - tf.gather(m["brownian_eff"], m["edge_parent_type_int"])
		times = m["edge_parent_time_delta"]
		
		# p(child fitness u | parent fitness) = -exp[(u_c - u_p)^2 / (2 * sigma)]
		# but we are dealing in log likelihood, so take log
		probs = tf.math.divide_no_nan(-0.5 * fit_shifts**2, sigma * times + self.epsilon)
		penalty = tf.reduce_sum(probs) # Sum log prob values

		# Penalty term will always be negative: the more negative,
		# the farther we are from optimal. Because our loss is a
		# negative log likelihood that we MINIMIZE,
		# the added penalty needs to be LARGER the farther we are from
		# optimal. So, multiply by -1
		penalty = penalty * -1

		# self.sigma_penalty_info = dict(
		# 	fit_shifts = fit_shifts,
		# 	probs = probs,
		# 	sigma_penalty = penalty.numpy(),
		# 	child_effs = tf.gather(m["brownian_eff"], m["edge_type_int"]),
		# 	parent_effs = tf.gather(m["brownian_eff"], m["edge_parent_type_int"]),
		# 	times = times,
		# )

		# print(f"sigma penalty={penalty}")

		# if (self.i % 100 == 0):
		# 	with np.printoptions(precision=4):
		# 		print(f"{self.i}: loss={loss.numpy()}, penalty={penalty.numpy()}, total={loss.numpy() + penalty.numpy()}")
		# 		print(m["rand_eff"][0:10].numpy())
		
		return loss + penalty

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

	def call_(self, m, **kwargs):
		"""
		Calculate edge likelihood using simplification from Barido-Sottani, Vaughan, and Stadler 2020
		"""

		gbd_sum = tf.add_n([m["edge_gamma"], m["edge_b"], m["edge_d"]])
		cnst_c = tf.sqrt(tf.square(gbd_sum) - 4 * m["edge_d"] * (1 - m["edge_s"]) * m["edge_b"])
		cnst_x = (-gbd_sum - cnst_c) / 2
		cnst_y = (-gbd_sum + cnst_c) / 2

		t_e = m["edge_back_time"]
		t_s = m["edge_back_time"] + m["edge_time_step"]

		pD_num = (cnst_y + (m["edge_b"] * (1 - m["edge_rho"]))) * tf.exp(-cnst_c * t_e) - cnst_x - (m["edge_b"] * (1 - m["edge_rho"]))
		pD_denom = (cnst_y + (m["edge_b"] * (1 - m["edge_rho"]))) * tf.exp(-cnst_c * t_s) - cnst_x - (m["edge_b"] * (1 - m["edge_rho"]))
		pD = tf.exp(-cnst_c * m["edge_time_step"]) * (pD_num / pD_denom) ** 2
		log_pD = self.safelog(pD)

		line_like = tf.reduce_sum(log_pD)
		sample_like = tf.reduce_sum(tf.math.log(m["sample_s"] * m["sample_d"]))
		sample_like_csa = tf.reduce_sum(tf.math.log(m["csa_rho"]))
		birth_like = tf.reduce_sum(tf.math.log(m["birth_b"]))

		loss = -(line_like + sample_like + sample_like_csa + birth_like)

		# self.edge_beta = m["edge_b"].numpy()
		# self.log_pD = log_pD.numpy()
		# self.log_sample_like = tf.math.log(m["sample_s"] * m["sample_d"]).numpy()
		# self.log_birth_like = tf.math.log(2 * m["birth_b"]).numpy()
		# self.log_sample_like_csa = tf.math.log(m["csa_rho"]).numpy()
		
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

	def calcIterativePEs(self, m):
		n_edges = m["edge_d"].shape[0]

		init_time = tf.constant(0, shape=[], dtype=tf.dtypes.float64)
		pE_init = tf.ones(shape=n_edges, dtype=tf.dtypes.float64)
		rho_scalar_init = (1 - m["pE_rho"][:, -1])

		pEs = tf.reshape(pE_init * rho_scalar_init, [1, -1])


		for i in m["pE_back_idxs"][::-1]:

			# This allows pEs to be concatenated to larger dimensions
			tf.autograph.experimental.set_loop_options(
       			shape_invariants=[(pEs, tf.TensorShape([None, n_edges]))]
    		)

			time = m["pE_back_times"][i]

			rho_scalar = (1 - m["pE_rho"][:, i])

			b = m["pE_b"][:, i]
			s = m["pE_s"][:, i]
			d = m["pE_d"][:, i]
			gamm = m["pE_gamma"][:, i]

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
			pE = pE * rho_scalar

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

	def calcPEs(self, m):
		all_pEs = self.calcIterativePEs(m)

		new_pE_inits = tf.gather_nd(all_pEs, m["edge_pE_interval"])

		edge_b = m["edge_b"]

		# Starting time is edge's event time (closest to present, in bkwds time)
		time = m["edge_back_time"]
		init_time = m["edge_pE_init_time"]

		gbd_sum = tf.add_n([m["edge_gamma"], edge_b, m["edge_d"]])

		# 2023-07-12: see calcIterativePEs for clip_by_value explanation
		cnst_c = tf.sqrt(tf.clip_by_value(tf.square(gbd_sum) - 4 * m["edge_d"] * (1 - m["edge_s"]) * edge_b, 0, np.inf))
		cnst_x = (-gbd_sum - cnst_c) / 2
		cnst_y = (-gbd_sum + cnst_c) / 2

		pE_num = (cnst_y + edge_b * new_pE_inits) * cnst_x * tf.exp(-cnst_c * time) - cnst_y * (cnst_x + edge_b * new_pE_inits) * tf.exp(-cnst_c * init_time)
		pE_denom = (cnst_y + edge_b * new_pE_inits) * tf.exp(-cnst_c * time) - (cnst_x + edge_b * new_pE_inits) * tf.exp(-cnst_c * init_time)
		pEs = self.safedivide((-1 / edge_b) * pE_num, pE_denom)

		return cnst_x, cnst_y, cnst_c, pEs

	def call_(self, m, **kwargs):
		cnst_x, cnst_y, cnst_c, pEs = self.calcPEs(m)

		# Calculate edge and full tree likelihood
		# ---------------------------------------
		pD_denom = ((cnst_y + m["edge_b"] * pEs) * tf.exp(-cnst_c * m["edge_time_step"])) - (cnst_x + m["edge_b"] * pEs)
		pD_intermed = self.safedivide((cnst_y - cnst_x), pD_denom)
		pD = tf.exp(-cnst_c * m["edge_time_step"]) * tf.square(pD_intermed)

		log_pD = self.safelog(pD)
		line_like = tf.reduce_sum(log_pD)
		sample_like = tf.reduce_sum(tf.math.log(m["sample_s"] * m["sample_d"]))
		sample_like_csa = tf.reduce_sum(tf.math.log(m["csa_rho"]))
		birth_like = tf.reduce_sum(tf.math.log(m["birth_b"]))

		loss = -(line_like + sample_like + sample_like_csa + birth_like)

		# print(f"{line_like=}")
		# print(f"{sample_like=}")
		# print(f"{sample_like_csa=}")
		# print(f"{birth_like=}")

		# breakpoint()

		# self.edge_beta = m["edge_b"].numpy()
		# self.log_pD = log_pD.numpy()
		# self.log_sample_like = tf.math.log(m["sample_s"] * m["sample_d"]).numpy()
		# self.log_birth_like = tf.math.log(m["birth_b"]).numpy()
		# self.log_sample_like_csa = tf.math.log(m["csa_rho"]).numpy()

		# if not tf.math.is_finite(loss):
		# 	breakpoint()
		
		return loss