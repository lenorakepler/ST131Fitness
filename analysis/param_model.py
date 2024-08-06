import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy.lib import recfunctions as rfn
from types import SimpleNamespace
from analysis.phylo_loss import PhyloLossNonIterative, PhyloLossIterative

class ParamModel(keras.Model):
	"""
	When making component parameters, make sure to think about which tensor
	groups require which parameters:

	edge_{component}: gamma, b, d, s, rho
	pE_{component}: gamma, b, d, s, rho
	birth_{component}: b
	sample_{}: s, d
	csa_{}: rho

	so, any parameter model that overrides, e.g., b needs to output
	edge_b and birth_b to p in its call() method

	Subclasses require call_ and call_pE methods; based on value of
	iterative_pE, one of those will be used in call()
	"""
	base_bdm_params = ['b', 's', 'd', 'rho', 'gamma']

	def __init__(self, data, iterative_pE, loss_kwargs, **kwargs):
		super().__init__(**kwargs)
		self.data = data
		self.iterative_pE = iterative_pE
		self.p = self.setP(data)
		self.n_edges = len(self.edge_arr)
		self.n_births = len(self.birth_arr)
		self.n_times = self.data.n_times
		self.loss_kwargs = loss_kwargs
		self.phylo_loss = PhyloLossIterative if self.iterative_pE else PhyloLossNonIterative

	def setMethods(self):
		for name, values in self.bdm.items():
			is_var = values[0]

			var = 'Var' if is_var else 'Const'
			pE = '_pE' if self.iterative_pE else ''

			# If the parameter is a variable, we must specify the method
			# to both initialize and get the values
			if is_var:
				tv = 'TV' if values[1] == True else ''
				setattr(self, f'{name}_call', getattr(self, f"get_{name}{var}{tv}{pE}"))

			# If the parameter is not a variable..
			# If it is one of the base bdm parameters (b, gamma, d, s, or rho),
			# then we automatically fetch the value from the data.
			# If it is not one of the base params, we must specify the method
			# to initialize the values
			if not is_var:
				if name not in self.base_bdm_params:
					setattr(self, f'{name}_call', getattr(self, f"get_{name}{var}{pE}"))

		self.call = getattr(self, f"call_{'PE' if self.iterative_pE else ''}")

	def getParamAll(self, arr, param, dtype=tf.dtypes.float64):
		"""
		Extract values of the indicated parameter from the data array
		for each time interval. Rows = value of a parameter for a given observation,
		Cols = at each param interval
		"""
		if self.data.param_interval_times.shape[0] > 1:
			cols = [n for n in arr.dtype.names if f'{param}_' in n]
			p_struct = arr[cols]
			p_arr = rfn.structured_to_unstructured(rfn.repack_fields(p_struct))
		else:
			p_arr = arr[param]

		return tf.constant(p_arr, dtype=dtype, shape=[p_arr.shape[0], p_arr.shape[1] if len(p_arr.shape) > 1 else 1])

	def getParamSingle(self, arr, param, dtype=tf.dtypes.float64):
		"""
		Extract the indicated parameter from the data array
		Rows = value of that parameter at a given observation's event time
		"""
		return tf.constant(arr[param], dtype=dtype)

	def setP(self, data):
		"""
		Get needed constant tensors from data array
		"""

		arr = data.array

		self.edge_arr = arr[arr['event'] == 4]
		self.birth_arr = arr[arr['event'] == 1]
		self.sample_arr = arr[arr['event'] == 2]
		self.rho_arr = arr[arr['event'] == 3]

		p = SimpleNamespace()

		param_dict = dict(
			edge=dict(
				arr=self.edge_arr,
				params=[
					'gamma', 'b', 'd', 's', 'rho',
					'time_step', 'back_time'
				],
			),
			birth=dict(
				arr=self.birth_arr,
				params=['b'],
			),
			sample=dict(
				arr=self.sample_arr,
				params=['s', 'd'],
			),
			csa=dict(
				arr=self.rho_arr,
				params=['rho'],
			)
		)

		for p_type, p_dict in param_dict.items():
			arr = p_dict['arr']
			for k in p_dict['params']:
				if k not in self.vars:
					setattr(p, f"{p_type}_{k}", self.getParamSingle(arr, k))

		p.edge_param_interval = self.getParamSingle(self.edge_arr, 'param_interval', dtype=tf.dtypes.int64)
		p.birth_param_interval = self.getParamSingle(self.birth_arr, 'param_interval', dtype=tf.dtypes.int64)
		p.sample_param_interval = self.getParamSingle(self.sample_arr, 'param_interval', dtype=tf.dtypes.int64)
		p.csa_param_interval = self.getParamSingle(self.rho_arr, 'param_interval', dtype=tf.dtypes.int64)

		if self.iterative_pE:

			arr = self.edge_arr
			pE_time_params = ['gamma', 'b', 'd', 's', 'rho']

			for k in pE_time_params:
				if k not in self.vars:
					setattr(p, f"pE_{k}", self.getParamAll(arr, k))

			# Since rho is associated with the end of a time interval
			# (the beginning in backward time) ...
			p.pE_rho = tf.concat([tf.zeros([p.pE_rho.shape[0], 1], dtype="float64"), p.pE_rho], axis=1)

			back_times = data.bkwd_interval_times

			# p.pE_back_times = list(zip(back_times, np.arange(len(back_times))))

			# To use tensorflow function, pE_back_times must be a tensor, and can't iterate across
			# a tensor in the way we want, so need to do this...
			p.pE_back_times = tf.constant(back_times, shape=len(back_times), dtype=tf.dtypes.float64)
			p.pE_back_idxs = tf.constant(np.arange(len(back_times)), shape=len(back_times), dtype=tf.dtypes.int64)

			edge_pE_interval = arr['pE_interval']
			p.edge_pE_init_time = tf.constant(np.take(np.append(back_times, 0), edge_pE_interval), dtype=tf.dtypes.float64)

			# breakpoint()
			p.edge_pE_interval = tf.constant(np.vstack([np.arange(0, edge_pE_interval.shape[0]), edge_pE_interval]).T, dtype=tf.dtypes.int64)

		# If not iterative, only need rho values from last param interval
		else:
			if 'rho' not in self.vars:
				setattr(p, f"pE_rho", self.getParamAll(self.edge_arr, 'rho')[:, -1])

		return p

	def call(self):
		# If a parameter is not a constant or a variable,
		# it needs to return a mapping
		return self.p

class ParamComponent():
	"""
	Blank base class for components that are used to compose the
	birth-death parameters birth rate, death rate, gamma, rho, s

	Requires methods:
		- init_{component}: specify initialization of either
						    constant or variable

		Methods specifying what occurs on call()
		Based on model input, one of these is selected:
		- get_{component}Const: for non-variable, non-iterative
		- get_{component}Const_pE: for non-variable, iterative
		- get_{component}Var: for variable, non-iterative
		- get_{component}Var_pE: for variable, iterative
		- get_{component}VarTV: for variable, time-variable, non-iterative
		- get_{component}VarTV_pE: for variable, time-variable, iterative
	"""
	pass

class ComponentSite(ParamComponent):
	"""
	site-specific effect methods
	"""
	def init_site(self):
		param_val = list(self.bdm['site'])

		if param_val[0] == True:
			self.edge_ft = np.array([np.fromiter(ft, dtype=int) for ft in self.edge_arr['ft']])
			self.birth_ft = np.array([np.fromiter(ft, dtype=int) for ft in self.birth_arr['ft']])

			site = tf.ones(shape=[1, self.edge_ft.shape[1]], dtype=tf.dtypes.float64)
			self.site = tf.Variable(site, name='site')

		else:
			self.edge_site_b = tf.ones(self.n_edges, dtype=tf.dtypes.float64)
			self.birth_site_b = tf.ones(self.n_births, dtype=tf.dtypes.float64)

			if self.iterative_pE:
				self.pE_site_b = self.edge_site_b

	def get_siteConst_pE(self):
		pass

	def get_siteConst(self):
		pass

	def get_siteVar(self):
		ls = tf.math.log(self.site)
		self.edge_site_b = tf.exp(tf.reduce_sum(ls * self.edge_ft, axis=1))
		self.birth_site_b = tf.exp(tf.reduce_sum(ls * self.birth_ft, axis=1))

	def get_siteVar_pE(self):
		ls = tf.math.log(self.site)
		self.edge_site_b = tf.exp(tf.reduce_sum(ls * self.edge_ft, axis=1))
		self.birth_site_b = tf.exp(tf.reduce_sum(ls * self.birth_ft, axis=1))
		self.pE_site_b = self.edge_site_b

class ComponentB0(ParamComponent):
	"""
	"base" birth rate (b0) effect methods
	"""
	def init_b0(self):
		param_val = list(self.bdm['b0'])

		if param_val[0] == False:
			self.edge_b0 = self.getParamSingle(self.edge_arr, 'b0')
			self.birth_b0 = self.getParamSingle(self.birth_arr, 'b0')

			if self.iterative_pE:
				self.pE_b0 = self.getParamAll(self.edge_arr, 'b0')

		else:
			# If don't want to vary with time
			if param_val[1] == False:
				self.b0 = tf.Variable(1.000001, dtype=tf.dtypes.float64, name='b0')

			else:
				self.b0 = tf.Variable(tf.ones(shape=self.n_times, dtype=tf.dtypes.float64) + .000001, name='b0')

			if self.iterative_pE:
				self.pE_ones = tf.ones(shape=[self.n_edges, self.n_times], dtype=tf.dtypes.float64)

	# get methods
	def get_b0Const(self):
		pass

	def get_b0Const_pE(self):
		pass

	def get_b0Var(self):
		self.edge_b0 = self.birth_b0 = self.b0

	def get_b0Var_pE(self):
		self.edge_b0 = self.birth_b0 = self.b0
		self.pE_b0 = self.b0 * self.pE_ones

	def get_b0VarTV(self):
		self.edge_b0 = tf.gather(self.b0, self.p.edge_param_interval)
		self.birth_b0 = tf.gather(self.b0, self.p.birth_param_interval)

	def get_b0VarTV_pE(self):
		self.edge_b0 = tf.gather(self.b0, self.p.edge_param_interval)
		self.birth_b0 = tf.gather(self.b0, self.p.birth_param_interval)
		self.pE_b0 = self.pE_ones * self.b0

class Site(ParamModel, ComponentB0, ComponentSite):
	"""
	model params: b0, site, d, s, rho
	call() calls self.edge_b0, self.birth_b0, + self.pE_b0 if pE
	"""
	def __init__(self, data, b0, site, gamma, d, s, rho, iterative_pE=True, **kwargs):
		self.bdm = dict(b0=b0, site=site, d=d, s=s, rho=rho, gamma=gamma)
		self.private_vars = [k for k, v in self.bdm.items() if v[0] == True]
		self.vars = ['b'] + self.private_vars

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
	
class ComponentCurrBeta2(ParamComponent):
	def init_currbeta(self):
		self.cb = tf.Variable(self.data.array['curr_beta'])
		self.where_birth, self.where_edge = self.data.getBirthDependent()

	def get_currbetaVar(self):
		self.edge_curr_beta = tf.gather(self.cb, self.where_edge)
		self.birth_curr_beta = tf.gather(self.cb, self.where_birth)

	def get_currbetaVar_pE(self):
		self.edge_curr_beta = tf.gather(self.cb, self.where_edge)
		self.birth_curr_beta = tf.gather(self.cb, self.where_birth)

class CurrBeta(ParamModel, ComponentCurrBeta2, ComponentB0):
	"""
	Model that overrides b
	b = curr_beta

	Used for getting current likelihood of a data set in boosting
	or for getting gradient of each phylogeny piece
	"""
	def __init__(self, data, **kwargs):
		self.bdm = dict(b0=[False], currbeta=[True, False], d=[False], s=[False], rho=[False], gamma=[False])
		self.private_vars = [k for k, v in self.bdm.items() if v[0] == True]
		self.vars = ['b'] + self.private_vars

		super().__init__(data=data, iterative_pE=data.iterative_pE, **kwargs)

		# Set parameter methods
		self.setMethods()

		# Also retreive b0 and individual_beta parameters
		getattr(self, f"init_currbeta")()
		getattr(self, f"init_b0")()

	def call_(self):
		for var in self.private_vars:
			getattr(self, f"{var}_call")()

		self.p.edge_b = self.edge_b0 * self.edge_curr_beta
		self.p.birth_b = self.birth_b0 * self.birth_curr_beta
		return self.p

	def call_PE(self):
		for var in self.private_vars:
			getattr(self, f"{var}_call")()

		self.p.edge_b = self.edge_b0 * self.edge_curr_beta
		self.p.birth_b = self.birth_b0 * self.birth_curr_beta
		self.p.pE_b = tf.transpose(tf.transpose(self.pE_b0) * self.edge_curr_beta)
		return self.p
	
class ComponentOffset(ParamComponent):
	"""
	Variable that controls how much to "offset" current fitness values
	No calculation needs to occur at call time, and will never be constant
	or time-varying, so only get_b0Var + get_b0Var_pE methods are defined, and are empty
	"""

	def init_offset(self):
		self.offset = tf.Variable(1, dtype=tf.dtypes.float64, name="Offset")

	def get_offsetVar(self):
		pass

	def get_offsetVar_pE(self):
		pass

class Offset(ParamModel, ComponentB0, ComponentOffset):
	"""
	Model that overrides b
	b = curr_beta + offset
	No BDM params required because only variable is offset
	Gets whether to do iterative pE from data attribute
	"""
	def __init__(self, data, **kwargs):
		self.bdm = dict(b0=[False], offset=[True, False], d=[False], s=[False], rho=[False], gamma=[False])
		self.private_vars = [k for k, v in self.bdm.items() if v[0] == True]
		self.vars = ['b'] + self.private_vars

		super().__init__(data=data, iterative_pE=data.iterative_pE, **kwargs)

		# Set parameter methods
		self.setMethods()

		# Also retreive b0 and curr_beta parameters, init offset
		self.edge_curr_beta = self.getParamSingle(self.edge_arr, 'curr_beta')
		self.birth_curr_beta = self.getParamSingle(self.birth_arr, 'curr_beta')
		getattr(self, f"init_b0")()
		getattr(self, f"init_offset")()

	def call_(self):
		self.p.edge_b = self.edge_b0 * (self.edge_curr_beta + self.offset)
		self.p.birth_b = self.birth_b0 * (self.birth_curr_beta + self.offset)
		return self.p

	def call_PE(self):
		self.p.edge_b = self.edge_b0 * (self.edge_curr_beta + self.offset)
		self.p.birth_b = self.birth_b0 * (self.birth_curr_beta + self.offset)
		self.p.pE_b = tf.transpose(tf.transpose(self.pE_b0) * (self.edge_curr_beta + self.offset))
		return self.p
