import numpy as np
import copy
from numpy.lib import recfunctions as rfn
import pandas as pd
import numpy.lib.recfunctions as rf
import math
import pickle
from analysis.param_model import CurrBeta, Offset
from analysis.optimizer import Optimizer
import tensorflow as tf
import re

class PhyloArrayer():
	"""
	Creates a matrix representation of a phylogenetic tree,
		which is stored in the attribute self.array
	Divides it into time intervals as specified
	Methods to add time- and object-specific birth-death model
		parameters to the matrix
	Methods for saving, subsetting, etc.

	TODO: Not strictly necessary that this is passed a phylo_object;
	just requires a Dendropy tree where each node and edge has a
	taxon.label and an age
	"""
	def __init__(self, phylo_obj, param_interval_times):
		self.phylo_obj = phylo_obj

		# Info from the phylogeny
		self.present_time = phylo_obj.present_time
		self.root_time = phylo_obj.root_time
		self.features_dict = phylo_obj.features_dict

		param_interval_times = [t for t in param_interval_times if t < self.present_time]
		self.param_interval_times = np.array(param_interval_times)
		self.bkwd_interval_times = self.present_time - self.param_interval_times

		# Create and add parameters to array
		self.array = self.createArray()

		# Subtract 1 because we define both beginning and end times,
		self.n_times = len(self.param_interval_times)

	def getParamInterval(self, time):
		"""
		Given an event time, returns the index of the
		parameter interval that time is contained in
		"""
		try:
			param_interval = np.where(time > self.param_interval_times)[0][-1]
		except:
			param_interval = 0
		return param_interval

	def createArray(self):
		"""
		Creates and returns a numpy record array containing
		information about the phylogeny. Each row maps to one
		phylogeny "piece", e.g. an edge segment, a birth node, or
		a sampled node.

		Columns are:
			'ft' (featuretype)
			'name' (as specified in the tree file --
					note that multiple phylogeny pieces may
					share a name)
			'birth_time'
			'event_time'
			'back_time' (e.g. time since present)
			'event' (1=birth, 2=poiss. samp., 3=rho samp., 4=edge)
			'param_interval'
			'pE_interval'
			'time_step' (event_time - birth_time)
		"""
		names = []
		birth_times = []
		event_times = []
		events = []
		ft_names = []
		parent_idx = []
		idx = []
		name_re = re.compile(r"_interval\d*$")
		for i, node in enumerate(self.phylo_obj.tree.preorder_node_iter()):
			node.idx = None
			node.annotations.add_bound_attribute("idx")
			node.idx = i

			name = node.taxon.label

			# This is in case we have previously split the tree at intervals
			ft_name = re.sub(name_re, "", name)

			if not node.edge_length:
				node.edge_length = 0

			birth_time = node.age - node.edge_length
			node_event_time = node.age

			# SAVE NODE INFO
			# --------------
			# If is birth event
			if node.num_child_nodes() > 1:
				record = True
				event = 1

			# If is sampling event
			elif node.is_leaf():
				record = True
				event = 2

			# Otherwise, is just a state change node,
			# so don't record anything
			else:
				record = False

			if record:
				events.append(event)
				birth_times.append(node_event_time)
				event_times.append(node_event_time)
				names.append(node.taxon.label)
				parent_idx.append(node.parent_node.idx if node.parent_node else -1)
				ft_names.append(ft_name)
				idx.append(node.idx)

			# SAVE EDGE INFO
			# --------------
			time = birth_time
			birth_times.append(time)
			names.append(name)
			events.append(4) # signifies edge
			event_times.append(node_event_time)
			parent_idx.append(node.parent_node.idx if node.parent_node else -1)
			ft_names.append(ft_name)
			idx.append(node.idx)

		time_steps = [e - b for e, b in zip(event_times, birth_times)]
		back_times = [self.present_time - t for t in event_times]
		param_intervals = [self.getParamInterval(t) for t in event_times]
		pE_intervals = [p+1 for p in param_intervals]
		sequences = [self.features_dict[name] for name in ft_names]

		arr_list = list(zip(
			idx,
			sequences,
			names,
			birth_times,
			event_times,
			back_times,
			events,
			param_intervals,
			pE_intervals,
			time_steps,
			parent_idx,
		))

		arr = np.array(arr_list, dtype=[
			('idx', int),
			('ft', object),
			('name', object),
			('birth_time', float),
			('event_time', object),
			('back_time', float),
			('event', int),
			('param_interval', int),
			('pE_interval', int),
			('time_step', float),
			('parent_idx', int),
		])

		self.phylo_obj.tree.write(
			path=str(self.phylo_obj.tree_file).replace(".nwk", "_idx.nwk"),
			schema="newick",
			suppress_annotations=False,
			annotations_as_nhx=True,
			suppress_leaf_node_labels=False,
			unquoted_underscores=True,  # Baltic can't deal with this
		)

		return arr

	def print(self):
		"""
		Prints .array in a more readable way
		"""
		with pd.option_context(
			'precision', 3,
			'display.max_rows', 20,
			'display.max_columns', None,
			'display.width', 0,
			'display.max_colwidth', 20,
		):
			print(pd.DataFrame(self.array))

	def arrayPrint(self, array):
		with pd.option_context(
			'display.precision', 3,
			'display.max_rows', 20,
			'display.max_columns', None,
			'display.width', 0,
			'display.max_colwidth', 20,
		):
			print(pd.DataFrame(array))

	def setCSAs(self):
		"""
		Extract samples that happened at CSAs and notate in array

		This is called after addArrayParams, because whether
		a sample is taken during a CSA is determined in part by
		whether rho=0 at the time interval it is in.
		"""

		# Subset of array corresponding to sampling event
		s_idx = np.where(self.array['event'] == 2)[0]

		# Extract event times and all rho values
		t = self.array['event_time'][s_idx]
		r = [list(_) for _ in self.array[[c for c in self.array.dtype.names if 'rho_' in c]][s_idx].tolist()]
		interval_times = self.param_interval_times

		# If a sample is taken (approximately) at the beginning of an interval
		# and that interval corresponds to a CSA (e.g. rho != 0), save idx and rho
		# value at that interval
		s_rho_idx = []
		rho_sample_intervals = []
		rho_sample_rhos = []

		for i, (time, rhos) in enumerate(zip(t, r)):
			for idx, it in enumerate(interval_times):
				if math.isclose(it, time, abs_tol=0.01):
					if (rho := rhos[idx]) != 0:
						s_rho_idx.append(i)
						rho_sample_rhos.append(rho)
						rho_sample_intervals.append(idx)
						break

		# Indices of original array
		rho_idx = s_idx[s_rho_idx]

		# Set all rho events to be type 3
		self.array['event'][rho_idx] = 3

		# Give correct rho value for sampling time
		self.array['rho'][rho_idx] = rho_sample_rhos

		# Give correct parameter interval
		self.array['param_interval'][rho_idx] = rho_sample_intervals

	def formatArrayParams(self, **kwargs):
		"""
		Format input birth-death parameters to make necessary
		columns for .array. Returns a list of names and values
		each self.n_obs long
		"""

		names = []
		values = []

		for param, value in kwargs.items():
			# If time varying
			if value[1]:
				# Get value of parameter for each phylogeny piece for each interval
				vals = [[value[0][i]] * self.n_obs for i in range(len(value[0]))]

				# Get value of parameter for phylogeny piece's own interval
				idx_vals = [v[i] for i, v in zip(self.array['param_interval'], zip(*vals))]

			else:
				# Get value of parameter for each phylogeny piece for each interval
				vals = [[value[0]] * self.n_obs for i in range(self.n_times)]

				# Get value of parameter for phylogeny piece's own interval
				idx_vals = [value[0]] * self.n_obs

			names += [param]

			interval_names = [f"{param}_{i}" for i in range(len(vals))]
			names += interval_names

			values += [idx_vals]
			values += vals

		return names, values

	def addArrayParams(self, **kwargs):
		"""
		Fills .array with birth-death-sampling parameters
		If time-varying, chooses the correct value for each row

		Input keys should be parameter names and values
		should be in the form of a tuple whose first entry
		corresponds to the parameter values (either a single value
		or a list if time-varying) and second, a boolean specifying
		whether the parameter should vary with time.

		e.g. addArrayParams(gamma=(0, False),
							s=(self.sample_rates, True),
							d=(po.params.removal_rate, False),
							)
		"""
		self.n_obs = self.array.size

		names, values = self.formatArrayParams(**kwargs)

		self.array = rfn.append_fields(self.array, names, values, dtypes=[float for _ in values], usemask=False)

		# Extract samples that happened at CSAs
		if 'rho' in kwargs:
			self.setCSAs()

		self.existing_fts = np.unique(self.array['ft'])

	def updateArrayParams(self, **kwargs):
		"""
		Update existing birth-death parameters in self.array
		"""
		names, values = self.formatArrayParams(**kwargs)

		for name, vals in zip(names, values):
			self.array[name] = vals

	def toArray(self, x):
		"""
		Get a column of self.array as a regular numpy array
		"""
		arr = rf.structured_to_unstructured(rf.repack_fields(x))
		return arr

	def getBirthDependent(self):
		"""
		Returns where_birth, an index of all birth events in
		self.array and where_edge, an index of all edges
		"""
		where_birth = np.flatnonzero(self.array['event'] == 1)
		where_edge = np.flatnonzero(self.array['event'] == 4)
		return where_birth, where_edge

	def makeTrainTest(self, proportion):
		"""
		Returns a PhyloData object with randomly-selected
		observations from self.array (train). The unselected
		observations go in a second returned "test" PhyloData object
		"""
		assert proportion <= 1, "Training proportion must be less than 1"

		num_to_train = int(len(self.array) * proportion)

		shuffled_array = copy.copy(self.array)
		np.random.shuffle(shuffled_array)

		split = np.split(shuffled_array, [num_to_train])

		train_array = split[0]
		test_array = split[1]

		train = PhyloData(train_array, **self.getDataParams())
		test = PhyloData(test_array, **self.getDataParams())

		return train, test

	def toData(self):
		return PhyloData(self.array.copy(), **self.getDataParams())

	def addColumn(self, field_name, field_values, field_dtype):
		"""
		Convenience function for adding a single column to .array
		"""
		if field_name in self.array.dtype.names:
			print(f"field {field_name} already exists, replacing values")
			self.array[field_name] = field_values
		else:
			self.array = rfn.append_fields(self.array, [field_name], [field_values], dtypes=[field_dtype], usemask=False)

	def setWeight(self, birth_weight):
		"""
		Set "weight" used for birth events when calculating
		RMSE, etc. (1 edge time unit corresponds to a weight of
		1 and sampling events do not have any weight because they
		do not depend on feature type).

		Sets self.birth_weight and adds a 'weight' column to .array
		"""
		self.birth_weight = birth_weight
		w_dict = {1: birth_weight, 2: 0, 3: 0}
		weight = [t if e == 4 else w_dict[e] for t, e in zip(self.array['time_step'], self.array['event'])]

		if 'weight' in self.array.dtype.names:
			self.array['weight'] = weight
		else:
			self.addColumn('weight', weight, float)

	def getDataParams(self):
		data_params = {k: v for k, v in self.__dict__.items() if k not in ['array', 'phylo_obj']}
		return data_params

	def save(self, name, out_folder):
		"""
		Save array as numpy object and if doesn't already exist, save data_params
		Enables loading for future use
		"""
		np.save(out_folder / name, self.array, allow_pickle=True, fix_imports=True)

		with open(out_folder / "data_dict.pkl", "wb") as f:
			pickle.dump(self.getDataParams(), f)

class PhyloData(PhyloArrayer):
	def __init__(self, array, **kwargs):
		# This should be taken care of now, but for all still out there...
		if 'phylo_obj' in kwargs:
			del kwargs['phylo_obj']

		self.array = array

		for k, v in kwargs.items():
			setattr(self, k, v)

	def returnCopyWithArray(self, array):
		return self.__class__(array=copy.deepcopy(array), **self.getDataParams())

	def returnCopy(self):
		return self.__class__(array=copy.deepcopy(self.array), **self.getDataParams())

	def getSubArray(self, sequences_to_select):
		new_array = self.array[np.isin(self.array['ft'], sequences_to_select)]
		sub_array_obj = self.__class__(array=new_array, **self.getDataParams())

		return sub_array_obj

	def getSubArraySpecific(self, indices):
		"""
		Given specific indices of .array, returns a new object
		with a .array containing only those observations
		"""
		new_array = self.array[indices]
		sub_array_obj = self.__class__(array=new_array, **self.getDataParams())

		return sub_array_obj

	def getEventArray(self, event, mask=False):
		if 'birth' in event:
			event_no = 1
		elif 'rho' in event or 'csa' in event:
			event_no = 3
		elif 'sampl' in event:
			event_no = 2
		elif 'edge' in event:
			event_no = 4

		event_mask = self.array['event']==event_no

		if mask:
			return self.array[event_mask], event_mask
		else:
			return self.array[event_mask]
			
	def generateFtdf(self, actual_beta_dict):
		data = self.returnCopy()
		fts = data.existing_fts
		ftdf = pd.DataFrame(index=fts, columns=['edges', 'births', 'weight', 'actual'])

		for ft in fts:
			tft = data.getSubArray([ft])

			where_birth, where_edge = tft.getBirthDependent()
			births = tft.array[where_birth].size
			edges = tft.array[where_edge]['time_step'].sum()
			weight = births + edges

			actual = actual_beta_dict[ft]

			ftdf.loc[ft, :] = [edges, births, weight, actual]

		self.ftdf = ftdf

	def initCurrBetas(self, init_beta, dtype=np.float64):
		"""
		Adds column 'curr_beta' to .array
		"""
		curr_beta = np.full_like(self.array['birth_time'], init_beta)
		self.addColumn('curr_beta', curr_beta, dtype)

	def splitKFold(self, k):
		shuffled_array = copy.deepcopy(self.array)
		np.random.shuffle(shuffled_array)
		k_arrs = np.array_split(shuffled_array, k)

		phylo_data_objs = []
		for i, arr in enumerate(k_arrs):
			test_k = self.__class__(arr, **self.getDataParams())
			train_k_arr = np.concatenate([a for j, a in enumerate(k_arrs) if j != i])
			train_k = self.__class__(train_k_arr, **self.getDataParams())

			phylo_data_objs.append((train_k, test_k))

		return phylo_data_objs

	def updateCurrBetasByFT(self, fts_to_select, add_value):
		"""
		Given a list of fts and a value, increases 'curr_beta'
		of all rows with that feature type by the specified amount
		"""
		bools = np.isin(self.array['ft'], fts_to_select)
		new_beta = self.array['curr_beta'] + (bools * add_value)

		# Prevent any birth rate from becoming 0 or negative
		self.array['curr_beta'] = np.where(new_beta > 1e-10, new_beta, 1e-10)

	def returnCurrLike(self, return_model=False):
		fit_model = CurrBeta(self, loss_kwargs={})
		phylo_loss = fit_model.phylo_loss()
		loss = phylo_loss.call(fit_model.call())

		if return_model:
			return loss, fit_model

		return loss

	def returnCurrLikePenalized(self, loss_kwargs, weights):
		fit_model = CurrBeta(self, loss_kwargs=loss_kwargs)
		phylo_loss = fit_model.phylo_loss(**loss_kwargs)
		loss = phylo_loss.call(fit_model.call(), weights=tf.constant(weights, dtype=tf.float64))

		return loss

	def getMLE(self):
		opt = Optimizer(
			fit_model=Offset,
			fit_model_kwargs=dict(data=self, loss_kwargs={}),
			n_epochs=10000, lr=0.01,
		)
		opt.verbose = False
		offset, loss = opt.doOpt()
		return offset['Variable']