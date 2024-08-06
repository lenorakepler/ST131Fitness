import dendropy
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from collections import OrderedDict

class ParamObj():
	"""
	Takes in a path string, Path, or dictionary
	If reading from file, tries to evaluate values
	This is mostly for passing to PhyloObj
	"""

	def __init__(self, params):
		if isinstance(params, str) or isinstance(params, Path):
			assert params.exists() == True, f"Parameter file does not exist ({params})"

			try:
				params_df = pd.read_csv(params, index_col=0).squeeze("columns")
				self.setParamsFromString(params_df)

			except:
				print("Couldn't read parameter file as csv")
				print(f"Tried to read: {params}")


		elif isinstance(params, dict):
			for k, v in params.items():
				setattr(self, k, v)

	def setParamsFromString(self, params):
		for k, v in params.items():
			try:
				setattr(self, k, eval(v))
			except:
				setattr(self, k, v)

class PhyloObj():
	"""
	Takes in, loads, and stores:
		dendropy tree object path
		feature dataframe or dict
		params as ParamObj

	Performs counts of feature types, sites, etc.
	"""

	def __init__(self, tree_file, tree_schema, features_file, params):
		if isinstance(features_file, str):
			features_file = Path(features_file)

		# Get and store objects
		self.tree = self.getDendroTree(tree_file, tree_schema)
		self.features_df = self.loadFeaturesDF(features_file)
		self.params = params

		# Compute and store attributes
		self.root = self.tree.seed_node
		self.root_time = self.root.age - (self.root.edge_length if self.root.edge_length else 0)
		self.present_time = self.tree.max_distance_from_root() + self.root.age
		self.count = self.features_df.size
		self.feature_names = self.features_df.columns.to_list()
		self.features_dict = self.genFeaturesDict(self.features_df)
		self.site_counts = self.features_df.sum(axis=0).to_list()
		self.observed_mutations = [i for i, count in enumerate(self.site_counts) if count != 0]
		self.feature_type_counts = self.getFeaturetypeCounts(self.features_dict)
		self.genome_length = len(list(self.features_dict.values())[0])

		self.feature_types = dict(
			observed = list(self.feature_type_counts.keys()),
		)

		# Files and input
		self.tree_file = tree_file
		self.tree_schema = tree_schema
		self.features_file = features_file

		self.phylo_obj_type = self.__class__.__name__

	def getDendroTree(self, tree_file, tree_schema):
		"""
		Loads pickled dendropy tree if it exists.
		If it does not, loads dendropy tree from file
		and saves a pickled version for later.
		"""

		sys.setrecursionlimit(10000)

		pickle_file = tree_file.parent / (tree_file.stem + ".pkl")

		if pickle_file.exists():
			with open(pickle_file, "rb") as p:
				tree = pickle.load(p)

		else:
			tree = dendropy.Tree.get(path=tree_file,
			                         schema=tree_schema,
			                         suppress_internal_node_taxa=False,
			                         preserve_underscores=True,
			                         extract_comment_metadata=True,
			                         )

			for node in tree.levelorder_node_iter():
				node.age = node.distance_from_root()

			with open(str(pickle_file), "wb") as p:
				pickle.dump(tree, p)

		return tree

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

	def split_edge(self, tree, parent, child, intermediate_length, taxon_label):
		taxon = tree.taxon_namespace.new_taxon(taxon_label)

		i = parent.child_nodes().index(child)

		# Remove connection between parent and old child
		parent.remove_child(child)

		# Connect new node object to parent and old child
		# Btw, for some reason this did not update the tree
		# representation with either .add_child or .new_child
		# I had to specify the index (either that or remove
		# child beforehand... see also discussion here:
		# https://stackoverflow.com/questions/72307591/dendropy-add-inner-node-midway-between-two-nodes
		intermediate = parent.insert_new_child(
			index=i,
			taxon=taxon,
			edge_length=intermediate_length,
		)
		intermediate.add_child(child)

		# Update edge length of child
		child.edge_length = child.edge_length - intermediate_length

		# Update ages
		intermediate.age = parent.age + intermediate_length

		tree.update_taxon_namespace()
		tree.update_bipartitions(
			suppress_unifurcations=False, suppress_storage=True,
			collapse_unrooted_basal_bifurcation=False,
		)

		for a in child.ancestor_iter():
			print(f"\t\t{a.taxon.label}: age={a.age:.3f}, len={a.edge_length:.3f}")

		assert child.edge_length >= 0
		return intermediate, tree

	def createIntervals(self, interval_times, save_name):
		self.interval_times = interval_times

		param_interval_times = [t for t in interval_times if t < self.present_time]
		self.param_interval_times = np.array(param_interval_times)

		for node in self.tree.preorder_node_iter():
			name = node.taxon.label

			if not node.edge_length:
				node.edge_length = 0

			birth_time = node.age - node.edge_length
			node_event_time = node.age

			e_interval_idx = self.getParamInterval(node_event_time)
			b_interval_idx = self.getParamInterval(birth_time)

			print(f"\nNode {name}")
			print(f"\t{birth_time:.3f} ({b_interval_idx}) to {node_event_time:.3f} ({e_interval_idx})")

			# Walk along the edge, split at time intervals
			# Starting values
			parent_node = node.parent_node
			child_node = node

			# Skip root branch, since this wouldn't be observed
			if parent_node:
				for time_idx in list(range(b_interval_idx + 1, e_interval_idx + 1)):
					prev_interval_time = self.interval_times[time_idx]

					if prev_interval_time < child_node.age:
						# Make this such that event node is in previous tiem interval
						intermediate_length = prev_interval_time - 0.0001 - parent_node.age

						print(f"\n\tSplitting current edge ({parent_node.age:.3f} to {child_node.age:.3f}), len={child_node.edge_length} into two at {prev_interval_time}")
						intermediate, self.tree = self.split_edge(
							tree=self.tree,
							parent=parent_node,
							child=child_node,
							intermediate_length=intermediate_length,
							taxon_label=node.taxon.label + f"_interval{time_idx}"
						)

						intermediate_birth_time = intermediate.age - intermediate_length

						print(f"\tNew node spans from {intermediate_birth_time:.3f} to {intermediate.age:.3f}")

						parent_node = intermediate

				if parent_node:
					print(f"\tFinal edge segment spans from {parent_node.age:.3f} to {node_event_time:.3f}")

		self.tree.write(
			path=str(save_name),
			schema="newick",
			annotations_as_nhx=False,
			suppress_leaf_node_labels=False,
			unquoted_underscores=True, # Baltic can't deal with this
		)

	def loadFeaturesDF(self, features_file):
		"""
		Loads feature types DataFrame from .csv
		"""
		if features_file.suffix == ".csv":
			return pd.read_csv(features_file, index_col=0)
		elif features_file.suffix == ".tsv":
			return pd.read_csv(features_file, sep="\t", index_col=0)
		elif features_file.suffix in [".fasta", ".fa", ".aln"]:
			feature_dict = {}
			feature_strs = features_file.read_text().split(">")[1:]
			for feature_str in feature_strs:
				name, features, _ = feature_str.split("\n")
				feature_dict[name] = list(map(int, features))
			return pd.DataFrame(feature_dict).T
		else:
			sys.exit(f"Features file must be .csv, .tsv, or .fasta/.fa/.aln (got {features_file.suffix})")

	def genFeaturesDict(self, features_df):
		"""
		Loads features types
		into ordered dict from .csv
		"""
		features_dict = OrderedDict({})

		for name, row in features_df.iterrows():
			features_dict[name] = ''.join(map(str, row.values.tolist()))

		return features_dict

	def getFeaturetypeCounts(self, features_dict):
		unique = np.unique(np.array(list(features_dict.values())), return_counts=True)
		return {ft: c for ft, c in zip(*unique)}

	def getFTsWithObsMuts(self, observed_mutations, all_possible_feature_types):
		with_observed_muts = []

		observed_muts = set(observed_mutations)

		for seq in all_possible_feature_types:
			muts = {i for i, val in enumerate(seq) if val == '1'}

			# If this sequence doesn't have any mutations that aren't
			# observed, add it to the list
			if not muts.difference(observed_muts):
				with_observed_muts.append(seq)

		return with_observed_muts

	def save(self, out_folder):
		with open(out_folder / "phylo_obj_dict.pkl", "wb") as f:
			pickle.dump(self.__dict__, f)

class PhyloObjInfo():
	def __init__(self, out_folder):
		with open(out_folder / "phylo_obj_dict.pkl", "rb") as f:
			param_dict = pickle.load(f)

		for k, v in param_dict.items():
			setattr(self, k, v)