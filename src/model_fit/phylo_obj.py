import dendropy
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from collections import OrderedDict

class PhyloObj():
	def __init__(self, tree_file, tree_schema, last_sample_date=None):
		self.tree = self.getDendroTree(tree_file, tree_schema)
		
		# Compute and store attributes
		self.root = self.tree.seed_node
		self.root_time = self.root.age - (self.root.edge_length if self.root.edge_length else 0)
		self.present_time = self.tree.max_distance_from_root() + self.root.age

		if last_sample_date:
			self.last_sample_date = last_sample_date
			for n in self.tree.nodes():
				n.age = n.age + (last_sample_date - self.present_time)
				
			self.root_time = self.root.age - (self.root.edge_length if self.root.edge_length else 0)
			self.present_time = last_sample_date
			
		# Files and input
		self.tree_file = Path(tree_file)
		self.tree_schema = tree_schema

		self.phylo_obj_type = self.__class__.__name__

		self.features_dict = {}

	def save(self, out_folder):
		with open(out_folder / "phylo_obj_dict.pkl", "wb") as f:
			pickle.dump(self.__dict__, f)

	def getDendroTree(self, tree_file, tree_schema):
		"""
		Loads pickled dendropy tree if it exists.
		If it does not, loads dendropy tree from file
		and saves a pickled version for later.
		"""

		sys.setrecursionlimit(10000)

		pickle_file = Path(tree_file).parent / (Path(tree_file).stem + ".pkl")

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

	def split_edge(self, tree, parent, child, intermediate_length, taxon_label, verbose):
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

		if verbose:
			for a in child.ancestor_iter():
				print(f"\t\t{a.taxon.label}: age={a.age:.3f}, len={a.edge_length:.3f}")

		assert child.edge_length >= 0
		return intermediate, tree

	def createIntervals(self, interval_times, save_name, verbose=True):
		self.interval_times = interval_times

		param_interval_times = [t for t in interval_times if t < self.present_time]
		self.param_interval_times = np.array(param_interval_times)

		n_nodes = len([n for n in self.tree.preorder_node_iter()])
		for node_idx, node in enumerate(self.tree.preorder_node_iter()):
			name = node.taxon.label

			if not node.edge_length:
				node.edge_length = 0

			birth_time = node.age - node.edge_length
			node_event_time = node.age

			e_interval_idx = self.getParamInterval(node_event_time)
			b_interval_idx = self.getParamInterval(birth_time)

			if not verbose:
				if node_idx % 100 == 0:
					print(f"Node {name} ({node_idx}/{n_nodes - 1})")

			if verbose:
				print(f"\nNode {name} ({node_idx + 1}/{n_nodes})")
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

						if verbose: print(f"\n\tSplitting current edge ({parent_node.age:.3f} to {child_node.age:.3f}), len={child_node.edge_length} into two at {prev_interval_time}")
						intermediate, self.tree = self.split_edge(
							tree=self.tree,
							parent=parent_node,
							child=child_node,
							intermediate_length=intermediate_length,
							taxon_label=node.taxon.label + f"_interval{time_idx}",
							verbose=verbose,
						)

						intermediate_birth_time = intermediate.age - intermediate_length

						if verbose: print(f"\tNew node spans from {intermediate_birth_time:.3f} to {intermediate.age:.3f}")

						parent_node = intermediate

				if parent_node:
					if verbose: print(f"\tFinal edge segment spans from {parent_node.age:.3f} to {node_event_time:.3f}")

		self.tree.write(
			path=str(save_name),
			schema="newick",
			annotations_as_nhx=False,
			suppress_leaf_node_labels=False,
			unquoted_underscores=True, # Baltic can't deal with this
		)

class PhyloObjInfo():
	def __init__(self, out_folder):
		with open(out_folder / "phylo_obj_dict.pkl", "rb") as f:
			param_dict = pickle.load(f)

		for k, v in param_dict.items():
			setattr(self, k, v)