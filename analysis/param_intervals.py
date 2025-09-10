from pathlib import Path
import pandas as pd
import numpy as np
import re
from numpy.lib import recfunctions as rfn
from analysis.phylo_obj import PhyloObjPlain

def make_intervals(interval_dir, original_tree_file, last_sample_date, *start_time_lists):
	"""
	Get all times, add root time if not there, sort,
	make interval tree, return interval list
	"""

	interval_dir = Path(interval_dir)
	print(start_time_lists)

	# -----------------------------------------------------
	# Load phylo obj, set dates
	# -----------------------------------------------------
	phylo_obj = PhyloObjPlain(
		tree_file=original_tree_file,
		tree_schema="newick",
	)
	
	for n in phylo_obj.tree.nodes():
		n.age = n.age + (last_sample_date - phylo_obj.present_time)

	phylo_obj.root = phylo_obj.tree.seed_node
	phylo_obj.root_time = phylo_obj.root.age - (phylo_obj.root.edge_length if phylo_obj.root.edge_length else 0)
	phylo_obj.present_time = last_sample_date

	# -----------------------------------------------------
	# Get list of all interval times, make interval tree
	# -----------------------------------------------------
	interval_times = sorted(list(set([phylo_obj.root_time] + [item for sublist in start_time_lists for item in sublist])))
	
	interval_dir.mkdir(exist_ok=True, parents=True)

	interval_tree = interval_dir / "phylo.nwk"
	phylo_obj.createIntervals(
		interval_times=interval_times,
		save_name=interval_tree,
		verbose=False,
	)

	np.savetxt(str(interval_dir / "interval_times.txt"), np.array(interval_times), delimiter=',')
	
	return interval_times, interval_tree