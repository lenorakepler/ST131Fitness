"""''
Output a phylogeny with tips colored
based on a specified trait
+ general functions for visualization, baltic
"""
import baltic as bt
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib._color_data as mcd
from matplotlib.pyplot import GridSpec
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import dendropy
from ete3 import Tree

def prune_tree_to_proportion(tree_file, proportion=False, count=False):
	"""
	Randomly prune the tree such that only a given proportion or count of leaves remain
	
	Parameters
	----------
	tree_file : newick or nexus (I think) formatted tree file
	proportion : proportion (0 < x < 1) of the original number of leaves you would like to remain
	count : number of leaves you would like to remain

	NOTE: EITHER proportion or count must be specified, but not both

	Returns
	-------
	tree : an ete3 tree pruned to specified size
	"""

	tree = Tree(str(tree_file), format=3)
	leaves = tree.get_leaves()

	if proportion:
		count = int(np.ceil(len(leaves) * proportion))

	sample = np.random.choice(leaves, size=count, replace=False)
	tree.prune(sample, preserve_branch_length=True)

	return tree

def removeLeaves(tree_file, remove_leaves):
	tree = Tree(str(tree_file), format=3)
	keep = [leaf for leaf in tree.get_leaves() if leaf.name not in remove_leaves]
	tree.prune(keep, preserve_branch_length=True)

	return tree

def reroot_to_string(tree_file, root):
	"""
	Load a newick tree and re-root it to a given node; 
	return newick string

	Parameters
	----------
	tree_file : newick-formatted tree file
	root : name of the node on which to reroot

	Returns
	-------
	tree : newick string of re-rooted tree
	"""

	tree = dendropy.Tree.get(
		path=tree_file,
		schema='newick',
	)
	root_node = tree.find_node_with_taxon_label(root)

	tree.reroot_at_node(
		root_node,
		update_bipartitions=False
	)

	return tree.as_string(schema="newick")[5:]

def midpoint_to_string(tree_file):
	"""
	Load a newick tree and midpoint root it;
	return newick string
	
	Parameters
	----------
	tree_file : newick-formatted tree file

	Returns
	-------
	tree : newick string of re-rooted tree
	"""

	tree = dendropy.Tree.get(
		path=tree_file,
		schema='newick',
	)

	tree.reroot_at_midpoint(update_bipartitions=False)

	return tree.as_string(schema="newick")[5:]

def loadTree(tree_file, internal=True, abs_time=0, sort_descending=True, format="newick"):
	"""
	Load a newick-formatted tree as a Baltic tree object
	Sets leaf and optionally internal node names as node trait 'label'
	Sets absolute time

	Parameters
	----------
	tree_file : path of newick-formatted tree file
	internal : whether tree file has internal node names to set
	abs_time : sets absolute time of present 
			   (e.g. adjust node ages so that last sampled leaf has this time)
			   defaults so present time is 0

	Returns
	-------
	tt : Baltic tree object
	"""


	if format.lower() in ["nexus", "nex", ".nex"]:
		tt = bt.loadNexus(str(tree_file), absoluteTime=False)
	elif format.lower() in ["string", "str"]:
		tt = bt.make_tree(tree_file)
	else:
		tt = bt.loadNewick(str(tree_file), absoluteTime=False)

	tt.traverse_tree()  # required to set heights
	tt.setAbsoluteTime(abs_time)

	if sort_descending:
		tt.sortBranches(descending=True)

	# Put node name in consistent location
	for i, k in enumerate(tt.Objects):
		if k.branchType == 'node':
			if internal:
				try:
					name = k.traits['label']
				except:
					if i == 0:
						name = "root"
			else:
				name = "internal"
		else:
			name = k.name

		k.traits['name'] = name

	return tt

def set_vertical(ax, line_style=(0.0, [1, 10]), color="lightgray"):
	# Get all vertical connecting line segments
	vert = []
	for i, seg in enumerate(ax.collections[0].get_segments()):
		if len(seg) > 1:
			if seg[0, 1] != seg[1, 1]:
				vert.append(i)

	# Make them dashed and a different color
	vert_rgba = list(mpl.colors.to_rgba(color))
	line_styles = [line_style if i in vert else ls for i, ls in enumerate(ax.collections[0].get_linestyle())]
	line_colors = [vert_rgba if i in vert else c for i, c in enumerate(ax.collections[0].get_color())]
	ax.collections[0].set_linestyles(line_styles)
	ax.collections[0].set_color(line_colors)

	return ax

def plotTrait(tt, node_c_func, edge_c_func, trait, save, s_func="default", tip_names=False, legend=False, lloc=3, figsize=(10, 20), zoom=False):
	"""
	Zoom can take a tuple or list of two values indicating the upper and lower limits of the "zoom"
	"""

	fig,ax = plt.subplots(figsize=figsize, facecolor='w')

	x_attr = lambda k: k.absoluteTime ## x coordinate of branches will be absoluteTime attribute
	s_func = s_func ## size of tips

	tt.plotTree(ax, x_attr=x_attr, colour=edge_c_func) ## plot branches
	if s_func:
		if s_func == "default":
			s_func = lambda k: 50 - 4500 / tt.treeHeight
		tt.plotPoints(ax, x_attr=x_attr, size=s_func, colour=node_c_func, zorder=100) ## plot circles at tips

	if tip_names:
		text_func = lambda k: k.name
		tt.addText(ax, x_attr=x_attr, text=text_func, color='k')

	y_padding = tt.ySpan * .01
	x_padding = 4.261781308000001 * .05
	ax.set_ylim(0 - y_padding, tt.ySpan + y_padding)
	ax.set_xlim(tt.root.absoluteTime - tt.root.length - x_padding, tt.mostRecent + x_padding)

	if zoom:
		ax.set_xlim(zoom[0], zoom[1])

	if legend:
		recs = [mpl.patches.Rectangle((0,0),1,1, fc=c) for c in legend.values()]
		plt.legend(recs, legend.keys(), loc=lloc, fontsize=20)

	ax.set_title(trait, fontsize=24)
	plt.tight_layout()

	if save:
		plt.savefig(save)

	plt.show()

def add_legend(legend_dict, ax, lloc="lower left"):
	recs = [mpl.patches.Rectangle((0,0),1,1, fc=c) for c in legend_dict.values()]
	ax.legend(recs, legend_dict.keys(), loc=lloc, fontsize=20)
	return ax

def plotInSet(tt, set_color, edge_names, sample_names, birth_names, figsize=(10, 20), show=True, save=False):
	# Init the matplotlib figure
	fig, ax = plt.subplots(figsize=figsize, facecolor='w')

	# Set color of non-selected phylogeny pieces
	ns_color = 'lightgray'

	# Set color of vertical connecting edges
	vert_color = 'lightgray'

	# X coordinate is absolute time
	x_attr = lambda k: k.absoluteTime

	# Set node size
	s_func = lambda k: 50

	# Outline differently if selected vs. not
	c_func_outline = lambda k: 'black' if k.traits['name'] in sample_names else ns_color

	# Color edge differently if selected vs. not
	c_func_edge = lambda k: set_color if k.traits['name'] in edge_names else ns_color

	# Color node differently if selected vs. not
	c_func_node = lambda k: set_color if k.traits['name'] in sample_names else 'white'

	# Plot edges
	tt.plotTree(ax, x_attr=x_attr, colour=c_func_edge)

	# Plot leaf nodes
	tt.plotPoints(ax, x_attr=x_attr, size=s_func, colour=c_func_node, outline_colour = c_func_outline, zorder=100) ## plot circles at tips

	# Get birth nodes, and their coordinates
	birth_nodes = [k for k in tt.Objects if k.is_node() and len(k.children) > 1]
	birth_x = [k.absoluteTime for k in birth_nodes]
	birth_y = [k.y for k in birth_nodes]

	# Set outline, color, and size of birth nodes to match leaf nodes
	birth_colors = [set_color if k.traits['name'] in birth_names else 'white' for k in birth_nodes]
	birth_outlines = ['black' if k.traits['name'] in birth_names else ns_color for k in birth_nodes]
	birth_size = [50 for k in birth_nodes]

	# Plot birth nodes and their outlines
	ax.scatter(birth_x, birth_y, facecolor=birth_colors, edgecolors='none', s=birth_size, zorder=200)
	ax.scatter(birth_x, birth_y, facecolor=birth_outlines, edgecolors='none', s=[b*2 for b in birth_size], zorder=199)

	# Get all vertical connecting line segments
	vert = []
	for i, seg in enumerate(ax.collections[0].get_segments()):
		if len(seg) > 1:
			if seg[0, 1] != seg[1, 1]:
				vert.append(i)

	# Make them dashed and a different color
	vert_rgba = list(mpl.colors.to_rgba(vert_color))
	line_style = [(0.0, [1, 10]) if i in vert else ls for i, ls in enumerate(ax.collections[0].get_linestyle())]
	line_color = [vert_rgba if i in vert else c for i, c in enumerate(ax.collections[0].get_color())]
	ax.collections[0].set_linestyles(line_style)
	ax.collections[0].set_color(line_color)

	# ax.set_title(trait, fontsize=24)
	y_padding = tt.ySpan * .05
	ax.set_ylim(0 - y_padding, tt.ySpan + y_padding)
	plt.tight_layout()

	if save:
		plt.savefig(save)

	if show:
		plt.show()

	plt.close("all")

def plotInternalNodes(tt, ax, c_func, outline_func, size_func=lambda k: 50):
	internal_nodes = [k for k in tt.Objects if k.is_node()]
	node_x = [k.absoluteTime for k in internal_nodes]
	node_y = [k.y for k in internal_nodes]

	# Set outline, color, and size of nodes
	node_colors = list(map(c_func, internal_nodes))
	node_outlines = list(map(outline_func, internal_nodes))
	node_size = list(map(size_func, internal_nodes))

	# Plot birth nodes and their outlines
	ax.scatter(node_x, node_y, facecolor=node_colors, edgecolors='none', s=node_size, zorder=200)
	ax.scatter(node_x, node_y, facecolor=node_outlines, edgecolors='none', s=[b * 2 for b in node_size], zorder=199)

	return ax

def plotBirthNodes(tt, ax, c_func_birth_node, c_func_birth_outline, c_func_birth_size=lambda k: 50):
	# Get birth nodes, and their coordinates
	birth_nodes = [k for k in tt.Objects if k.is_node() and len(k.children) > 1]
	birth_x = [k.absoluteTime for k in birth_nodes]
	birth_y = [k.y for k in birth_nodes]

	# Set outline, color, and size of birth nodes to match leaf nodes
	birth_colors = list(map(c_func_birth_node, birth_nodes))
	birth_outlines = list(map(c_func_birth_outline, birth_nodes))
	birth_size = list(map(c_func_birth_size, birth_nodes))

	# Plot birth nodes and their outlines
	ax.scatter(birth_x, birth_y, facecolor=birth_colors, edgecolors='none', s=birth_size, zorder=200)
	ax.scatter(birth_x, birth_y, facecolor=birth_outlines, edgecolors='none', s=[b * 2 for b in birth_size], zorder=199)
	
	return ax

def plotInSetFunc(
	tt,
	c_func_edge, c_func_node, c_func_outline,
	c_func_birth_node, c_func_birth_outline,
	vert_color='lightgray', s_func=lambda k: 50,
	save=True, show=False,
):

	# Init the matplotlib figure
	fig, ax = plt.subplots(figsize=(10, 20), facecolor='w')

	# X coordinate is absolute time
	x_attr = lambda k: k.absoluteTime

	# Plot edges
	tt.plotTree(ax, x_attr=x_attr, colour=c_func_edge)

	# Plot leaf nodes
	tt.plotPoints(ax, x_attr=x_attr, size=s_func, colour=c_func_node, outline_colour=c_func_outline, marker="s", zorder=100)  ## plot circles at tips

	# Plot birth nodes

	# Get all vertical connecting line segments
	vert = []
	for i, seg in enumerate(ax.collections[0].get_segments()):
		if len(seg) > 1:
			if seg[0, 1] != seg[1, 1]:
				vert.append(i)

	# Make them dashed and a different color
	vert_rgba = list(mpl.colors.to_rgba(vert_color))
	line_style = [(0.0, [1, 10]) if i in vert else ls for i, ls in enumerate(ax.collections[0].get_linestyle())]
	line_color = [vert_rgba if i in vert else c for i, c in enumerate(ax.collections[0].get_color())]
	ax.collections[0].set_linestyles(line_style)
	ax.collections[0].set_color(line_color)

	# ax.set_title(trait, fontsize=24)
	y_padding = tt.ySpan * .05
	ax.set_ylim(0 - y_padding, tt.ySpan + y_padding)
	ax.set_xlabel("")
	ax.set_xticks([])
	ax.set_ylabel("")
	ax.set_yticks([])
	plt.tight_layout()

	if save:
		plt.savefig(save)
	if show:
		plt.show()

def plotTraitAx(ax, tt, edge_c_func, node_c_func, title, s_func=None, tip_names=False, birth_events=False, tips=True, zoom=False, birth_c_func=None):

	# X position of nodes will be absolute time
	x_attr = lambda k: k.absoluteTime

	if s_func:
		s_func = s_func
	elif tips:
		s_func = lambda k: 30
	else:
		s_func = lambda k: 0

	tpt = tt.plotTree(ax, x_attr=x_attr, colour=edge_c_func)
	tpp = tt.plotPoints(ax, x_attr=x_attr, size=s_func, colour=node_c_func, zorder=100)

	if tip_names:
		text_func = lambda k: k.name
		tt.addText(ax, x_attr=x_attr, text=text_func, color='k')

	if birth_events:
		# Get birth nodes, and their coordinates
		birth_nodes = [k for k in tt.Objects if k.is_node() and len(k.children) > 1]
		birth_x = [k.absoluteTime for k in birth_nodes]
		birth_y = [k.y for k in birth_nodes]

		# Set outline, color, and size of birth nodes to match leaf nodes
		birth_colors = [birth_c_func(k) for k in birth_nodes]
		# birth_outlines = ['black' for k in birth_nodes]
		birth_size = [50 for k in birth_nodes]

		# Plot birth nodes and their outlines
		ax.scatter(birth_x, birth_y, facecolor=birth_colors, edgecolors='none', s=birth_size, zorder=200)
		# ax.scatter(birth_x, birth_y, facecolor=birth_outlines, edgecolors='none', s=[b * 2 for b in birth_size], zorder=199)

	y_padding = tt.ySpan * .05
	x_padding = 4.261781308000001 * .05
	ax.set_ylim(0 - y_padding, tt.ySpan + y_padding)
	ax.set_xlim(tt.root.absoluteTime - tt.root.length - x_padding, tt.mostRecent + x_padding)

	if zoom:
		ax.set_xlim(zoom[0], zoom[1])

	ax.set_title(title)
	ax.set_xlabel("time")

	ax.set_yticks([])
	ax.set_yticklabels([])
	[ax.spines[loc].set_visible(False) for loc in ax.spines if loc not in ['bottom']]
	
	return ax

def get_center(center, min_trait_value, max_trait_value):
	min_max_offset = (np.abs(center - min_trait_value), np.abs(center - max_trait_value))

	# If lower bound is farthest from center value,
	# set upper bound to be equal distance
	# away from center as lower bound
	if min_max_offset[0] > min_max_offset[1]:
		min_val = min_trait_value
		max_val = center + min_max_offset[0]

	# If upper bound is farthest from center value,
	# set lower bound to be equal distance
	# away from center as upper bound
	elif min_max_offset[1] > min_max_offset[0]:
		max_val = max_trait_value
		min_val = center - min_max_offset[1]

	# Otherwise, if equal, already centered
	else:
		max_val = max_trait_value
		min_val = min_trait_value

	return min_val, max_val

def continuousFunc(trait_dict, trait, cmap='viridis', center=None, vmin=None, vmax=None, null_color="#eeeeee", norm="norm"):
	"""
	"""
	trait_values = list(set([t for t in trait_dict.values() if t != float("nan")]))
	
	if vmin:
		min_val = vmin
	else:
		min_val = min(trait_values)
	if vmax:
		max_val = vmax
	else:
		max_val = max(trait_values)

	if center:
		min_val, max_val = get_center(center, min_val, max_val)

	if isinstance(cmap, str):
		cmap = getattr(mpl.cm, cmap)
	else:
		cmap = cmap

	if norm == "norm":
		norm = mpl.colors.Normalize(min_val, max_val)
	elif norm == "log" or norm == "lognorm":
		norm = mpl.colors.LogNorm(min_val, max_val)

	def cFunc(k):
		try:
			return cmap(norm(trait_dict[k.traits[trait]]))
		except:
			return null_color

	c_func = lambda k: cFunc(k)
	return c_func, cmap, norm

def categoricalFunc(trait_dict, trait, legend=False, color_list=None, null_color="#eeeeee"):
	"""
	Output function that maps node/edge names to color corresponding to given trait
	"""
	trait_values = list(set(trait_dict.values()))

	if not color_list:
		color_list = list(mcd.XKCD_COLORS.values())

	nc = len(color_list)

	colors = {t: color_list[i % nc] for i, t in enumerate(trait_values)}
	c_func = lambda k: colors[trait_dict[k.traits[trait]]] if k.traits[trait] in trait_dict else null_color

	if legend:
		return colors, c_func
	else:
		return c_func

def removeLeavesBaltic(tt, remove_leaves):
	keep_leaves = [k for k in tt.Objects if k.is_leaf() and k.name not in remove_leaves]
	tt = tt.reduceTree(keep_leaves)

	return tt

def outputMatrix(tt, c_func, matrix, fname):
	fig = plt.subplots(figsize=(20, 20), facecolor='w')
	gs = GridSpec(1, 2, width_ratios=[3, 3], wspace=0.0)
	ax_tree = plt.subplot(gs[0])
	ax_matrix = plt.subplot(gs[1], sharey=ax_tree)
	matrix_names = matrix.columns
	n_traits = len(matrix_names)

	ax_tree = plotTraitAx(ax_tree, tt, c_func, "", tip_names=False, birth_events=False)

	# Iterate over branches
	for k in tt.Objects:
		if k.branchType=='leaf':
			for i, feat in enumerate(matrix_names):
				try:
					trait_val = matrix.loc[k.name][feat]

					# Get color
					c = 'darkgrey' if trait_val == "." else 'midnightblue'
				except:
					print(f"{feat} for {k.name} doesn't exist")
					trait_val = np.nan
					c = 'white'

				# Define rectangle with height and width 1, at y position of tip and at the index of the key
				lineage=plt.Rectangle((i, k.y-0.5), 1, 1, facecolor=c, edgecolor='none')

				# Add rectangle to plot
				ax_matrix.add_patch(lineage)

	ax_matrix.set_xticks(np.arange(0.5, n_traits + 0.5))
	ax_matrix.set_xticklabels(matrix_names, rotation=90)
	[ax_matrix.axvline(x, color='w') for x in range(n_traits)]

	ax_matrix.set_xlim(0, n_traits)
	y_padding = tt.ySpan * .05
	ax_tree.set_ylim(0 - y_padding, tt.ySpan + y_padding)

	"Turn axis spines invisible"
	[ax_tree.spines[loc].set_visible(False) for loc in ['top', 'right', 'left']]  ## no axes
	[ax_matrix.spines[loc].set_visible(False) for loc in ['top', 'right', 'left', 'bottom']]  ## no axes

	ax_tree.tick_params(axis='x', size=24)  ## no labels
	ax_matrix.tick_params(size=0, labelsize=14)
	ax_tree.set_yticklabels([])
	ax_matrix.xaxis.set_ticks_position('top')
	ax_tree.grid(axis='x')

	# plt.subplots_adjust(left=0.22)
	plt.savefig(fname, dpi=300)
	plt.show()

def plotColorPalette(color_list, show=True, save=False):
	"""
	Utility function to help decide on color palettes.
	Given a list of hex values or rgb tuples, show/save an image displaying them in order

	Parameters
	----------
	color_list : list of hex values or rgb tuples (can be mixed)
	show : bool that determines whether to display figure (e.g. for iPython or similar)
	save : string of where to save figure; defaults to False, which will not save
	
	# from https://stackoverflow.com/questions/58404270/draw-a-grid-of-colors-using-their-hex-values
	"""

	def hex_to_rgb(hex_value):
		h = hex_value.lstrip('#')
		return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

	rgb_color_list = [hex_to_rgb(c) if isinstance(c, str) else c for c in color_list]

	sns.palplot(rgb_color_list)

	ax = plt.gca()

	for i, _ in enumerate(rgb_color_list):
		ax.text(i, 0, i)

	if save:
		plt.savefig(save, dpi=300)

	if show:
		plt.show()

	plt.close("all")

def add_cmap_colorbar(ax, cmap, norm=None):
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

	return ax
	