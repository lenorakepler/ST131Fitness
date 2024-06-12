from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import transmission_sim.utils.plot_phylo_standalone as pp
import json
import numpy as np
import re
import seaborn as sns
from transmission_sim.ecoli.analyze import load_data_and_RO_from_file, dropbox_dir, analyses_dir, final_dir, writeup_dir
from transmission_sim.ecoli.branch_fitness import effects_to_fitness_tsim, site_effects_to_fitness_tsim

def color_tree_probability(tt, tdf, feature, out_file):
	plt.rcParams.update({'font.size': 20})
	
	fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(13, 30))

	fit_dict = tdf["1"].to_dict()

	c_func, cmap, norm = pp.continuousFunc(trait_dict=fit_dict, trait="name", cmap=sns.color_palette("flare", as_cmap=True), vmin=0, vmax=1)
	
	axs = pp.plotTraitAx(
		axs,
		tt,
		edge_c_func=c_func,
		node_c_func=c_func,
		tip_names=False,
		zoom=None,
		title=feature,
	)

	pp.add_cmap_colorbar(fig, axs, cmap, norm=norm)
	
	plt.tight_layout()
	plt.savefig(out_file)
	plt.close("all")

def plot_ancestral_probs(tt, pastml, out_dir):
	for trait_file in pastml.glob("marginal_probabilities.*.tab"):
		trait = trait_file.name.replace("marginal_probabilities.character_", "").replace(".model_F81.tab", "")
		tdf = pd.read_csv(trait_file, index_col=0, sep="\t")
		
		if "1" in tdf.columns:
			color_tree_probability(tt, tdf, trait, out_dir / f"{trait}.png")
		else:
			print(trait)

def get_feature_changepoints(tt, feature, features_file):
	anc_df = pd.read_csv(features_file, index_col=0)[feature]

	# Find "changepoints" where clades switch
	changepoints = []
	for node in tt.Objects:
		if node.branchType == 'node':
			node_name = node.traits['name'].split("_")[0]
			node_clade = anc_df[node_name]
			
			for child in node.children:
				child_name = child.traits['name'].split("_")[0]
				child_clade = anc_df[child_name]

				if child_clade != node_clade:
					change_date = node.absoluteTime
					changepoints.append(dict(
						change_date=change_date,
						parent_clade=node_clade,
						child_clade=child_clade,
						parent_name=node_name,
						child_name=child_name,
						str=f"{change_date:.1f}: {node_clade} -> {child_clade}",
						leaves=len(child.leaves) if child.branchType=="node" else 0,
						))

	# Save changepoints to csv
	change_dir = final_dir / "feature_changepoints"
	change_dir.mkdir(exist_ok=True, parents=True)

	changepoints = pd.DataFrame(changepoints)
	changepoints.to_csv(change_dir / f"{feature}.csv", index=False)

	changepoints.set_index("parent_name", inplace=True)

	# # Only display core divergences
	# changepoints = changepoints[changepoints["leaves"] > 10]

	print(f"\n{feature}")
	print(changepoints)

	return changepoints

def get_tree_probabilities(feature, pastml_dir):
	trait_file = pastml_dir / f"marginal_probabilities.character_{feature}.model_F81.tab"

	if trait_file.exists():
		tdf = pd.read_csv(trait_file, index_col=0, sep="\t")["1"]
	else:
		print(f"No ancestral file found for {feature} ({trait_file})")
		tdf = None

	return tdf

def plot_ancestral_ax(tt, feature, feature_dict, ax, color):
	c_func, cmap, norm = pp.continuousFunc(trait_dict=feature_dict.to_dict(), trait="name", cmap=sns.color_palette(f"blend:gainsboro,{color}", as_cmap=True), vmin=0, vmax=1)
	
	ax = pp.plotTraitAx(
		ax,
		tt,
		edge_c_func=c_func,
		node_c_func=c_func,
		tip_names=False,
		tips=False,
		zoom=None,
		title=feature,
	)

	return ax, cmap, norm

def plot_presences(tt, features_file, features_list, out_dir, pastml_dir=None, plot_changepoints=False, fname=None):
	n_features = len(features_list)
	plt.rcParams.update({'font.size': 20})

	fig, axs = plt.subplots(nrows=1, ncols=n_features, figsize=(13 * n_features, 30))
	if n_features == 1: 
		axs = [axs]
	else:
		axs = axs.ravel()

	if pastml_dir:
		probabilities = True

	else:
		probabilities = False
		features = pd.read_csv(features_file, index_col=0)

	for i, feature in enumerate(features_list):
		ax = axs[i]
		color = "darkblue"
		if probabilities:
			feature_dict = get_tree_probabilities(feature, pastml_dir)
		else:
			feature_dict = features[feature]

		if isinstance(feature_dict, pd.Series):
			ax, cmap, norm = plot_ancestral_ax(tt, feature, feature_dict, ax, color)

		if plot_changepoints:
			# -----------------------------------------------------
			# Annotate phylogeny with clade divergence times
			# -----------------------------------------------------
			change_file = final_dir / "feature_changepoints" / f"{feature}.csv"
			
			if change_file.exists():
				changepoints = pd.read_csv(change_file)
				changepoints.set_index("parent_name", inplace=True)
			else:
				changepoints = get_feature_changepoints(tt, feature, features_file)

			changepoints = changepoints.loc[~changepoints.index.duplicated()]


			# Set x and y text coordinates, positioning
			text_x_attr = lambda k: k.absoluteTime - 2
			text_y_attr = lambda k: k.y - 5
			kwargs = {'va': 'top', 'ha': 'right', 'size': 16}

			gain = changepoints[changepoints["child_clade"] == 1]
			loss = changepoints[changepoints["child_clade"] == 0]

			# Annotate acquisition events
			target_func = lambda k: k.traits['name'] in gain.index.to_list()
			text_func = lambda k: f"{gain.loc[k.traits['name'], 'change_date']:.1f}"
			tt.addText(ax, x_attr=text_x_attr, y_attr=text_y_attr, target=target_func, text=text_func, **kwargs)
			tt.plotPoints(ax, x_attr=lambda k: k.absoluteTime, y_attr=lambda k: k.y, target=target_func, size=36, colour="black")

			# Annotate loss events
			kwargs["color"] = "firebrick"
			target_func = lambda k: k.traits['name'] in loss.index.to_list()
			text_func = lambda k: f"{loss.loc[k.traits['name'], 'change_date']:.1f}"
			tt.addText(ax, x_attr=text_x_attr, y_attr=text_y_attr, target=target_func, text=text_func, **kwargs)
			tt.plotPoints(ax, x_attr=lambda k: k.absoluteTime, y_attr=lambda k: k.y, target=target_func, size=36, colour="firebrick")

	if not fname:
		features_str = "_".join([f.split("_")[0] for f in features_list])
		if probabilities:
			features_str += "_probabilties"
		fname = features_str + ".png"
	else:
		if ".png" not in fname:
			fname += ".png"

	out_dir.mkdir(exist_ok=True, parents=True)
	plt.tight_layout()
	plt.savefig(out_dir / fname, dpi=300)
	plt.close("all")

if __name__ == "__main__":	
	if Path("/home/lenora/Dropbox").exists():
		dropbox_dir = Path("/home/lenora/Dropbox")

	else:
		dropbox_dir = Path("/Users/lenorakepler/Dropbox")

	ncbi = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset"
	dir = ncbi / "final"

	pastml = dir / "pastml" / "lsd.date.noref_pastml"
	out_dir = pastml / "plots"
	out_dir.mkdir(exist_ok=True, parents=True)

	tree_file = dir / "named.tree_lsd.date.noref.pruned_unannotated.nwk"
	tt = pp.loadTree(
		tree_file,
		internal=True,
		abs_time=2023,
	)

	# plot_presences(tt, dir / "combined_ancestral_states_binary_grouped_diverse_uncorr.csv", 
	# 	["cnf-hly_VIR", "cva_VIR", "fde_VIR", "afa-nfa_VIR", "iss_VIR", "ssl_VIR", "sat_VIR", "mch_VIR"], 
	# 	out_dir
	# 	)

	# plot_presences(tt, dir / "combined_ancestral_states_binary_grouped_diverse_uncorr.csv", 
	# 	["Col_PLASMID", "IncX_PLASMID", "IncI_PLASMID", "IncFII_PLASMID"], 
	# 	out_dir
	# 	)

	# plot_presences(tt, dir / "combined_ancestral_states_binary_grouped_diverse_uncorr.csv", 
	# 	["gyr_AMR", "par_AMR", "aac_AMR", "blaCTX-M-27_AMR", "blaCTX-M-15_AMR", "tet(B)_AMR", "tet(A)_AMR"], 
	# 	out_dir
	# 	)

	all_ancestral_file = final_dir / "features" / "combined_ancestral_states_binary.csv"
	all_ancestral = pd.read_csv(all_ancestral_file, index_col=0)
	all_ancestral_features = all_ancestral.columns.to_list()

	ancestral_out = out_dir / "grouped"
	binary_out = out_dir / "binary" / "grouped"

	# plot_presences(tt, all_ancestral_file, [f for f in all_ancestral_features if "par" in f], binary_out, fname="all_par", pastml_dir=None, plot_changepoints=True)
	# plot_presences(tt, all_ancestral_file, [f for f in all_ancestral_features if "gyr" in f], binary_out, fname="all_gyr", pastml_dir=None, plot_changepoints=True)
	# plot_presences(tt, all_ancestral_file, [f for f in all_ancestral_features if "aac" in f], binary_out, fname="all_aac", pastml_dir=None, plot_changepoints=True)

	# for feature in ["gyr", "par", "aac", "IncFIA", "IncFIB", "IncFII", "Col", "IncX", "IncI", "blaCTXM27", "blaCTXM15", "blaCTXM14", "blaCTXM24", "blaOXA", "cvaC", "sslE", "tet", "amp"]:
	# 	plot_presences(tt, all_ancestral_file, [f for f in all_ancestral_features if feature in f], binary_out, fname=f"all_{feature}_changepoints", pastml_dir=None, plot_changepoints=True)

	# plot_presences(
	# 	tt, 
	# 	all_ancestral_file, 
	# 	["cnf1_VIR", "cvaC_VIR", "fdeC_VIR", "afaC_VIR"], 
	# 	binary_out, fname=f"pos_toxin-colonization_changepoints", pastml_dir=None, plot_changepoints=True
	# 	)

	# plot_presences(
	# 	tt, 
	# 	all_ancestral_file, 
	# 	["cnf1_VIR", "cvaC_VIR", "fdeC_VIR", "afaC_VIR"],
	# 	binary_out, fname=f"pos_toxin-colonization_changepoints", pastml_dir=None, plot_changepoints=True
	# 	)

	wanted = ['aac', 'glp', 'mdt', 'tet', 'blaOXA1']
	for feature in all_ancestral_features:
		if any([w in feature for w in wanted]):
			plot_presences(
				tt, 
				all_ancestral_file, 
				[feature],
				binary_out, fname=f"{feature}_changepoints", pastml_dir=None, plot_changepoints=True
				)

	# for feature in ['mer', 'iro', 'mch', 'pap', 'amp', 'iuc']:
	for feature in ['sul', 'sat']:
		plot_presences(
			tt, 
			all_ancestral_file, 
			[f for f in all_ancestral_features if feature in f], 
			binary_out, 
			fname=f"all_{feature}_presences", 
			pastml_dir=None, 
			plot_changepoints=False,
			)

	# plot_presences(
	# 	tt, 
	# 	all_ancestral_file, 
	# 	["sat_VIR", "mchB_VIR", "mchF_VIR"], 
	# 	binary_out, fname=f"neg_toxin_changepoints", pastml_dir=None, plot_changepoints=True
	# 	)

	# plot_presences(
	# 	tt, 
	# 	all_ancestral_file, 
	# 	["iss_VIR", "sslE_VIR"], 
	# 	binary_out, fname=f"serum-survival_changepoints", pastml_dir=None, plot_changepoints=True
	# 	)

	# plot_presences(
	# 	tt, 
	# 	dir / "combined_ancestral_states_binary_grouped_diverse_uncorr.csv", 
	# 	['IncFII_PLASMID', 'IncFIA_PLASMID', 'IncFIB_PLASMID'],
	# 	binary_out, fname="IncFClasses_changepoints", pastml_dir=None, plot_changepoints=True,
	# 	)

	# plot_presences(
	# 	tt, 
	# 	dir / "combined_ancestral_states_binary_grouped_diverse_uncorr.csv",
	# 	["blaCTX-M-27_AMR", "blaCTX-M-15_AMR", "blaCTX-M-14_AMR"],
	# 	writeup_dir / "figures", fname="bla_changepoints", pastml_dir=None, plot_changepoints=True,
	# 	)

	# # for feature in ['mer', 'iro', 'mch', 'pap', 'amp', 'iuc']:
	# for feature in ['pco', 'qnr', 'sil', 'ter', 'ybt', 'dfr', 'cat']:
	# 	plot_presences(
	# 		tt, 
	# 		all_ancestral_file, 
	# 		[f for f in all_ancestral_features if feature in f], 
	# 		binary_out, 
	# 		fname=f"all_{feature}_presences", 
	# 		pastml_dir=None, 
	# 		plot_changepoints=False,
	# 		)



	




