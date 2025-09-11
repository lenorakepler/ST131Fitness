import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import re
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
import json
from _analysis.plot_ancestral import plot_presences
import analysis.plot_phylo_standalone as pp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib as mpl
import matplotlib.ticker as ticker
from pastml.acr import pastml_pipeline
from pastml.ml import MARGINAL_PROBABILITIES, is_ml, is_marginal, MPPA, ml_acr, \
    ML_METHODS, MAP, JOINT, ALL, ML, META_ML_METHODS, MARGINAL_ML_METHODS, get_default_ml_method
from pastml.models import MODEL, SCALING_FACTOR, SMOOTHING_FACTOR
from pastml.models.CustomRatesModel import CustomRatesModel, CUSTOM_RATES
from pastml.models.EFTModel import EFTModel, EFT
from pastml.models.F81Model import F81Model, F81
from pastml.models.HKYModel import HKYModel, HKY, HKY_STATES
from pastml.models.JCModel import JCModel, JC
from pastml.models.JTTModel import JTTModel, JTT, JTT_STATES
from pastml.parsimony import ACCTRAN, DELTRAN, DOWNPASS, MP
from _analysis.plot_correlation import plot_correlation_structured
from _analysis.corr_group import filter_undiverse

model_types = dict(
	JC=JC, 
	F81=F81, 
	EFT=EFT, 
	HKY=HKY, 
	JTT=JTT, 
	CUSTOM_RATES=CUSTOM_RATES,
	)

prediction_methods = dict(
	MPPA=MPPA, # RECOMMENDED - Marginal Posterior Probabilities Approximation (ML), keeps a subset of
			   # likely states for each node that minimizes the prediction error measured by the Brier score.
	MAP=MAP, # Maximum a Posteriori (ML), chooses predicted states based on all possible scenarios
	JOINT=JOINT, # (ML), Reconstructs the states of the scenario with the highest likelihood
	DOWNPASS=DOWNPASS, # (MP)
	ACCTRAN=ACCTRAN, # (MP)
	DELTRAN=DELTRAN, # (MP)
	COPY='COPY', # Keep the annotated character states as-is without inference
	ALL=ALL, # Use all methods
	ML=ML, # Use all Maxiumum Likelhood methods
	MP=MP, # Use all Maximum Parsimony methods
	)

def to_pastml(features_file, out_dir, threshold=0.005):
	name = features_file.stem

	# Filter to only those that pass count threshold
	df = pd.read_csv(features_file, index_col=0)
	df = filter_undiverse(df, threshold=threshold)
	df.to_csv(out_dir / f"{name}_{threshold}.csv")

	# Strip special characters for use in pastml
	to_pastml = {f: re.sub(r"[\'/\\\"\)\(\-\+]", "", f) for f in df.columns}
	(out_dir / f"{name}_pastml-dict.yml").write_text(dump(to_pastml, Dumper=Dumper))
	
	# Make sure stripping special characters doen't produce duplicate keys
	if len(set(to_pastml.keys())) == len(set(to_pastml.values())):
		df = df.rename(columns=to_pastml)
		df.to_csv(out_dir / f"{name}_{threshold}_for-pastml.csv")

def threshold_genetic_for_pastml():
	dir = Path("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/")
	samples = (dir / "data_filters" / "post_reconstruction_bioproject_pass.txt").read_text().splitlines()

	out = Path("_analysis/out")

	dn = Path("data_new")
	dn.mkdir(exist_ok=True, parents=True)

	plas = pd.read_csv(out / "plasmid_gr0.csv", index_col=0)
	plas = plas.loc[[s for s in samples if s in plas.index.to_list()], :]

	amrf = pd.read_csv(out / "amr_gr90.csv", index_col=0)
	amrf = amrf.loc[[s for s in samples if s in amrf.index.to_list()], :]

	feat = pd.concat([amrf, plas], axis=1).fillna(0).astype(int)
	feat.to_csv(dn / "genetic_features_ungrouped.csv")

	cols = feat.columns.to_list()

	to_pastml = {f: re.sub(r"[\'/\\\"\)\(-]", "", f) for f in cols}
	(dn / "pastml_conversion.yml").write_text(dump(to_pastml, Dumper=Dumper))
	
	print(f"No Duplicates? {len(set(to_pastml.keys())) == len(set(to_pastml.values()))}")

	feat = feat.rename(columns=to_pastml)
	feat.to_csv(dn / "genetic_features_ungrouped_forpastml.csv")

def concat_params(dir, do_feature_count=True):
	dir = Path(dir)

	if do_feature_count:
		features_file = list(dir.parent.glob("tip_features_*"))[0]
		features = pd.read_csv(features_file, index_col=0)
		feature_count = features.sum(axis=0)
		feature_count.name = "count"

	col_dtypes = {
		"character": str,
		"log_likelihood": float,
		"log_likelihood_restricted_JOINT": float,
		"log_likelihood_restricted_MAP": float,
		"log_likelihood_restricted_MPPA": float,
		"num_scenarios": int,
		"num_states_per_node_avg": float,
		"num_unresolved_nodes": int,
		"percentage_of_unresolved_nodes": float,
		"method": str,
		"model": str,
		"num_nodes": int,
		"num_tips": int,
		"scaling_factor": float,
		"state_changes_per_avg_branch": float,
		"smoothing_factor": float,
		"0": float,
		"1": float,
	}

	def try_convert(k):
		max = 1000000
		try:
			if int(k) > max:
				return max
			else:
				return int(k)
		except:
			return max

	df = pd.concat([pd.read_csv(f, index_col=0, sep="\t") for f in dir.glob("params.character_*")], axis=1).T
	df["num_scenarios"] = df["num_scenarios"].apply(lambda k: try_convert(k))
	df = df.drop(columns=["num_nodes", "num_tips", "pastml_version"])

	col_dtypes = {c: val for c, val in col_dtypes.items() if c in df.columns}
	df = df.astype(col_dtypes)
	df = df.rename(columns = {c: c.replace("log_likelihood_restricted", "llr") for c in df.columns})
	df = df.rename(columns = {
		"smoothing_factor": "smoothing", 
		"scaling_factor": "scaling", 
		"num_states_per_node_avg": "states_per_node", 
		"num_scenarios": "scenarios",
		"num_unresolved_nodes": "unresolved",
		"state_changes_per_avg_branch": "states_per_branch",
		"percentage_of_unresolved_nodes": "pct_unresolved",
		})
	df = df.set_index("character")
	df = df.sort_values(by='unresolved', ascending=False)

	if do_feature_count:
		df = pd.concat([df, feature_count], axis=1)

	df.to_csv(dir.parent / "params.csv")

def concat_marginal_states(dir):
	dir = Path(dir)

	prob_files = list(dir.glob("marginal_probabilities.character_*"))
	
	one_probs = []
	for f in prob_files:
		fdf = pd.read_csv(f, sep="\t", index_col=0)
		feature = re.search(r"character_(.*?)\.", f.name).group(1)
		if fdf.columns.to_list() == ['0', '1']:
			one_prob = fdf['1']
			one_prob.name = feature
			one_probs.append(one_prob)
		else:
			fdf = fdf.rename(columns={c: f"{feature}_{c}" for c in fdf.columns})
			one_probs.append(fdf)

	if len(one_probs) > 1:
		df = pd.concat(one_probs, axis=1)
	else:
		df = one_probs[0]

	df.to_csv(dir / "marginal_states.csv")

def plot_uncertain_features(pastml_dir, features_file, tree_file):
	tt = pp.loadTree(
		tree_file,
		internal=True,
		abs_time=2023,
	)
	
	unc = concat_params(pastml_dir, features_file)
	unc = unc.iloc[0:5]

	out_dir = pastml_dir.parent / f"plot_{pastml_dir.name}"
	
	plot_presences(tt, features_file, unc.index.to_list(), out_dir, pastml_dir=pastml_dir, plot_changepoints=False, fname=None)

def run_pastml(out_dir, tree_file, features_file,
			  threads=16, forced_joint=False, map_viz=True, tree_viz=True, 
			  prediction_method="MPPA", model="F81", verbose=True, **pastml_kwargs):
	
	out_dir = Path(out_dir)
	out_dir.mkdir(exist_ok=True, parents=True)

	work_dir = out_dir / "work"
	work_dir.mkdir(exist_ok=True, parents=True)

	shutil.copy(tree_file, out_dir / f"tree_{Path(tree_file).name}")
	shutil.copy(features_file, out_dir / f"tip_features_{Path(features_file).name}")

	map_viz_path = out_dir / "pastml_map_viz.html" if map_viz else None
	tree_viz_path = out_dir / "pastml_tree_viz.html" if tree_viz else None

	prediction_method = prediction_methods[prediction_method]
	model = model_types[model]

	pastml_pipeline(
		tree=tree_file, data=features_file, data_sep=',', threads=threads, work_dir=work_dir, out_data=out_dir / "combined_ancestral_states.tab",
		prediction_method=prediction_method, model=model, html_compressed=map_viz_path, html=tree_viz_path, verbose=verbose,
		**pastml_kwargs,
	)

def unannotate(pastml_dir):
	pastml_dir = Path(pastml_dir)

	tree_file_annot = list(pastml_dir.glob("work/named.tree_*"))[0]
	tree_unannot = re.sub(r"\[.*?\]", "", tree_file_annot.read_text())
	(pastml_dir / "pastml_tree.nwk").write_text(tree_unannot)

def convert_tabs(pastml_dir):
	pastml_dir = Path(pastml_dir)
	df = pd.read_csv(pastml_dir / "combined_ancestral_states.tab", index_col=0, sep="\t")
	df.to_csv(pastml_dir / "combined_ancestral_states.csv")

	ms = pd.read_csv(pastml_dir / "work" / "marginal_states.csv", index_col=0)

	# Unresolved nodes are those with multiple rows
	unresolved = df[df.index.duplicated(keep=False)]

	# Enumerate unresolved features of each node
	for s, sdf in unresolved.groupby("node"):
		n_poss = len(sdf)
		ur = sdf.loc[:, sdf.notna().all(axis=0)].astype(int)
		ur.loc["prob", :] = ms.loc[s, ur.columns]

		# urf = ur.loc[:, ~ur.isna().all(axis=0)]
		print(ur)

	unresolved = df[df.isna().any(axis=1)]

	na = unresolved.sum(axis=0)
	na = na[na > 0].sort_values(ascending=False)
	na.name = "num_unresolved"
	na.to_csv(pastml_dir / "work" / "unresolved_by_feature.csv")

	na = unresolved.sum(axis=1)
	na = na[na > 0].sort_values(ascending=False)
	na.name = "num_unresolved"
	na.to_csv(pastml_dir / "work" / "unresolved_by_node.csv")

def find_nb_tips(pastml_dir):
	pastml_dir = Path(pastml_dir)

	df = pd.read_csv(pastml_dir / "work" / "marginal_states.csv")
	df = df[df['node'].str.contains('SAMN') == True]

	df = df.set_index('node')
	nb = df[df.isin([1, 0]).all(axis=1) == False]
	nb = nb.loc[:, nb.isin([1, 0]).all(axis=0) == False]

	nb_round = nb.round(decimals=0).astype(int)

	features_file = list(pastml_dir.glob("tip_features_*"))[0]
	feat = pd.read_csv(features_file, index_col=0)
	feat = feat.rename(columns={c: re.sub(r"[\'/\\\"\)\(-]", "", c) for c in feat.columns})
	feat = feat.loc[nb.index, nb.columns]

	tab = pd.read_csv(pastml_dir / "combined_ancestral_states.csv", index_col=0)
	tab = tab.loc[nb.index, nb.columns]

	print(feat.eq(nb_round))
	print(tab.eq(feat))

def check_trees(pastml_dir):
	pastml_dir = Path(pastml_dir)

	tt_in = pp.loadTree(
		list(pastml_dir.glob("tree_*"))[0],
		internal=True,
		abs_time=2023,
	)

	tt = pp.loadTree(
		pastml_dir / "pastml_tree.nwk",
		internal=True,
		abs_time=2023,
	)

	nodes_in = {k.traits['name']: k.length for k in tt_in.Objects}
	nodes_out = {k.traits['name']: k.length for k in tt.Objects}

	for node, nl in nodes_in.items():
		ol = nodes_out[node]
		if not np.isclose(ol, nl, atol=1e-5):
			print(f"{node}: {nl} vs. {ol}")

def collapse_corr(corr_yaml, feat_file, out_dir):
	out_dir = Path(out_dir)

	corr_groups = load(Path(corr_yaml).read_text(), Loader=Loader)
	features = pd.read_csv(feat_file, index_col=0)

	for group, members in corr_groups.items():
		member_bin = features.loc[:, members]
		cats = '/'.join(sorted(list(set([f.rsplit("_", maxsplit=1)[-1] for f in members]))))
		new_name = f"{group}_{cats}"

		# If any have the feature, set as 1
		new_bin = (member_bin.sum(axis=1) > 0).astype(int)
		new_bin.name = new_name

		features = features.drop(columns=members)
		features = pd.concat([features, new_bin], axis=1)

	features.to_csv(out_dir / "corr_grouped_binary_features.csv")

	plot_correlation_structured(None, features, filename=out_dir / "corr_grouped_binary_features", include_count=True, save=True, plot=True)

def _examine_marginal_binary(pastml_dir):
	dir = Path(pastml_dir)

	marginal_file = dir / "work" / "marginal_states.csv"
	
	if not marginal_file.exists():
		concat_marginal_states(dir / "work")

	mar = pd.read_csv(marginal_file, index_col=0)
	bins = pd.read_csv(dir / "combined_ancestral_states.tab", index_col=0, sep="\t")

	# Get map of unresolved node/feature combos
	unr_map = pd.DataFrame(index=bins.index[~bins.index.duplicated(keep="first")], columns=bins.columns).fillna(False)

	# Set unr_map to True at unresolved node/feature pairs
	unresolved = bins[bins.index.duplicated(keep=False)]
	extra = unresolved[unresolved.isna().any(axis=1)]
	unr_map.loc[extra.index, extra.columns] = unresolved[unresolved.isna().any(axis=1)].isna() == False

	extra = extra.loc[:, ~extra.isna().all(axis=0)]

	nb = mar[mar.isin([1, 0]).all(axis=1) == False]
	nb = nb.loc[:, nb.isin([1, 0]).all(axis=0) == False]
	tips = nb[nb.index.str.contains('SAMN')==True]

	nb_tips = (~tips.isin([1, 0])).sum(axis=1)
	nb_feat = (~tips.isin([1, 0])).sum(axis=0)

	for t, row in tips.iterrows():
		tnb = row[row.isin([1, 0]) == False]
		tbin = bins.loc[t, tnb.index]
		tvals = pd.concat([tnb, tbin], axis=1)
	
	## Binary vs. marginal
	mar_nb = mar.round(decimals=0).astype(int)
	eq = mar_nb == bins.loc[~bins.index.duplicated(keep="first"), mar_nb.columns]

	for i, row in (~eq.loc[~eq.all(axis=1)]).iterrows():
		brow = bins.loc[i, row[row == True].index]
		if len(brow) == 1:
			print(brow)

def count_unresolved(pastml_dir):
	dir = Path(pastml_dir)
	bins = pd.read_csv(dir / "combined_ancestral_states.tab", index_col=0, sep="\t")
	unresolved = bins[bins.index.duplicated(keep=False)]
	extra = unresolved[unresolved.isna().any(axis=1)]

	extra.notna().astype(int).sum(axis=0).sort_values(ascending=False).to_csv(pastml_dir / "unresolved_by_node.csv")
	extra.notna().astype(int).sum(axis=1).sort_values(ascending=False).to_csv(pastml_dir / "unresolved_by_feature.csv")

def marginal_for_analysis(pastml_dir, pastml_dict_file, meta=False, drop_first=False):
	dir = Path(pastml_dir)

	marginal_file = dir / "work" / "marginal_states.csv"
	
	if not marginal_file.exists():
		concat_marginal_states(dir / "work")

	mar = pd.read_csv(marginal_file, index_col=0)

	# We need to override marginal states of tips because it allows to be diff than specified
	features_file = list(dir.glob("tip_features_*"))[0]
	features = pd.read_csv(features_file, index_col=0)
	tip_features = features.loc[features.index.str.contains("SAMN") == True]

	if meta:
		# Convert tip states to dummy to match marginal
		tip_features = pd.get_dummies(tip_features, drop_first=drop_first).astype(int)

	missing = [i for i in tip_features.index if i not in mar.index]

	# Get only sampled tips
	tip_features = tip_features.loc[[i for i in tip_features.index if i in mar.index], :]

	# Save and remove dropped column from marginal
	removed = set(mar.columns.to_list()) ^ set(tip_features.columns.to_list())
	removed_bioproject = removed.pop()
	(dir / "reference_bioproject.txt").write_text(removed_bioproject)

	mar = mar[tip_features.columns]

	# Update marginal tips with known feature value
	mar.loc[tip_features.index, tip_features.columns] = tip_features

	if pastml_dict_file:
		# Change names back to allow special characters
		disp_names = load(Path(pastml_dict_file).read_text(), Loader=Loader)
		disp_names = {v: k for k, v in disp_names.items()}
		mar = mar.rename(columns=lambda c: disp_names[c])

	if meta:
		mar = mar.rename(columns=lambda c: c.rsplit("_", maxsplit=1)[-1] + "_META")

	if 'specimen' in tip_features.columns[0]:
		mar = mar.drop(columns=['urine_META'])

	mar.to_csv(dir / "marginal_states.csv")

def _examine_marginal_cat(pastml_dir):
	dir = Path(pastml_dir)

	marginal_file = dir / "work" / "marginal_states.csv"
	
	if not marginal_file.exists():
		concat_marginal_states(dir / "work")

	mar = pd.read_csv(marginal_file, index_col=0)
	mar = mar.rename(columns=lambda c: c.replace("bioproject_id_META_", ""))
	bins = pd.read_csv(dir / "combined_ancestral_states.tab", index_col=0, sep="\t")

	unresolved = bins[bins.index.duplicated(keep=False)]
	unr_dict =  {n: ndf['bioproject_id_META'].to_list() for n, ndf in unresolved.groupby("node")}

	for n, features in unr_dict.items(): print(mar.loc[n, features])

	extra = unresolved[unresolved.isna().any(axis=1)]
	extra = extra.loc[:, ~extra.isna().all(axis=0)]

	mar = mar.loc[extra.index, extra.columns]

	marginal_values = []
	for node, row in extra.iterrows():
		uncertain_features = row[row.isna()==False]
		values = mar.loc[node, uncertain_features.index]
		marginal_values += values.to_list()

	m = mar[mar.index.str.contains('SAMN') == True]
	m = m.replace(1, True)
	print(m.loc[~(m == True).any(axis=1), :])

# def meta_to_dummy(bioproject_file, specimen_file):
# 	bp = pd.read_csv(bioproject_file, index_col=0)
# 	sp = pd.read_csv(specimen_file, index_col=0)

# 	feat = pd.concat([bp, sp], axis=1)

# 	feat.drop(columns=['urine', 'specimen_type_META'], inplace=True)

# 	proj_id = pd.get_dummies(feat['bioproject_id_META'])
# 	proj_id.rename(mapper=lambda c: c + "_META", axis=1, inplace=True)
# 	feat = pd.concat([feat, proj_id], axis=1)
# 	feat.drop(columns=["bioproject_id_META"], inplace=True)

# 	feat.to_csv(output[0])

def concat_meta_features(out_file, *feature_files):
	df = pd.concat([pd.read_csv(f, index_col=0) for f in feature_files], axis=1)
	df.to_csv(out_file)

	print(df)

if __name__ == "__main__":
	from _analysis.corr_group import plot_corr_groups_indiv

	dn = Path("data_new")

	if False:
		plot_corr_groups_indiv(corr_file=dn / "functional_group_features_full-corr.csv", feat_file=dn / "functional_group_features.csv", out_dir=dn / "functional_groups_corr")

	if False:
		collapse_corr(dn / "functional_groups_corr" / "groups_0.95.yml", dn / "functional_group_features.csv", out_dir=dn / "functional_groups_corr")

	if False:
		run_pastml(
			out_dir=dn / "meta_features_bp",
			tree_file="named.tree_lsd.date.noref.pruned_unannotated.nwk", 
			features_file="/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/meta/meta_features_bponly.csv",
			prediction_method="MPPA",
			)

	if False:
		run_pastml(
			out_dir=dn / "meta_features_specimen",
			tree_file="named.tree_lsd.date.noref.pruned_unannotated.nwk", 
			features_file="/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/meta/meta_features_specimenonly.csv",
			prediction_method="MPPA",
			)

	if False:
		to_pastml(dn / "functional_groups_corr" / "corr_grouped_binary_features.csv", dn / "functional_groups_corr", threshold=0.005)

	if False:
		run_pastml(
			out_dir=dn / "functional_groups_corr_pastml",
			tree_file="named.tree_lsd.date.noref.pruned_unannotated.nwk", 
			features_file=dn / "functional_groups_corr" / "corr_grouped_binary_features_0.005_for-pastml.csv",
			prediction_method="MPPA",
			)

	if False:
		marginal_for_analysis(dn / "functional_groups_corr_pastml", dn / "functional_groups_corr" / "corr_grouped_binary_features_pastml-dict.yml")

	if False:
		count_unresolved(dn / "functional_groups_corr_pastml")
		count_unresolved(dn / "meta_features_specimen")

		concat_params(dn / "meta_features_specimen/work", do_feature_count=True)
		concat_params(dn / "meta_features_bp/work", do_feature_count=True)
		concat_params(dn / "functional_groups_corr_pastml/work", do_feature_count=True)
		concat_params(dn / "named.tree_lsd.date.noref.pruned_unannotated_pastml/work", do_feature_count=True)
		

	if False:
		marginal_for_analysis(dn / "meta_features_specimen", None, meta=True)
		marginal_for_analysis(dn / "functional_groups_corr_pastml", None, meta=False)

	if True:
		marginal_for_analysis(dn / "meta_features_bp", None, meta=True, drop_first=True)
		
	if True:
		concat_meta_features(dn / "meta_features_marginal.csv", dn / "meta_features_bp" / "marginal_states.csv", dn / "meta_features_specimen" / "marginal_states.csv")

	def pal_plot(color_list, size=1, n=20):
		_, ax = plt.subplots(1, 1, figsize=(n * size, size))
		ax.imshow(np.arange(n).reshape(1, n),
				  cmap=LinearSegmentedColormap.from_list("new", color_list),
				  interpolation="nearest", aspect="auto")
		ax.set_xticks(np.arange(n) - .5)
		ax.set_yticks([-.5, .5])

		# Ensure nice border between colors
		ax.set_xticklabels(["" for _ in range(n)])

		# The proper way to set no ticks
		ax.yaxis.set_major_locator(ticker.NullLocator())

	# pal_plot(colors)
	# plt.savefig("sns_coolwarm.png", dpi=300)

	# mp = pd.read_csv(dn / "named.tree_lsd.date.noref.pruned_unannotated_pastml" / "marginal_probabilities.character_mphA_AMR.model_F81.tab", sep="\t", header=None)
	# mp = mp[mp[0].str.contains("SAMN") == True]
	# mp = mp[mp[1] != 1]
	# mp = mp[mp[1] != 0]
	# print(mp)

	# features = pd.read_csv(dn / "genetic_features_ungrouped_forpastml.csv", index_col=0)
	# print(features.loc["SAMN02801877", "mphA_AMR"])
	
	# feat = pd.read_csv(dn / "genetic_features_ungrouped_forpastml.csv", index_col=0)
	# feat = feat.loc[mp[0], :]
	# print(feat)
