from pathlib import Path
import pandas as pd
import numpy as np
from data_prep.plot_correlation import plot_correlation_structured, plot_full_correlation, corr_heatmap
import re
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
import json

def group_corr(feat_df, corr_df, threshold, out_dir, filename):
	Path(out_dir).mkdir(exist_ok=True, parents=True)
	corr = corr_df.loc[corr_df.abs().apply(lambda row: (row >= threshold).any(), axis=1)].index

	if len(corr) > 1:
		# print(f"{len(corr)} > 0")
		plot_correlation_structured(None, feat_df[corr], sep=",", methods=["average"], filename=Path(out_dir) / f"{filename}_corr_gr_{threshold*100:.0f}.csv", save=False)

def filter_undiverse(so, threshold=0.005):
	# Remove features that are present in less than 0.5% of samples, or more than 99.5% of samples
	N = len(so)
	n = N * threshold
	print(f"{threshold * 100:.2f}% threshold is n={n}. Removing features with < {n} count or >= {N - n}")

	count_gr = so.sum(axis=0) > n
	count_lt = so.sum(axis=0) <= (N - n)
	count = count_gr & count_lt

	print(f"Removing {len(count)} features")

	feat = so.loc[:, count]
	return feat

def plot_corr_groups_auto():
	corr = pd.read_csv("_analysis/out/indiv_features_full-corr.csv", index_col=0)
	np.fill_diagonal(corr.values, 0)

	groups = [c[0:3] for c in corr.columns]
	group_names = list(set(groups))
	for group_name in group_names:
		group = [g for g, gn in zip(corr.columns, groups) if gn == group_name]

		for i in range(4):
			add_corr = []
			for g in group:
				add_corr += some_corr[g]

			group = list(set(group + add_corr))

		corr_df = corr.loc[group, group]
		feat_df = feat.loc[:, group]

		group_corr(feat_df, corr_df, 0, Path("_analysis/out/all_grouped"), group_name)
		group_corr(feat_df, corr_df, 0.5, Path("_analysis/out/gr50"), group_name)
		group_corr(feat_df, corr_df, 0.95, Path("_analysis/out/gr95"), group_name)

def plot_corr_groups_indiv(corr_file, feat_file, out_dir):
	out_dir.mkdir(exist_ok=True, parents=True)

	corr = pd.read_csv(corr_file, index_col=0)
	feat = pd.read_csv(feat_file, index_col=0)

	np.fill_diagonal(corr.values, 0)

	for threshold in [0.9, 0.95, 0.99]:
		corr_dict = {i: row[(row.abs() >= threshold) == True].index.to_list() for i, row in corr.iterrows()}
		poss_groups = [tuple(sorted(group + [i])) for i, group in corr_dict.items() if len(group) > 0]

		groups = set()
		for g in poss_groups:
			# print(f"\npossible group: {g}\n========")
			if g not in groups:
				curr_groups = list(groups)
				for gr in curr_groups:
					if len(set(gr).intersection(set(g))) > 0:
						
						print(f"existing group: {gr}")
						print(f"difference: {set(gr).symmetric_difference(set(g))}")
						print(f"NEW GROUP: {tuple(sorted(list(set(gr).union(set(g)))))}")
						print(f"---")
						groups.remove(gr)
						g = tuple(sorted(list(set(gr).union(set(g)))))

				groups.add(g)

		groups_dict = {}
		for g in groups:
			group = list(g)
			group_name = "+".join([g.rsplit("_", maxsplit=1)[0] for g in group])

			corr_df = corr.loc[group, group]
			feat_df = feat.loc[:, group]

			if any(feat_df.sum(axis=0) > 1):
				group_corr(feat_df, corr_df, threshold, out_dir / f"groups_corr_{threshold}", group_name)

			groups_dict[group_name] = group

		(out_dir / f"groups_{threshold}.yml").write_text(dump(groups_dict, Dumper=Dumper))

def make_new_feature_groups():
	ng = pd.read_csv("all_feature_info_man-group.csv")
	ng['Cat'] = ng['Category'].apply(lambda k: k.upper().replace('VIRULENCE', 'VIR'))
	ng['NFG'] = ng.apply(lambda row: row['New Feature Group'] + f"_{row['Cat']}", axis=1)
	ng['FN'] = ng.apply(lambda row: row['Feature Name'] + f"_{row['Cat']}", axis=1)

	feature_dict = {}
	flags = []
	for group, gdf in ng.groupby('NFG'):
		info = {c: gdf[c].unique() for c in ['Category', 'AMRFinder+ Type', 'Resistance Class']}
		
		the_type = [re.match(r".*\((.*)\)", str(t)) for t in info['AMRFinder+ Type']]
		info['AMRFinder+ Type'] = [m.group(1) for m in the_type if m]

		mul = {c: i for c, i in info.items() if len(i) > 1}
		if mul:
			flags.append({group: mul})

		group_dict = {
			'display name': group.rsplit("_", maxsplit=1)[0],
			'category': info['Category'][0],
			'members': gdf['Feature Name'].to_list(),
			'members_long': gdf['FN'].to_list(),
		}

		if not pd.isna(info['Resistance Class'][0]):
			group_dict['resistance'] = info['Resistance Class'][0]

		if info['AMRFinder+ Type']:
			group_dict['type'] = info['AMRFinder+ Type'][0]

		else:
			if info['Category'] == 'AMR':
				group_dict['type'] = 'Gene'
		
		feature_dict[group] = group_dict

	(dn / "new_groups.yml").write_text(dump(feature_dict, Dumper=Dumper))

def make_new_binaries():
	dir = Path("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/")
	samples = (dir / "data_filters" / "post_reconstruction_bioproject_pass.txt").read_text().splitlines()

	out = Path("_analysis/out")
	dn = Path("data_new")
	dn.mkdir(exist_ok=True, parents=True)

	plas = pd.read_csv(out / "plasmid_gr0.csv", index_col=0)
	plas = plas.loc[[s for s in samples if s in plas.index.to_list()], :]

	amrf = pd.read_csv(out / "amr_gr90.csv", index_col=0)
	amrf = amrf.loc[[s for s in samples if s in amrf.index.to_list()], :]

	print(len(plas))
	print(len(amrf))
	print(len(samples))

	feat = pd.concat([amrf, plas], axis=1).fillna(0).astype(int)

	groups = load((dn / "new_groups.yml").read_text(), Loader=Loader)

	make_new_feature_groups()

	binaries = []
	for group, gdict in groups.items():
		not_in_feat = [m for m in gdict['members_long'] if m not in feat.columns]
		gdf = feat[[m for m in gdict['members_long'] if m not in not_in_feat]]

		if len(gdf.columns) == 0:
			continue

		if "iroBCDEN_VIR" in group:
			gdf = (gdf[['iroC_VIR', 'iroD_VIR', 'iroN_VIR']].sum(axis=1) == 3).astype(int).to_frame()

		binary = (gdf.sum(axis=1) > 0).astype(int)
		binary.name = group
		binaries.append(binary)

	features = pd.concat(binaries, axis=1)
	features.to_csv(dn / "functional_group_features.csv")

	features = filter_undiverse(features, 0.005)
	features.to_csv(dn / "functional_group_features_above-half-percent.csv")

if __name__ == "__main__":
	dir = Path("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/")
	samples = (dir / "data_filters" / "post_reconstruction_bioproject_pass.txt").read_text().splitlines()

	out = Path("_analysis/out")

	dn = Path("data_new")
	dn.mkdir(exist_ok=True, parents=True)

	# make_new_binaries()

	# plot_full_correlation(dn / "functional_group_features_above-half-percent.csv", sep=",")
	# plot_full_correlation(dn / "functional_group_features.csv", sep=",")

	# plot_correlation_structured(dn / "functional_group_features_above-half-percent_full-corr.csv", dn / "functional_group_features_above-half-percent.csv", sep=",", methods=["average"])
	# plot_correlation_structured(dn / "functional_group_features_full-corr.csv", dn / "functional_group_features.csv", sep=",", methods=["average"])

	# plot_corr_groups_indiv(dn / "functional_group_features_above-half-percent_full-corr.csv", dn / "functional_group_features_above-half-percent.csv", dn / "functional_group_correlations")

	# print(f"not in groupings: {[f for f in feat.columns if f not in ng['FN'].to_list()]}")

	# Ancestral states file
	# all_anc = pd.read_csv("../data/combined_ancestral_states_binary.csv", index_col=0)

	# Samples only
	# so = all_anc.loc[[i for i in all_anc.index if 'SAMN' in i], :]

	# so = pd.read_csv("_analysis/out/amr_gr90.csv", index_col=0)

	# feat = so
	# plot_full_correlation(feat, sep=",", filename="_analysis/out/indiv_features.csv")
	# plot_correlation_structured("_analysis/out/indiv_features_full-corr.csv", feat, sep=",", methods=["average"], filename="_analysis/out/indiv_features.csv")

	# some_corr = {i: row[(row.abs() > 0.8) == True].index.to_list() for i, row in corr.iterrows()}






