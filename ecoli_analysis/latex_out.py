import json
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from transmission_sim.ecoli.analyze import load_data_and_RO_from_file, dropbox_dir
from transmission_sim.utils.commonFuncs import ppp
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
import yaml
import re

dir = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset" / "final"

tex_dir = dropbox_dir / "NCSU/Lab/Writing/st131_git"

def is_significant(c, df):
	c_min = df.loc[c, 'lower_CI']
	c_max = df.loc[c, 'upper_CI']
	if (c_min > 1) and (c_max > 1):
		return True
	elif (c_min < 1) and (c_max < 1):
		return True
	else:
		return False

def mle_to_tex(tex, group, i, row):
	if group == "VIR/STRESS":
		tex += f"\n\\subparagraph{{\\gene{{{i}}}}}\n\\gene{{{i}}} ({row['feature_type'].lower()}) MLE \\num{{{row['mle']}}} \\ci{{{row['lower_CI']}}}{{{row['upper_CI']}}}"
	else:
		tex += f"\n\\subparagraph{{\\gene{{{i}}}}}\n\\gene{{{i}}} MLE \\num{{{row['mle']}}} \\ci{{{row['lower_CI']}}}{{{row['upper_CI']}}}"
	return tex + "\n"

def mle_text(results_dir):
	df_all = pd.read_csv(results_dir / "profile_CIs.csv", index_col=0)
	df_all['sig'] = [is_significant(c, df_all) for c in df_all.index]
	df_all['non_one'] = df_all.loc[:, 'mle'].between(.999, 1.001) == False

	df = df_all[df_all['sig'] == True]

	df['feature_type'] = [i.split("_")[1] for i in df.index]
	df['feature_type_g'] = ["VIR/STRESS" if ft == 'VIR' or ft == "STRESS" else ft for ft in df['feature_type']]
	df['feature_name'] = [i.split("_")[0] for i in df.index]
	df['abs_mle'] = [1 - m if m < 1 else m - 1 for m in df['mle']]

	df.set_index('feature_name', inplace=True)

	tex = ''
	for group, gdf in df.groupby('feature_type_g'):
		tex += f"\n{group}\n"
		

		gdfb = gdf[gdf['mle'] >= 1].sort_values(by="mle", ascending=False)
		gdfd = gdf[gdf['mle'] < 1].sort_values(by="mle", ascending=True)

		tex += "\nbeneficial\n"
		for i, row in gdfb.iterrows():
			tex = mle_to_tex(tex, group, i, row)

		tex += "\ndeleterious"
		for i, row in gdfd.iterrows():
			tex = mle_to_tex(tex, group, i, row)

	# print(tex)
	(results_dir / "mle_ci_text.tex").write_text(tex)

	return df_all

def feature_info_table_old(results_dir):
	fd = dir / "features"

	# Confidence intervals + MLE
	ci = mle_text(dir / "analysis" / "3-interval_constrained-sampling")

	# Correlation groups
	cg = json.loads((fd / "correlation_groups.json").read_text())
	
	cg_rev = {}
	for group, members in cg.items():
		for member in members:
			cg_rev[member] = group

	# Clade designations
	clades_df = pd.read_csv(dir / "ST131_Typer" / "ST131_Typer_Summary.csv", index_col=0)
	clades_df.drop('SAMN11319293', inplace=True)

	# Grouped binary features
	ftg = pd.read_csv(fd / "combined_ancestral_states_binary_grouped.csv", index_col=0)

	# Raw binary features
	ft = pd.read_csv(fd / "combined_ancestral_states_binary.csv", index_col=0)
	
	# Dropped features
	dropped = (fd / "undiverse_dropped_features.txt").read_text().splitlines()

	# Features info
	df = pd.read_csv(fd / "features_dict_info.csv")
	df['Dropped'] = [g in dropped for g in df['group_name']]

	# Subset to same samples
	samples = clades_df.index.to_list()
	clades = clades_df.loc[samples, :]
	ftg = ftg.loc[samples, :]
	ft = ft.loc[samples, :]

	profiles = {pt: clades_df[clades_df['PCR_Profile_Type']==pt].index.to_list() for pt in clades_df['PCR_Profile_Type'].unique()}
	clades = {clade: clades_df[clades_df['Clade']==clade].index.to_list() for clade in clades_df['Clade'].unique()}
	# cp = {**clades, **profiles}
	cp = clades

	groups = []
	for g, gdf in df.groupby('group_name'):
		is_dropped = g in dropped

		groups.append({'group_name': g, 'raw_feature_name': f'000 {g.split("_")[0]} Grouped', 'feature_type': g.split("_")[1], 'pastml_name': float("nan"), 'dropped': g in dropped})

	df = pd.concat([df, pd.DataFrame(groups)], axis=0, ignore_index=True)
	df['Count'] = 0
	df['MLE'] = ''
	df['Sig'] = ''

	for clade_prof_name in cp.keys():
		df[clade_prof_name] = 0

	for i, row in df.iterrows():
		try:
			if '000' in row['raw_feature_name']:
				if row['dropped']:
					count = "-"
					df.loc[i, 'MLE'] = "-"
					df.loc[i, 'Sig'] = False

				else:
					group_name = row['group_name']
					if group_name in cg_rev:
						group_name = cg_rev[group_name]

					df.loc[i, 'MLE'] = ci.loc[group_name, 'mle']
					df.loc[i, 'Sig'] = ci.loc[group_name, 'sig']
					
					count = ftg.loc[:, row['group_name']].sum()
					
					for p, p_samples in cp.items():
						p_count = ftg.loc[p_samples, row['group_name']].sum()
						p_pct = p_count / len(p_samples) * 100
						df.loc[i, p] = f"{np.round(p_pct, 1)}"

					df.loc[i, 'group_name'] = group_name

			else:
				count = ft[row['pastml_name']].sum()

				for p, p_samples in cp.items():
					p_count = ft.loc[p_samples, row['pastml_name']].sum()
					p_pct = p_count / len(p_samples) * 100
					df.loc[i, p] = f"{np.round(p_pct, 1)}"

			df.loc[i, 'Count'] = count

		except Exception as e:
			print("")
			print(e)
			print(row)
			print("")

	df.rename(columns={'raw_feature_name': 'Feature', 'group_name': 'Feature Group', 'feature_type': 'Feature Type'}, inplace=True)
	df = df.sort_values(by="Feature")
	df.replace(
		{'100.00%': '100%',
		'0.0': '-',
		'100.0': '100',
		'0': '-',
		}, regex=False, inplace=True)
	df['Feature Group'] = df['Feature Group'].replace(cg_rev)
	df["Feature Group"] = [f.split("_")[0] + f" ({f.split('_')[1]})" for f in df["Feature Group"]]

	wanted_cols = ["Feature Type", 'Count', 'Sig', 'MLE'] + natsorted(list(cp.keys()))


	print(df)

	df = df.set_index(["Feature Group", "Feature"])
	df = df[wanted_cols]
	df = df.sort_index()
	df.drop(columns=["NT"], inplace=True)
	df.rename(index={i[1]: i[1].replace("000 ", '') for i in df.index if 'Grouped' in i[1]}, inplace=True)
	df.rename(columns={c: c + " %" for c in cp.keys()}, inplace=True)
	
	print(df)
	
	# tex = ''
	# for g, gdf in df.groupby(['Feature Group', 'Feature Type']):
	# 	print(g)
	# 	if g not in dropped:
	# 		tex += gdf.to_latex(
	# 			header=[c.replace("_Percent", "") for c in gdf.columns],
	# 			sparsify=True,
	# 			bold_rows=True,
	# 			)
	# 		tex += "\n\\vspace{16pt}\n"
			
	# (fd / "features_distribution.tex").write_text(tex)

	def feature_info_to_tex(df, fname):
		ldf = df.drop(columns=['Feature Type'])
		ldf.rename(index={f[1]: f[1][0:14] + ".." if len(f[1]) > 16 else f[1] for f in ldf.index}, inplace=True)
		try:
			ldf['MLE'] = [np.round(mle, 3) if not isinstance(mle, str) else mle for mle in ldf['MLE']]
		except:
			breakpoint()
		if 'Sig' in ldf.columns:
			ldf['Sig'] = ldf['Sig'].apply(lambda x: "Y" if x else "N")
		ldf.to_latex(
			tex_dir / "tables" / fname,
			sparsify=True,
			bold_rows=True,
			longtable=True,
		)

	df.to_csv(fd / "all_features_info.csv")
	feature_info_to_tex(df, "all_features_info.tex")
	
	df.drop([d.split("_")[0] + f" ({d.split('_')[1]})" for d in dropped if 'PRJNA' not in d]).to_csv(fd / "all_non_dropped_features_info.csv")
	feature_info_to_tex(df.drop(index=[d.split("_")[0] + f" ({d.split('_')[1]})" for d in dropped if 'PRJNA' not in d]), "all_non_dropped_features_info.tex")

	sig = [j for j in set([i[0] for i in df[df['Sig'] == False].index])]
	df.drop(index=sig, columns=['Sig']).to_csv(fd / "all_sig_features_info.csv")
	feature_info_to_tex(df.drop(index=sig), "all_sig_features_info.tex")

	return df

def cat_display(cat):
	cds = {
		'Amr': 'AMR',
		'Vir': 'Virulence',
		'Stress': 'Stress',
		'Plasmid': 'Plasmid Replicon',
		'nan': 'Background'
	}
	print(cds[str(cat)])
	return cds[str(cat)]

def feature_info_table_old_2():
		# print("Table out:")
	# print(tex_out[0:1000])
	
	# breakpoint()

	# # Dropped features
	# dropped = (fd / "undiverse_dropped_features.txt").read_text().splitlines()

	# # Correlation groups
	# cg = json.loads((fd / "correlation_groups.json").read_text())
	
	# cg_rev = {}
	# for group, members in cg.items():
	# 	for member in members:
	# 		cg_rev[member] = group

	# # Clade designations
	# clades_df = pd.read_csv(dir / "ST131_Typer" / "ST131_Typer_Summary.csv", index_col=0)
	# clades_df.drop('SAMN11319293', inplace=True)

	# # Grouped binary features
	# ftg = pd.read_csv(fd / "combined_ancestral_states_binary_grouped.csv", index_col=0)

	# # Raw binary features
	# ft = pd.read_csv(fd / "combined_ancestral_states_binary.csv", index_col=0)
	


	# # Features info
	# df = pd.read_csv(fd / "features_dict_info.csv")
	# df['Dropped'] = [g in dropped for g in df['group_name']]

	# # Subset to same samples
	# samples = clades_df.index.to_list()
	# clades = clades_df.loc[samples, :]
	# ftg = ftg.loc[samples, :]
	# ft = ft.loc[samples, :]

	# profiles = {pt: clades_df[clades_df['PCR_Profile_Type']==pt].index.to_list() for pt in clades_df['PCR_Profile_Type'].unique()}
	# clades = {clade: clades_df[clades_df['Clade']==clade].index.to_list() for clade in clades_df['Clade'].unique()}
	# # cp = {**clades, **profiles}
	# cp = clades

	# groups = []
	# for g, gdf in df.groupby('group_name'):
	# 	is_dropped = g in dropped

	# 	groups.append({'group_name': g, 'raw_feature_name': f'000 {g.split("_")[0]} Grouped', 'feature_type': g.split("_")[1], 'pastml_name': float("nan"), 'dropped': g in dropped})

	# df = pd.concat([df, pd.DataFrame(groups)], axis=0, ignore_index=True)
	# df['Count'] = 0
	# df['MLE'] = ''
	# df['Sig'] = ''

	# for clade_prof_name in cp.keys():
	# 	df[clade_prof_name] = 0

	# for i, row in df.iterrows():
	# 	try:
	# 		if '000' in row['raw_feature_name']:
	# 			if row['dropped']:
	# 				count = "-"
	# 				df.loc[i, 'MLE'] = "-"
	# 				df.loc[i, 'Sig'] = False

	# 			else:
	# 				group_name = row['group_name']
	# 				if group_name in cg_rev:
	# 					group_name = cg_rev[group_name]

	# 				df.loc[i, 'MLE'] = ci.loc[group_name, 'mle']
	# 				df.loc[i, 'Sig'] = ci.loc[group_name, 'sig']
					
	# 				count = ftg.loc[:, row['group_name']].sum()
					
	# 				for p, p_samples in cp.items():
	# 					p_count = ftg.loc[p_samples, row['group_name']].sum()
	# 					p_pct = p_count / len(p_samples) * 100
	# 					df.loc[i, p] = f"{np.round(p_pct, 1)}"

	# 				df.loc[i, 'group_name'] = group_name

	# 		else:
	# 			count = ft[row['pastml_name']].sum()

	# 			for p, p_samples in cp.items():
	# 				p_count = ft.loc[p_samples, row['pastml_name']].sum()
	# 				p_pct = p_count / len(p_samples) * 100
	# 				df.loc[i, p] = f"{np.round(p_pct, 1)}"

	# 		df.loc[i, 'Count'] = count

	# 	except Exception as e:
	# 		print("")
	# 		print(e)
	# 		print(row)
	# 		print("")

	# df.rename(columns={'raw_feature_name': 'Feature', 'group_name': 'Feature Group', 'feature_type': 'Feature Type'}, inplace=True)
	# df = df.sort_values(by="Feature")
	# df.replace(
	# 	{'100.00%': '100%',
	# 	'0.0': '-',
	# 	'100.0': '100',
	# 	'0': '-',
	# 	}, regex=False, inplace=True)
	# df['Feature Group'] = df['Feature Group'].replace(cg_rev)
	# df["Feature Group"] = [f.split("_")[0] + f" ({f.split('_')[1]})" for f in df["Feature Group"]]

	# wanted_cols = ["Feature Type", 'Count', 'Sig', 'MLE'] + natsorted(list(cp.keys()))


	# print(df)

	# df = df.set_index(["Feature Group", "Feature"])
	# df = df[wanted_cols]
	# df = df.sort_index()
	# df.drop(columns=["NT"], inplace=True)
	# df.rename(index={i[1]: i[1].replace("000 ", '') for i in df.index if 'Grouped' in i[1]}, inplace=True)
	# df.rename(columns={c: c + " %" for c in cp.keys()}, inplace=True)
	
	# print(df)
	
	# # tex = ''
	# # for g, gdf in df.groupby(['Feature Group', 'Feature Type']):
	# # 	print(g)
	# # 	if g not in dropped:
	# # 		tex += gdf.to_latex(
	# # 			header=[c.replace("_Percent", "") for c in gdf.columns],
	# # 			sparsify=True,
	# # 			bold_rows=True,
	# # 			)
	# # 		tex += "\n\\vspace{16pt}\n"
			
	# # (fd / "features_distribution.tex").write_text(tex)

	# def feature_info_to_tex(df, fname):
	# 	ldf = df.drop(columns=['Feature Type'])
	# 	ldf.rename(index={f[1]: f[1][0:14] + ".." if len(f[1]) > 16 else f[1] for f in ldf.index}, inplace=True)
	# 	try:
	# 		ldf['MLE'] = [np.round(mle, 3) if not isinstance(mle, str) else mle for mle in ldf['MLE']]
	# 	except:
	# 		breakpoint()
	# 	if 'Sig' in ldf.columns:
	# 		ldf['Sig'] = ldf['Sig'].apply(lambda x: "Y" if x else "N")
	# 	ldf.to_latex(
	# 		tex_dir / "tables" / fname,
	# 		sparsify=True,
	# 		bold_rows=True,
	# 		longtable=True,
	# 	)

	# df.to_csv(fd / "all_features_info.csv")
	# feature_info_to_tex(df, "all_features_info.tex")
	
	# df.drop([d.split("_")[0] + f" ({d.split('_')[1]})" for d in dropped if 'PRJNA' not in d]).to_csv(fd / "all_non_dropped_features_info.csv")
	# feature_info_to_tex(df.drop(index=[d.split("_")[0] + f" ({d.split('_')[1]})" for d in dropped if 'PRJNA' not in d]), "all_non_dropped_features_info.tex")

	# sig = [j for j in set([i[0] for i in df[df['Sig'] == False].index])]
	# df.drop(index=sig, columns=['Sig']).to_csv(fd / "all_sig_features_info.csv")
	# feature_info_to_tex(df.drop(index=sig), "all_sig_features_info.tex")

	# return df

def format_tex(tex):
	new_tex = []
	for line in tex.splitlines():
		# Get rid of empty row beneath header
		if line.replace("&", "").replace(" ", "").replace("\\", "") == "sig":
			continue

		if '*' in line:
			line = re.sub(r" +", " ", line)
			line = "\\textbf{" + line.replace(" & ", "} & \\textbf{")
			line = line.replace(" \\\\", "} \\\\")

		new_tex.append(line)
	
	tex_out = '\n'.join(new_tex)
	tex_out = '\\small\n' + tex_out + '\\normalsize'

	# # Remove feature header + significance header
	# tex_out = tex_out.replace("Display Name", "{}").replace("sig", "{}")

	# To narrow table:
	tex_out = tex_out.replace(" Replicon", "").replace(" + ", "+")
	return tex_out

def feature_info_table(results_dir):
	fd = dir / "features"

	# Confidence intervals + MLE
	mle = mle_text(dir / "analysis" / "3-interval_constrained-sampling")

	# Name to display name
	dnames = yaml.load((fd / "group_short_name_to_display_manual.yml").read_text())
	for name, nd in dnames.items():
		nd['csv_name'] = name + "_" + nd['category'].upper()

	dname_df = []
	for name, nd in dnames.items():
		name = name.replace("*", "")
		dname_df.append({'Name': name + "_" + nd['category'].upper(), 'Display Name': nd['display_name'], "Feature Type": nd['category']})

	dname_df = pd.DataFrame(dname_df)
	dname_df.set_index('Name', inplace=True)

	df = pd.concat([mle, dname_df], axis=1).dropna(subset='mle')
	df["Feature Type"] = df["Feature Type"].apply(lambda x: cat_display(x))
	
	no_dn = df[df['Display Name'].isna() == True].index
	df.loc[no_dn, 'Display Name'] = [n.split("_")[0] for n in no_dn]
	df['95% CI'] = df.apply(lambda row: f"{row['lower_CI']:.2f}-{row['upper_CI']:.2f}", axis=1)
	df.rename(columns={'mle': 'MLE', 'Display Name': 'Feature'}, inplace=True)
	
	# Display whether significant (95% doesn't overlap 1) or non-one
	# Set as index
	def classify_sig(row):
		if row['sig']:
			return '*'
		elif row['non_one']:
			return '+'
		else:
			return ''

	df['Flag'] = df.apply(lambda x: classify_sig(x), axis=1)
	df.set_index('Flag', inplace=True)

	# Get table of all features
	breakpoint()

	# Table of features that are non-one, not background
	no = df.loc[(df['non_one'] == True) & (df["Feature Type"] != 'Background'), ['Feature', "Feature Type", 'MLE', '95% CI']].sort_values(by="MLE", ascending=False)
	no['sig'] = ['*' if s else '' for s in no['sig']]
	
	
	tex = no.to_latex(
		None,
		sparsify=False,
		bold_rows=False,
		longtable=False,
		float_format="{:.3f}".format,
		caption="Maximum-likelihood estimates of genetic features that were not dropped out of the model. Significant features are bolded with an asterisk.",
		label="table:non-one features"
		)

	tex_out = format_tex(tex)
	(tex_dir / "tables" / "mle_nonone.tex").write_text(tex_out)


if __name__ == "__main__":
	df = feature_info_table(dir / "analysis" / "3-interval_constrained-sampling")
