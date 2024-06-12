from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import transmission_sim.utils.plot_phylo_standalone as pp
import json
import numpy as np
import re
import seaborn as sns
import matplotlib as mpl
from transmission_sim.ecoli.branch_fitness import color_tree_components, site_effects_to_fitness_tsim, effects_to_fitness_tsim, dir
from transmission_sim.ecoli.feature_matrix import plot_phylo_matrix, get_clade_changepoints
from transmission_sim.ecoli.analyze import writeup_dir, ncbi_dir
from natsort import natsorted, natsort_keygen

def plot_clade_membership(clade_summary_file, clade_ancestral_file, data, params, out_file_clade_phylo, out_file_changepoints):
	# Load tree
	tt = pp.loadTree(
		Path(params['tree_file']),
		internal=True,
		abs_time=2023,
	)

	samples = [(n.traits['name'], n.y) for n in tt.Objects if n.branchType=="leaf"]

	data, changepoints = get_clade_changepoints(tt, data, clade_ancestral_file, out_file_changepoints)

	# Read in file with clade attributes of tips (samples)
	# Some columns of this will be the matrix
	clade_info = pd.read_csv(clade_summary_file, index_col=0)
	matrix = clade_info.loc[:, [
		'trpA72', 'rfb_O16', 'rfb_O25b', 
		'fliC_H4', 'fliC_H5', 'fimH22', 'fimH27', 'fimH30', 'fimH35', 'fimH41', 
		'plsB', 'nupC', 'kefC', 'rmuC', 'prophage', 'sbmA', 'ybbW', 'parC_E84V'
		]].fillna("NF")

	# # Calculate clade fitness
	# clade_fitness = {}
	# for clade, clade_df in data.groupby('clade'):
	# 	average_fitness = clade_df['total_fitness'].mean()
	# 	data.loc[clade_df.index, 'clade_fitness'] = average_fitness
	# 	clade_fitness[clade] = average_fitness
	# 	print(f"Average fitness of {clade}: {average_fitness}")

	# # ------------------------------------------------------------
	# # Plot phylogeny and matrix of genes used to determine
	# # Colored by clade designation
	# # ------------------------------------------------------------
	# # Color the phylogeny by clade, and matrix by presence/absence of alleles
	# colors, tree_c_func = pp.categoricalFunc(trait_dict=data['clade'].to_dict(), trait="name", color_list=sns.color_palette("Set2"), legend=True, null_color="red")
	# matrix_c_func = lambda cell: "white" if cell == "NF" else "skyblue"
	# plt, fig, ax_tree, ax_matrix = plot_phylo_matrix(tt, matrix, tree_c_func, matrix_c_func, None, None, changepoints, None, cbar=False, save=False)

	# # Create and add legend to tree
	# legend_dict = {f"{clade} (avg. fitness = {clade_fitness[clade]:.3f})": color for clade, color in colors.items()}
	# ax_tree = pp.add_legend(legend_dict, ax_tree, lloc="lower left")

	# # Annotate tree with clade divergence times
	# target_func = lambda k: k.traits['name'] in changepoints.index.to_list() ## which branches will be annotated
	# text_func = lambda k: changepoints.loc[k.traits['name'], "str"] ## what text is plotted
	# text_x_attr = lambda k: k.absoluteTime - 2 ## where x coordinate for text is
	# text_y_attr = lambda k: k.y - 5 ## where x coordinate for text is
	# kwargs = {'va':'top','ha':'right','size': 20} ## kwargs for text
	# tt.addText(ax_tree, x_attr=text_x_attr, y_attr=text_y_attr, target=target_func, text=text_func, **kwargs)
	# tt.plotPoints(ax_tree, x_attr=lambda k: k.absoluteTime, y_attr=lambda k: k.y, target=target_func, size=36, colour="black")
	# plt.savefig(out_file_clade_phylo, dpi=300)
	# plt.close("all")

	# ------------------------------------------------------------
	# Plot phylogeny and matrix of genes used to determine
	# Colored by profile type
	# ------------------------------------------------------------

	# Load unsegmented tree
	tt = pp.loadTree(
		ncbi_dir / "final" / "ml_outlier_pruned_mpr-decr.nwk",
		internal=False,
	)

	tt = tt.reduceTree(keep=[n for n in tt.getExternal() if n.traits['name'] in matrix.index.to_list()])

	# Color the phylogeny by clade, and matrix by presence/absence of alleles
	clade_info = clade_info.sort_values(
   		by='PCR_Profile_Type',
    	key=natsort_keygen()
	)

	colors, tree_c_func = pp.categoricalFunc(trait_dict=clade_info['PCR_Profile_Type'].to_dict(), trait="name", color_list=sns.color_palette("Paired"), legend=True, null_color="#dddddd")
	matrix_c_func = lambda cell: "white" if cell == "NF" else "skyblue"
	plt, fig, ax_tree, ax_matrix = plot_phylo_matrix(tt, matrix, tree_c_func, matrix_c_func, None, None, changepoints, None, zoom=[-.01, 0.0], cbar=False, save=False)

	ax_tree = pp.add_legend({c: colors[c] for c in natsorted(colors.keys())}, ax_tree, lloc="lower left")

	# Annotate tree with clade divergence times
	plt.savefig(Path(str(out_file_clade_phylo).replace(".png", "_profile.png")), dpi=300)
	plt.close("all")

def plot_clade_sampling_over_time(clade_summary_file, data, params):
	tt = pp.loadTree(
		Path(params['tree_file']),
		internal=True,
		abs_time=2023,
	)

	clade_info = pd.read_csv(clade_summary_file, index_col=0)
	bioproject = pd.read_csv(ncbi_dir / "meta" / "biosample_bioproj_all.csv", index_col=0)
	bioproject = bioproject[~bioproject.index.duplicated(keep="first")]

	years = pd.Series({n.traits['name']: np.floor(n.absoluteTime) for n in tt.getExternal()}, name="Year")
	clade_info = pd.concat([clade_info, years, bioproject], axis=1)
	clade_info.dropna(subset="Clade", inplace=True)

	# nums = clade_info.groupby(['Clade', 'BioProject', 'Year']).size().reset_index()
	# nums.columns = ['Clade', 'Bioproject', 'Year', 'Count']

	nums = clade_info.groupby(['Clade', 'Year']).size().reset_index()
	nums.columns = ['Clade', 'Year', 'Count']

	# sns.set_style('whitegrid')
	# sns.relplot(kind="line", data=nums, x="Year", y="Count", hue="Clade", col="Bioproject", col_wrap=5, marker='o')
	# plt.tight_layout()
	# plt.savefig(writeup_dir / "figures" / "clade_count_time.png", dpi=300)

	# sns.set_style('whitegrid')
	# sns.lineplot(data=nums, x="Year", y="Count", hue="Clade", marker='o')
	# plt.tight_layout()
	# plt.savefig(writeup_dir / "figures" / "clade_count_time.png", dpi=300)
	# plt.close('all')

	counts = pd.DataFrame(index=nums['Clade'].unique(), columns=sorted(nums['Year'].unique()))
	
	for (year, clade), cdf in clade_info.groupby(['Year', 'Clade']):
		counts.loc[clade, year] = len(cdf)

	counts.fillna(0, inplace=True)
	prop = counts.apply(lambda col: col / col.sum(), axis=0)

	breakpoint()

	fig, ax = plt.subplots(1, 1, figsize=(6, 5))
	ax2 = ax.twinx()

	# Stackplot of clade proportions
	ax.stackplot(prop.columns, prop.values, labels=prop.index, colors=sns.color_palette("Paired"))
	
	# Line plot of total number sampled
	ax2.plot(counts.columns, counts.sum(axis=0), color="black")
	ax2.scatter(counts.columns, counts.sum(axis=0), color="black", s=4)
	
	ax.set_xlabel("Year")
	ax.set_ylabel("Proportion")
	ax2.set_ylabel("Sampled Count")

	ax.legend(loc='upper left')
	plt.tight_layout()
	plt.savefig(writeup_dir / "figures" / "clade_count_time_stacked.png", dpi=300)
	plt.close('all')

	# counts = pd.DataFrame(index=nums['Clade'].unique(), columns=sorted(nums['Year'].unique()))
	
	# clade_info = clade_info[clade_info['BioProject'] != 'PRJNA809394']
	# for (year, clade), cdf in clade_info.groupby(['Year', 'Clade']):
	# 	counts.loc[clade, year] = len(cdf)

	# counts.fillna(0, inplace=True)
	# prop = counts.apply(lambda col: col / col.sum(), axis=0)

	# plt.stackplot(prop.columns, prop.values, labels=prop.index, colors=sns.color_palette("Paired"))
	# plt.xlabel("Year")
	# plt.ylabel("Proportion")
	# plt.legend(loc='upper left')
	# plt.tight_layout()
	# plt.savefig(writeup_dir / "figures" / "clade_count_time_stacked_no-PRJNA809394.png", dpi=300)

def get_clade_fitness(data, clade_ancestral_file, analysis_dir):
	anc_df = pd.read_csv(clade_ancestral_file, index_col=0).squeeze()
	data['clade'] = [anc_df[n.split("_")[0]] for n in data.index]

	data['background_fitness'] = data.time_fitness * data.PRJNA
	data['genetic_fitness'] = data.AMR * data.VIR * data.PLASMID * data.STRESS

	# Calculate clade fitness
	clade_fitness = {}
	for clade, clade_df in data.groupby('clade'):
		clade_fitness[clade] = {}
		clade_fitness[clade]['background'] = clade_df['background_fitness'].mean()
		clade_fitness[clade]['genetic'] = clade_df['genetic_fitness'].mean()
		clade_fitness[clade]['random'] = clade_df['total_fitness'].mean()
		clade_fitness[clade]['total'] = clade_df['total_fitness'].mean()
		clade_fitness[clade]['AMR'] = clade_df['AMR'].mean()
		clade_fitness[clade]['virulence'] = clade_df['VIR'].mean()
		clade_fitness[clade]['plasmid'] = clade_df['PLASMID'].mean()
		clade_fitness[clade]['stress'] = clade_df['STRESS'].mean()
	
	longdict = []
	for clade, cd in clade_fitness.items():
		print(f"Clade {clade}:")
		for k, v in cd.items():
			print(f"\t{k}: {v:.3f}")
			longdict.append({'Clade': clade, 'Component': k, 'Fitness': v})

	df = pd.DataFrame(clade_fitness).T
	print(df)

	df.to_csv(analysis_dir / "clade_fitness_breakdown.csv")

	ld = pd.DataFrame(longdict)
	ld['Fitness'] = np.log(ld['Fitness'])

	sns.set_style('whitegrid')
	categories = ['random', 'background', 'genetic']
	sns.histplot(
		data=ld[ld['Component'].isin(categories)],
		x="Clade", hue="Component", weights="Fitness",
		multiple="stack", hue_order=categories,
		palette = sns.color_palette("Set2"), alpha=1,
   		shrink=.8,
	)
	plt.savefig(writeup_dir / "figures" / "clade_fitness_breakdown.png", dpi=300)
	plt.close("all")

	sns.set_style('whitegrid')
	categories = ['random', 'background', 'stress', 'virulence', 'plasmid', 'AMR']
	sns.histplot(
		data=ld[ld['Component'].isin(categories)],
		x="Clade", hue="Component", weights="Fitness",
		multiple="stack", hue_order=categories,
		palette = sns.color_palette("Set2"), alpha=1,
   		shrink=.8
	)
	plt.savefig(writeup_dir / "figures" / "clade_fitness_breakdown_by_cat.png", dpi=300)

	print(ld[ld['Fitness'] < 0])

def relative_clade_fitness(analysis_dir):
	df = pd.read_csv(analysis_dir / "clade_fitness_breakdown.csv", index_col=0)
	(df / df.loc['B1']).to_csv(analysis_dir / "clade_fitness_breakdown_relToB1.csv")

def clade_a_info(clade_summary_file, data, params):
	tt = pp.loadTree(
		Path(params['tree_file']),
		internal=True,
		abs_time=2023,
	)

	clade_info = pd.read_csv(clade_summary_file, index_col=0)
	bioproject = pd.read_csv(ncbi_dir / "meta" / "biosample_bioproj_all.csv", index_col=0)
	bioproject = bioproject[~bioproject.index.duplicated(keep="first")]

	years = pd.Series({n.traits['name']: np.floor(n.absoluteTime) for n in tt.getExternal()}, name="Year")
	clade_info = pd.concat([clade_info, years, bioproject], axis=1)
	clade_info.dropna(subset="Clade", inplace=True)

	a = clade_info[clade_info['Clade']=='A']
	
	# Drop non-unique
	au = a.loc[:, ~a.eq(a.iloc[0, :], axis=1).all(0)]
	unique_prof = au.columns.to_list()
	
	features = pd.read_csv('/home/lenora/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/combined_ancestral_states_binary_grouped_diverse_uncorr.csv', index_col=0).loc[a.index, :]
	
	# Drop non-unique
	featuresu = features.loc[:, ~features.eq(features.iloc[0, :], axis=1).all(0)]
	featuresu.drop(columns=[c for c in featuresu.columns if 'PRJNA' in c], inplace=True)
	unique_feat = featuresu.columns.to_list()

	cm = pd.read_csv(ncbi_dir / "combined_meta.csv", index_col=0).loc[a.index, :]

	data = data.loc[a.index, ['event_time', 'site_fitness',
       'birth_interval', 'time_fitness', 'random_fitness',
       'total_model_fitness', 'total_fitness']]

	df = pd.concat([data, au, featuresu], axis=1)
	df.dropna(how="all", inplace=True, axis=1)

	df2 = pd.read_csv(ncbi_dir / "final" / "NCBI-Assembly_ST131_Typer_Summary.csv", index_col=0)
	df2
	breakpoint()
	# cc = pd.concat([df['Clade'], df2['Clade'].rename(columns={'Clade': 'Assembly_Clade'})], axis=1)



	df_phan = df.drop(columns=[c for c in df.columns if 'fim' in c] + ['Description', 'event_time', 'birth_interval', 'time_fitness', 'total_fitness'])
	df_phan.to_csv(ncbi_dir / "final" / "clade_A_phandango.csv")

if __name__ == "__main__":
	analysis_dir = dir / "analysis" / "3-interval_constrained-sampling"
	
	data, params = effects_to_fitness_tsim(analysis_dir, random_name = "Est-Random_Fixed-BetaSite")
	data2, params = site_effects_to_fitness_tsim(analysis_dir)

	data = pd.concat([data, data2], axis=1)

	clade_summary_file = dir / "ST131_Typer/ST131_Typer_Summary.csv"
	clade_ancestral_file = dir / "ST131_Typer/combined_ancestral_states.csv"
	clade_ancestral_tree_file = dir / "ST131_Typer/pastml/named.tree_lsd.date.noref.nexus"

	out_file_clade_phylo = writeup_dir / "figures" / "Phylo_Clade_Info.png"
	out_file_changepoints = analysis_dir / "Clade_Changepoints.csv"
	# plot_clade_membership(clade_summary_file, clade_ancestral_file, data, params, out_file_clade_phylo, out_file_changepoints)

	# plot_clade_sampling_over_time(clade_summary_file, data, params)
	# get_clade_fitness(data, clade_ancestral_file, analysis_dir)
	relative_clade_fitness(analysis_dir)

	# clade_a_info(clade_summary_file, data, params)


