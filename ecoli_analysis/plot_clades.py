from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import transmission_sim.utils.plot_phylo_standalone as pp
import json
import numpy as np
import re
import seaborn as sns
import matplotlib as mpl
from ecoli_analysis.branch_fitness import color_tree_components, site_effects_to_fitness_tsim, effects_to_fitness_tsim, dir
from ecoli_analysis.feature_matrix import plot_phylo_matrix, get_clade_changepoints
from natsort import natsorted, natsort_keygen

def plot_clade_sampling_over_time(clade_ancestral_file, analysis_dir, out_dir):
	analysis_dir = Path(analysis_dir)
	out_dir = Path(out_dir)

	params = json.loads((analysis_dir / "params.json").read_text())

	tt = pp.loadTree(
		Path(params['tree_file']),
		internal=True,
		abs_time=2023,
	)
	sample_years = {n.traits['name']: np.floor(n.absoluteTime) for n in tt.getExternal()}

	clade_info = pd.read_csv(clade_ancestral_file, index_col=0)
	clade_info = clade_info.loc[clade_info.index.str.contains('SAMN') == True]

	clade_info['Year'] = [sample_years[i] if i in sample_years else np.nan for i in clade_info.index]

	nums = clade_info.groupby(['Clade', 'Year']).size().reset_index()
	nums.columns = ['Clade', 'Year', 'Count']

	counts = pd.DataFrame(index=nums['Clade'].unique(), columns=sorted(nums['Year'].unique()))
	
	for (year, clade), cdf in clade_info.groupby(['Year', 'Clade']):
		counts.loc[clade, year] = len(cdf)

	counts.fillna(0, inplace=True)
	prop = counts.apply(lambda col: col / col.sum(), axis=0)

	fig, ax = plt.subplots(1, 1, figsize=(8, 5))
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
	plt.savefig(out_dir/ "Figure-2_clade_count_time_stacked.png", dpi=300)
	plt.close('all')

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
	
	col_order = ['total', 'genetic', 'background', 'random', 'AMR', 'virulence', 'stress', 'plasmid']
	df = pd.DataFrame(clade_fitness).T
	df[col_order].to_csv(analysis_dir / "clade_fitness_breakdown.csv")

def relative_clade_fitness(analysis_dir):
	df = pd.read_csv(analysis_dir / "clade_fitness_breakdown.csv", index_col=0)
	(df / df.loc['B1']).to_csv(analysis_dir / "clade_fitness_breakdown_relToB1.csv")

if __name__ == "__main__":
	analysis_name = "3-interval_constrained-sampling"

	data_dir = Path() / "data"
	analysis_dir = data_dir / "analysis" / analysis_name
	figures_dir = data_dir / "figures"

	clade_summary_file = data_dir / "ST131_Typer_Summary.csv"
	clade_ancestral_file = data_dir / "combined_ancestral_states.csv"
	clade_ancestral_tree_file = data_dir / "named.tree_lsd.date.noref.nexus"

	plot_clade_sampling_over_time(clade_ancestral_file, analysis_dir)


