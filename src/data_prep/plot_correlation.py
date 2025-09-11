from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

def corr_heatmap(corr, out_file):
	dim = int(np.ceil(corr.shape[1] / 4))
	dim = max(8, dim)

	figsize = (dim + 2, dim)

	f, ax = plt.subplots(figsize=figsize)
	cmap = sns.diverging_palette(250, 10, as_cmap=True)
	sns.heatmap(corr,
	            cmap=cmap,
	            vmin=-1,
	            vmax=1,
	            xticklabels=corr.columns,
	            yticklabels=corr.index,
	            annot=True, fmt=".2f",
	            ax=ax,
	            square=True,
	            annot_kws={"size": 6}
	            )
	plt.tight_layout()
	plt.savefig(out_file)
	plt.close("all")

def plot_full_correlation(file, sep=",", filename="", save=True, plot=True, do_return=False):
	if isinstance(file, pd.DataFrame):
		df = file
		file = Path(filename)
	
	else:
		df = pd.read_csv(file, index_col=0, sep=sep)
		if filename:
			file = Path(filename)

	# print(f"Read csv (elapsed: {getElapsedTime(times) / 60})")

	# Find and drop invariant columns
	df = df.loc[:, df.nunique() != 1]

	# Calculate correlations
	# ---------------------
	corr = df.corr()
	# print(f"Calculated correlation (elapsed: {getElapsedTime(times) / 60})")

	# Save correlations
	# -----------------
	if save:
		corr.to_csv(str(file).replace(".csv", "_full-corr.csv"))

	# Plot correlations
	# -----------------
	if plot:
		corr_heatmap(corr, str(file).replace(".csv", "_full-corr.png"))

	if do_return:
		return corr

def plot_correlation_structured(corr_csv, file, sep=",", methods=["average"], filename="", include_count=True, save=True, plot=True):
	if isinstance(file, pd.DataFrame):
		df = file
		file = Path(filename)
	
	else:
		df = pd.read_csv(file, index_col=0, sep=sep)

	features = df.columns.to_list()

	# Find and drop invariant columns
	df = df.loc[:, df.nunique() != 1]

	if corr_csv:
		corr = pd.read_csv(corr_csv, index_col=0)
		corr = corr.loc[df.columns, df.columns]

	else:
		corr = plot_full_correlation(df, sep=",", filename="", save=False, plot=False, do_return=True)

	for method in methods:
		# Do clustering
		# -------------
		lk = linkage(df.T, method=method, metric='correlation', optimal_ordering=True)
		dn = dendrogram(lk, no_plot=True, color_threshold=-np.inf)
		dn_order = dn["leaves"]
		clustered_col_index = [features[i] for i in dn_order]
		# print(f"Made {method} clustermap (elapsed: {getElapsedTime(times) / 60})")

		# Reindex based on clustering
		# ---------------------------
		err = [c for c in clustered_col_index if c not in corr.index.to_list()]
		if err:
			print(err)
			
		corr = corr.loc[clustered_col_index, clustered_col_index]
		if save:
			corr.to_csv(str(file).replace(".csv", f"_{method}-ordered-corr.csv"))
		# print(f"Reindexed corr cols (elapsed: {getElapsedTime(times) / 60})")

		# Re-plot correlation
		# -------------------
		if include_count:
			corr = corr.rename(index=lambda i: f"{i} (n={df[i].sum():.0f})")
		
		if plot:
			corr_heatmap(corr, str(file).replace(".csv", f"_{method}-ordered-corr.png"))
