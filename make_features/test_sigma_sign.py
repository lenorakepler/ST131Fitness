import pandas as pd
import numpy as np
# from ecoli_analysis.results_obj import ResultsObj
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.DataFrame(
	dict(
		parent_fitness=[1, .2, 3.4, 1, 1, 5, .51],
		time_delta=[1, 1, 1, 1, 1, 1, 1],
		)
	)
df['time_delta'] = df['time_delta'] * 3

df["child_fitness_3_more"] = df["parent_fitness"] + 3
df["child_fitness_.5_more"] = df["parent_fitness"] + .5
df["child_fitness_.5_less"] = df["parent_fitness"] - .5

epsilon = 0.0000005
sigma = .5

penalties = []
for sigma in [0, 0.01, 1, 10]:
	sigma_dict = {"sigma": sigma}
	for cf_key in [k for k in df.columns if 'child_fitness' in k]:
		df["fit_shifts"] = df[cf_key] - df["parent_fitness"]

		denom = sigma * df["time_delta"]
		num = -0.5 * np.square(df["fit_shifts"])
		prob = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)
		penalty = prob.sum() * -1

		sigma_dict[cf_key] = penalty
	
	penalties.append(sigma_dict)

print(pd.DataFrame(penalties))

# ===========================================================
# analysis_dir = Path("data_new/analysis/three_sampling_intervals")

# RO = ResultsObj(analysis_dir)

# for result_key, results_dict in RO.results_dict.items():
# 	if 'brownian_motion' in result_key and 'test' not in result_key:	
# 		sigma_dict = []
# 		sigma_dict_alt = []

# 		cf_type = 'time' if 'sigma-opt' in result_key else 'random'

# 		for param_key, param_dict in results_dict['results_list'].items():
# 			fold_dict = {**param_dict['h_combo'], 'lr': param_dict['lr'], 'n_epochs': param_dict['n_epochs']}

# 			edf = pd.DataFrame([estimates['brownian_motion'] for estimates in param_dict['fold_estimates']]).mean()
# 			sigma_dict_alt.append({**fold_dict, **edf.to_dict()})
			
# 			for i, e in edf.items():
# 				sigma_dict.append({**fold_dict, 'estimate': e, 'position': i})

# 		df = pd.DataFrame(sigma_dict_alt)
# 		mean_site = df[[c for c in df.columns if isinstance(c, int)]].mean(axis=0)
# 		mean_site_order = {i: o for o, i in enumerate(mean_site.sort_values().index)}

# 		rand = mean_site.sample(n=200, random_state=8).index.to_list()

# 		df2 = pd.DataFrame(sigma_dict)
# 		df2['mean_estimate'] = df2["position"].apply(lambda k: mean_site[k])
# 		df2['index_ordered'] = df2["position"].apply(lambda k: mean_site_order[k])
# 		df2 = df2[df2['mean_estimate'] != 1]

# 		fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# 		sns.lineplot(data=df2, x="mean_estimate", y="estimate", hue="sigma", style="lr", palette=sns.color_palette("flare", as_cmap=True), ax=ax)
# 		plt.tight_layout()
# 		plt.savefig(analysis_dir / f"sigma_eff_{result_key}.png", dpi=300)
# 		plt.close("all")

# 		fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# 		sns.lineplot(data=df2, x="index_ordered", y="estimate", hue="sigma", style="lr", palette=sns.color_palette("flare", as_cmap=True), ax=ax)
# 		plt.tight_layout()
# 		plt.savefig(analysis_dir / f"sigma_eff_by-ordered-index_{result_key}.png", dpi=300)
# 		plt.close("all")

# 		df3 = df2[df2["position"].isin(rand) == True]

# 		fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# 		sns.lineplot(data=df3, x="mean_estimate", y="estimate", hue="sigma", style="lr", palette=sns.color_palette("flare", as_cmap=True), ax=ax)
# 		plt.tight_layout()
# 		plt.savefig(analysis_dir / f"sigma_eff_subset_{result_key}.png", dpi=300)
# 		plt.close("all")

# 		fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# 		sns.lineplot(data=df3, x="index_ordered", y="estimate", hue="sigma", style="lr", palette=sns.color_palette("flare", as_cmap=True), ax=ax)
# 		plt.tight_layout()
# 		plt.savefig(analysis_dir / f"sigma_eff_by-ordered-index_subset_{result_key}.png", dpi=300)
# 		plt.close("all")
