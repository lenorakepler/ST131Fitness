import copy
import tensorflow as tf
from pathlib import Path
import itertools
import copy
import pandas as pd
import numpy as np
import yaml
import json
import tensorflow as tf
from transmission_sim.analysis.optimizer import Optimizer
from transmission_sim.analysis.param_model import Site, ComponentB0, ParamModel, ComponentSite
from transmission_sim.ecoli.results_obj import Site2
import transmission_sim.utils.plot_phylo_standalone as pp
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def plot_effect_profile(site, profile_results, mle_eff, out_dir):
	profile_mle = list(profile_results[site].keys())[np.argmin(list(profile_results[site].values()))]
	profile_mle_loss = np.min(list(profile_results[site].values()))

	plt.plot(list(profile_results[site].keys()), list(profile_results[site].values()))
	plt.axvline(mle_eff, color='red', label=f"Estm MLE ({mle_eff:.3f})")
	plt.axvline(profile_mle, color='red', linestyle="--", label=f"Prof ({profile_mle:.3f}) - {profile_mle_loss:.2f}")
	plt.xlabel(site)
	plt.ylabel('likelihood')
	plt.title(f'Likelihood surface around MLE of {site} effect')
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_dir / f'{site}.png')
	plt.close("all")

def make_profile(train, estimates, features_file, bdm_params, h_combo, birth_rate_idx, out_dir):
	profile_dir = out_dir / "profiles"
	profile_dir.mkdir(exist_ok=True, parents=True)

	features = pd.read_csv(features_file, index_col=0)

	if isinstance(estimates, dict):
		time_estimates = estimates.get('b0:0', [])
		site_estimates = estimates.get('site:0', [])

	else:
		if bdm_params['b0'][0]:
			time_estimates = estimates[0]
			if bdm_params['site'][0]:
				site_estimates = estimates[1]
		else:
			if bdm_params['site'][0]:
				site_estimates = estimates[0]

	time_names = [c for c in features.columns if 'Interval' in c] if time_estimates else []
	site_names = [c for c in features.columns if 'Interval' not in c] if site_estimates else []

	profile_results = {}

	# -----------------------------------------------------
	# For each time estimate
	# -----------------------------------------------------
	for i, site in enumerate(time_names):
		print(f"\nTime feature {site} ({i})\n=========================================")

		profile_results[site] = {}

		# -----------------------------------------------------
		# Get loss at each value
		# -----------------------------------------------------
		mle_eff = time_estimates[i]
		for n in np.linspace(mle_eff - .75, mle_eff + .75, 200, endpoint=False):
			effs = np.array(time_estimates).copy()
			mle_eff = effs[i].copy()
			effs[i] = n

			fit_model_kwargs = dict(
				**bdm_params,
				birth_rate_idx=birth_rate_idx,
				data=train,
				iterative_pE=True,
				loss_kwargs=h_combo,
			)
			fit_model = Site2(**fit_model_kwargs)
			phylo_loss = fit_model.phylo_loss(**fit_model.loss_kwargs)

			fit_model.b0 = tf.Variable(effs, dtype=tf.dtypes.float64)

			if bdm_params['site'][0]:
				fit_model.site = tf.Variable(site_estimates.copy(), dtype=tf.dtypes.float64)

			c = fit_model.call()

			weights = tf.concat([tf.reshape(v, [-1]) for v in fit_model.trainable_variables], axis=-1)
			loss = phylo_loss.call(c, weights=weights).numpy()

			profile_results[site][n] = loss
			print(f"\t{n:.2f}: {loss:.2f}")

		plot_effect_profile(site, profile_results, mle_eff, profile_dir)

	# -----------------------------------------------------
	# For each site estimate
	# -----------------------------------------------------
	for i, site in enumerate(site_names):
		print(f"\nFeature {site} ({i})\n=========================================")

		profile_results[site] = {}

		# -----------------------------------------------------
		# Get loss at each value
		# -----------------------------------------------------
		mle_eff = site_estimates[0][i]
		for n in np.linspace(mle_eff - .75, mle_eff + .75, 200, endpoint=False):
			effs = np.array(site_estimates).copy()
			effs[0, i] = n

			fit_model_kwargs = dict(
				**bdm_params,
				birth_rate_idx=birth_rate_idx,
				data=train,
				iterative_pE=True,
				loss_kwargs=h_combo,
			)
			fit_model = Site2(**fit_model_kwargs)
			phylo_loss = fit_model.phylo_loss(**fit_model.loss_kwargs)

			fit_model.site = tf.Variable(effs, dtype=tf.dtypes.float64)

			if bdm_params['b0'][0]:
				fit_model.b0 = tf.Variable(time_estimates, dtype=tf.dtypes.float64)
				
			c = fit_model.call()

			weights = tf.concat([tf.reshape(v, [-1]) for v in fit_model.trainable_variables], axis=-1)
			loss = phylo_loss.call(c.__dict__, weights=weights).numpy()

			profile_results[site][n] = loss
			print(f"\t{n:.2f}: {loss:.2f}")

		plot_effect_profile(site, profile_results, mle_eff, profile_dir)

	profile_df = pd.DataFrame.from_dict(profile_results)
	profile_df.to_csv(out_dir / f"likelihood_profile.csv")

	# breakpoint()

	# profile_mles = pd.Series(profile_mles, name="estimated").to_frame().T
	# profile_mles = profile_mles[res.columns.to_list()]
	# profile_mles.to_csv(out_dir / f"profile_mles.csv")

def get_CIs(analysis_dir):
	estimates = pd.read_csv(analysis_dir / "estimates.csv", index_col=0)
	estimates = estimates.set_index("feature").squeeze()
	like_profile = pd.read_csv(analysis_dir / f"likelihood_profile.csv", index_col=0)

	ci_dict = {}

	for f, c in enumerate(like_profile.columns):
		# Get MLE estimate
		mle = estimates[c]

		# Get values where this feature was estimated and
		# their associated losses
		feature_profile = like_profile.loc[like_profile[c].isna()==False, c]
		L = feature_profile.values * -1

		# Get value with lowest loss, calculate distance between
		# loss at MLE and loss at other values
		mle_index = np.argmax(L)
		deltaL = L - L[mle_index]

		# Get 95% CI upper and lower indices
		lower_index = np.argmin(np.abs(deltaL[:mle_index] + 1.92))
		upper_index = mle_index + np.argmin(np.abs(deltaL[mle_index:] + 1.92))

		# breakpoint()

		# plt.plot(feature_profile.index, L, color="black", label="raw")
		# plt.plot(feature_profile.index, deltaL, color="blue", label="delta")
		# plt.plot(feature_profile.index[:mle_index], np.abs(deltaL[:mle_index] + 1.92), label="lower")
		# plt.plot(feature_profile.index[mle_index:], np.abs(deltaL[mle_index:] + 1.92), label="upper")
		# plt.axvline(lower_CI, color="red", label="lower CI")
		# plt.axvline(upper_CI, color="red", label="upper CI")
		# plt.axvline(-1.92, color="green", label="1.92")
		# plt.xlim(.9, 1.2)
		# plt.ylim(-10, 10)
		# plt.savefig(analysis_dir / "citest.png", dpi=300)
		# plt.close("all")

		# target_loss = feature_profile.values.min()
		# get distance from target loss


		plt.plot(feature_profile, color="blue")
		plt.scatter(feature_profile.index, feature_profile.values, color="black")
		plt.axhline(feature_profile.values.min() + 1.92, color="red")
		plt.xlim(.9, 1.2)
		plt.ylim(feature_profile.values.min() - 10, feature_profile.values.min() + 10)
		plt.savefig(analysis_dir / "citest2.png", dpi=300)




		lower_CI = feature_profile.index[lower_index]
		upper_CI = feature_profile.index[upper_index]

		lower_CI_delta = np.abs(lower_CI - mle)
		upper_CI_delta = np.abs(upper_CI - mle)

		ci_dict[c] = dict(
			initial_mle=mle,
			mle=feature_profile.index[mle_index],
			upper_CI=upper_CI,
			lower_CI=lower_CI,
			lower_CI_delta=lower_CI_delta,
			upper_CI_delta=upper_CI_delta
		)

	df = pd.DataFrame(ci_dict).T
	df.to_csv(analysis_dir / f"profile_CIs.csv")

def cat_display(cat):
	cds = {
		'Amr': 'AMR',
		'Vir': 'Virulence',
		'Stress': 'Stress',
		'Plasmid': 'Plasmid Replicon',
		'Meta': 'Background',
		'nan': 'Background'
	}
	return cds[str(cat)]

def box_plot(analysis_dir, out_dir):
	est_df = pd.read_csv(analysis_dir / f"profile_CIs.csv", index_col=0)
	est_df.drop(columns=["initial_mle"], inplace=True)

	df = est_df[['lower_CI', 'upper_CI', 'mle']]

	# Doesn't matter what these values are, one just needs to be
	# slightly smaller than the lower CI, the other slightly larger
	# than the upper CI 
	# so that with a list of N = 5 (a_lower, lower, mle, upper, a_upper),
	# the index of the first quartile is h = (5 - 1) * 1/4 + 1 = 2 (1 with 0-based indexing)
	# and the second element in the list is the lower bound
	# https://en.wikipedia.org/wiki/Quantile -- numpy uses linear interpolation
	df['artificial_lower_CI'] = df['lower_CI'] - .01
	df['artificial_upper_CI'] = df['upper_CI'] + .01

	# df = df.T.reset_index()
	# df.drop(columns=["index"], inplace=True)
	df = df.T

	# Change to display name
	# ---------------------------
	# Load display names
	dnames = yaml.load((analysis_dir.parent.parent / "features" / "group_short_name_to_display_manual.yml").read_text())
	for name, nd in dnames.items():
		nd['csv_name'] = name + "_" + nd['category'].upper()

	# Make df to convert names, hold category
	display_df = []
	for name, nd in dnames.items():
		name = name.replace("*", "")
		display_df.append({'Name': name + "_" + nd['category'].upper(), 'Display Name': nd['display_name'], "Feature Type": nd['category']})

	display_df = pd.DataFrame(display_df)
	display_df["Display Type"] = display_df["Feature Type"].apply(lambda x: cat_display(x))
	display_df["Feature Type"] = display_df["Feature Type"].str.upper()
	display_df.set_index('Name', inplace=True)

	# Convert columns to correct display name
	df.columns = [display_df.loc[n, 'Display Name'] for n in df.columns]
	est_df.index = [display_df.loc[n, 'Display Name'] for n in est_df.index]

	# Then set display_df index to display name
	display_df.set_index('Display Name', inplace=True)

	non_1_df = df.loc[:, df.loc['mle', :].between(.999, 1.001) == False]

	order = est_df.sort_values(by='mle', ascending=True).index.to_list()

	def is_significant(c):
		c_min = est_df.loc[c, 'lower_CI']
		c_max = est_df.loc[c, 'upper_CI']
		if (c_min > 1) and (c_max > 1):
			return True
		elif (c_min < 1) and (c_max < 1):
			return True
		else:
			return False

	def get_color(c, display_df):
		cat = display_df.loc[c, "Display Type"]
		shades = sns.light_palette(colors[cat], n_colors=12, as_cmap=False)
		if is_significant(c):
			return shades[-1]
		else:
			return shades[2]

	# BOX PLOT OF ALL EFFECTS
	# -------------------------
	# significance_palette = {c: "#4ea5ef" if is_significant(c) else "#ecf8fe" for c in df.columns}
	color_list = sns.color_palette("Set2")
	colors = {cat: color_list[i] for i, cat in enumerate(['AMR', 'Plasmid Replicon', 'Virulence', 'Stress', 'Background'])}
	category_palette = {c: get_color(c, display_df) for c in df.columns}

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(figsize=(12, 20))
	sns.boxplot(
		data=df,
		palette=category_palette,
		order=order,
		whis=0.0, showfliers=False,
		orient="h",
	)
	axs.axvline(1, color='k', alpha=0.4)
	axs.set_xlabel('Transmission Fitness Effect', fontsize=16, labelpad=15)
	axs.set_ylabel('Feature', fontsize=16, labelpad=25)
	recs = [mpl.patches.Rectangle((0,0),1,1, fc=c) for c in colors.values()]
	axs.legend(recs, colors.keys(), loc='upper right', fontsize=20)
	axs.tick_params(axis='both', labelsize=14)

	fig.tight_layout()
	plt.savefig(out_dir / f"profile_CIs_boxplot.png", dpi=300)
	plt.show()

	category_palette = {c: get_color(c, display_df) for c in non_1_df.columns}

	sns.set_style("whitegrid")
	fig, axs = plt.subplots(figsize=(12, 20))
	sns.boxplot(
		data=non_1_df,
		palette=category_palette,
		order=[o for o in order if o in non_1_df.columns],
		whis=0.0, showfliers=False,
		orient="h",
	)
	axs.axvline(1, color='k', alpha=0.4)
	axs.set_xlabel('Transmission Fitness Effect', fontsize=16, labelpad=15)
	axs.set_ylabel('Feature', fontsize=16, labelpad=25)
	recs = [mpl.patches.Rectangle((0,0),1,1, fc=c) for c in colors.values()]
	axs.legend(recs, colors.keys(), loc='upper right', fontsize=20)
	axs.tick_params(axis='both', labelsize=14)

	fig.tight_layout()
	plt.savefig(out_dir / f"profile_CIs_boxplot_non1.png", dpi=300)
	plt.show()

	color_list = sns.color_palette("Set2")
	colors = {cat: color_list[i] for i, cat in enumerate(['AMR', 'Plasmid Replicon', 'Virulence', 'Stress'])}

	group_columns = [c for c in df.columns if display_df.loc[c, 'Display Type'] != 'Background']
	category_palette = {c: get_color(c, display_df) for c in group_columns}

	fig, axs = plt.subplots(figsize=(12, 1 + .5 * len(group_columns)))
	sns.boxplot(
		data=df[group_columns],
		palette=category_palette,
		order=[o for o in order if o in group_columns],
		whis=0.0, showfliers=False,
		orient="h",
	)
	axs.axvline(1, color='k', alpha=0.4)
	axs.set_xlabel('Transmission Fitness Effect', fontsize=16, labelpad=15)
	axs.set_ylabel('Feature', fontsize=16, labelpad=25)
	recs = [mpl.patches.Rectangle((0,0),1,1, fc=c) for c in colors.values()]
	axs.legend(recs, colors.keys(), loc='upper right', fontsize=20)
	axs.tick_params(axis='both', labelsize=14)

	plt.tight_layout()
	plt.savefig(out_dir / f"profile_CIs_boxplot_non-Background.png", dpi=300)
	plt.close("all")

	group_columns = [c for c in non_1_df.columns if display_df.loc[c, 'Display Type'] != 'Background']

	fig, axs = plt.subplots(figsize=(12, 1 + .5 * len(group_columns)))
	sns.boxplot(
		data=non_1_df[group_columns],
		palette=category_palette,
		order=[o for o in order if o in group_columns],
		whis=0.0, showfliers=False,
		orient="h",
	)
	axs.axvline(1, color='k', alpha=0.4)
	axs.set_xlabel('Transmission Fitness Effect', fontsize=16, labelpad=15)
	recs = [mpl.patches.Rectangle((0,0),1,1, fc=c) for c in colors.values()]
	axs.legend(recs, colors.keys(), loc='upper right', fontsize=20)
	axs.set_ylabel('Feature', fontsize=16, labelpad=25)
	axs.tick_params(axis='both', labelsize=14)

	plt.tight_layout()
	plt.savefig(out_dir / f"profile_CIs_boxplot_non-background_non1.png", dpi=300)
	plt.close("all")

	# BOX PLOTS OF NON-META FEATURES
	# only significant
	# Colored by feature type
	# -------------------------
	group_columns = [c for c in non_1_df.columns if display_df.loc[c, 'Display Type'] != 'Background' and is_significant(c)]
	fig, axs = plt.subplots(figsize=(12, 1 + .5 * len(group_columns)))
	box = sns.boxplot(
		data=non_1_df[group_columns],
		palette=category_palette,
		order=[o for o in order if o in group_columns],
		whis=0.0, showfliers=False,
		orient="h",
	)
	axs.axvline(1, color='k', alpha=0.4)
	axs.set_xlabel('Transmission Fitness Effect', fontsize=16, labelpad=15)
	axs.set_ylabel('Feature', fontsize=16, labelpad=25)
	recs = [mpl.patches.Rectangle((0,0),1,1, fc=c) for c in colors.values()]
	axs.legend(recs, colors.keys(), loc='upper right', fontsize=20)
	axs.tick_params(axis='both', labelsize=14)

	plt.tight_layout()
	plt.savefig(out_dir / f"profile_CIs_boxplot_non-background_sig.png", dpi=300)
	plt.close("all")

if __name__ == "__main__":
	from transmission_sim.ecoli.analyze import load_data_and_RO_from_file

	name = "3-interval_constrained-sampling"
	data, phylo_obj, RO, params, analysis_dir = load_data_and_RO_from_file(name)
	train = RO.loadDataByIdx(RO.train_idx)
	estimates = RO.results_dict['b0_TV+site']["full"]["estimates"]
	bdm_params = RO.results_dict['b0_TV+site']["bdm_params"]
	h_combo = RO.results_dict['b0_TV+site']["full"]["h_combo"]
	features_file = params["features_file"]

	# make_profile(train, estimates, features_file, bdm_params, h_combo, params["birth_rate_idx"], out_dir)
	# get_CIs(out_dir)

	box_plot(analysis_dir, analysis_dir)

	# cis_to_latex(out_dir)

	# color_list = sns.color_palette("Set2")
	# colors = {cat: color_list[i] for i, cat in enumerate(['AMR', 'PLASMID', 'VIR', 'STRESS'])}

	# def get_color(c, n_colors, i):
	# 	cat = c.rsplit("_", 1)[-1]
	# 	shades = sns.light_palette(colors[cat], n_colors=n_colors, as_cmap=False)
	# 	return shades[i]

	# def plot_shade_colors(n_colors, i, j):
	# 	plot_colors = []
	# 	for cat in colors.keys():
	# 		plot_colors.append(get_color(cat, n_colors, i))
	# 		plot_colors.append(get_color(cat, n_colors, j))

	# 	pp.plotColorPalette(plot_colors, save=out_dir / f"category_palette_{n_colors}-colors_{i}-{j}.png")

	# plot_shade_colors(6, -1, 2)
	# plot_shade_colors(12, -1, 2)
	# plot_shade_colors(12, -1, 1)

