from pathlib import Path
from multiprocessing import Pool
import yaml
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from ecoli_analysis.results_obj import SiteMod
from ecoli_analysis.utils import cat_display

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

def single_profile(profile_type, i, site, site_estimates, fit_model_kwargs, bdm_params, time_estimates):
	n_sites = len(site_estimates[0]) + len(time_estimates)

	mle_eff = site_estimates[0][i]
	site_results = {}
	
	# -----------------------------------------------------
	# Get loss at each value
	# -----------------------------------------------------
	for n in np.linspace(mle_eff - .75, mle_eff + .75, 200, endpoint=False):
		fit_model = SiteMod(**fit_model_kwargs)
		phylo_loss = fit_model.phylo_loss(**fit_model.loss_kwargs)

		if profile_type == "site":
			effs = np.array(site_estimates).copy()
			effs[0, i] = n
			fit_model.site = tf.Variable(effs, dtype=tf.dtypes.float64)
			fit_model.b0 = tf.Variable(time_estimates, dtype=tf.dtypes.float64)
		
		elif profile_type == "time":
			effs = np.array(time_estimates).copy()
			mle_eff = effs[i].copy()
			effs[i] = n
			fit_model.b0 = tf.Variable(effs, dtype=tf.dtypes.float64)
			fit_model.site = tf.Variable(site_estimates.copy(), dtype=tf.dtypes.float64)

		c = fit_model.call()

		weights = tf.concat([tf.reshape(v, [-1]) for v in fit_model.trainable_variables], axis=-1)
		loss = phylo_loss.call(c.__dict__, weights=weights).numpy()

		site_results[n] = loss

	print(f"Completed profile of {site} ({i+1}/{n_sites})")
	return {site: site_results}

def make_profiles(train, estimates, features_file, bdm_params, h_combo, birth_rate_idx, analysis_dir, plot_dir=None, plot_effect_profiles=False):
	if plot_effect_profiles:
		if not plot_dir:
			profile_dir = analysis_dir / "profiles"
		else:
			profile_dir = plot_dir / "profiles"

		profile_dir.mkdir(exist_ok=True, parents=True)

	features = pd.read_csv(features_file, index_col=0)

	if isinstance(estimates, dict):
		time_estimates = estimates.get('b0', [])
		site_estimates = estimates.get('site', [])

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

	def get_fit_model_kwargs():
		return dict(
		**bdm_params,
		birth_rate_idx=birth_rate_idx,
		data=train.returnCopy(),
		iterative_pE=True,
		loss_kwargs={**h_combo, "graph": False},
		)

	# ---------------------------------------------------------
	# Get likelihood profile of each time and profile estimate
	# ---------------------------------------------------------
	all_sites = [["site", i, site] for i, site in enumerate(site_names)] + [("time", i, site) for i, site in enumerate(time_names)]
	pool_args = [s + [site_estimates, get_fit_model_kwargs(), bdm_params, time_estimates] for s in all_sites]

	with Pool() as pool:
		profile_results_list = pool.starmap(single_profile, pool_args)

	profile_results = {site: site_profile for profile_dict in profile_results_list for site, site_profile in profile_dict.items()}

	if plot_effect_profiles:
		for i, site in enumerate(site_names):
			mle_eff = site_estimates[0][i]
			plot_effect_profile(site, profile_results, mle_eff, profile_dir)

	profile_df = pd.DataFrame.from_dict(profile_results).sort_index()
	profile_df.to_csv(analysis_dir / f"likelihood_profile_test.csv")

def is_significant(row):
	c_min = row['lower_CI']
	c_max = row['upper_CI']
	if (c_min > 1) and (c_max > 1):
		return True
	elif (c_min < 1) and (c_max < 1):
		return True
	else:
		return False

def get_CIs(analysis_dir):
	"""
	Given likelihood profiles, calculate confidence intervals
	"""

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

	df["included"] = df['mle'].between(.999, 1.001) == False
	df["significant"] = df.apply(lambda row: is_significant(row), axis=1)
	df.to_csv(analysis_dir / f"profile_CIs.csv")

def box_plot(df, category_palette, colors, order, out_fig):
	sns.set_style("whitegrid")
	fig, axs = plt.subplots(figsize=(12, 1 + .5 * df.shape[1]))
	sns.boxplot(
		data=df,
		palette={cat: color for cat, color in category_palette.items() if cat in df.columns},
		order=[o for o in order if o in df.columns],
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
	plt.savefig(out_fig, dpi=300)
	plt.close("all")

def do_box_plots(analysis_dir, out_dir, category_info_file, extra_plots=True):
	"""
	Output box plots of feature estimates with 95% CIs
	"""
	analysis_dir = Path(analysis_dir)
	out_dir = Path(out_dir)
	out_dir.mkdir(exist_ok=True, parents=True)

	# -----------------------------------------------------
	# Load confidence interval estimates
	# -----------------------------------------------------
	est_df = pd.read_csv(analysis_dir / f"profile_CIs.csv", index_col=0)
	est_df.drop(columns=["initial_mle"], inplace=True)

	df = est_df.copy()

	# -----------------------------------------------------
	# Re-set CI bounds so that plots correctly
	# -----------------------------------------------------
	# 	Doesn't matter what these values are, one just needs to be
	# 	slightly smaller than the lower CI, the other slightly larger
	# 	than the upper CI so that with a list of
	# 	N = 5 (a_lower, lower, mle, upper, a_upper), the index of
	# 	the first quartile is h = (5 - 1) * 1/4 + 1 = 2 (1 with 0-based indexing)
	# 	and the second element in the list is the lower bound
	# 	https://en.wikipedia.org/wiki/Quantile -- numpy uses linear interpolation
	df['artificial_lower_CI'] = df['lower_CI'] - .01
	df['artificial_upper_CI'] = df['upper_CI'] + .01

	df = df.T

	# -----------------------------------------------------
	# Format feature names for display
	# -----------------------------------------------------
	# Load display names
	dnames = yaml.load(Path(category_info_file).read_text(), Loader=yaml.CLoader)
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

	# -----------------------------------------------------
	# Set up colors
	# -----------------------------------------------------
	def get_color(c, df, display_df, colors):
		cat = display_df.loc[c, "Display Type"]
		shades = sns.light_palette(colors[cat], n_colors=12, as_cmap=False)
		if is_significant(df[c]):
			return shades[-1]
		else:
			return shades[2]

	color_list = sns.color_palette("Set2")
	colors = {cat: color_list[i] for i, cat in enumerate(['AMR', 'Plasmid Replicon', 'Virulence', 'Stress', 'Background'])}
	category_palette = {c: get_color(c, df, display_df, colors) for c in df.columns}
	order = est_df.sort_values(by='mle', ascending=True).index.to_list()

	# -----------------------------------------------------
	# Get non-dropped features, significant features
	# -----------------------------------------------------
	included_features = df.loc[:, df.loc['included'] == True].columns.to_list()
	sig_features = df.loc[:, df.loc['significant'] == True].columns.to_list()
	non_bg_features = [n for n in display_df[display_df['Display Type'] != 'Background'].index if n in df.columns]

	df = df.drop(index=['included', 'significant'])

	# -----------------------------------------------------
	# Box plot of all significant, non-background effects
	# -----------------------------------------------------
	wanted_features = list(set(sig_features).intersection(non_bg_features))
	box_plot(df[wanted_features], category_palette, colors, order, out_dir / f"Figure-4_profile_CIs_boxplot_non-background_sig.png")

	if extra_plots:
		# -----------------------------------------------------
		# Box plot of all effects
		# -----------------------------------------------------
		box_plot(df, category_palette, colors, order, out_dir / f"profile_CIs_boxplot.png")

		# -----------------------------------------------------
		# Box plot of all non-1 (not dropped out) effects
		# -----------------------------------------------------
		box_plot(df[included_features], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_non1.png")

		# -----------------------------------------------------
		# Box plot of all significant effects
		# -----------------------------------------------------
		box_plot(df[sig_features], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_all_sig.png")
		
		# -----------------------------------------------------
		# Box plot of all non-background effects
		# -----------------------------------------------------
		box_plot(df[non_bg_features], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_non-Background.png")

		# -----------------------------------------------------
		# Box plot of all non-1, non-background effects
		# -----------------------------------------------------
		wanted_features = list(set(included_features).intersection(non_bg_features))
		box_plot(df[wanted_features], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_non-background_non1.png")