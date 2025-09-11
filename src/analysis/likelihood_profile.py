from pathlib import Path
from multiprocessing import Pool
import yaml
import click
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from model_fit.results_obj import ResultsObj
from model_fit.fitness_model  import BirthSamplingSite
from model_fit.phylo_loss import PhyloLoss, PhyloLossIterative
import time
import copy
import tqdm
import json
# from analysis.utils import cat_display

def plot_effect_profile(site, profile_results, mle_eff, CI, out_dir):
	profile = profile_results[site].dropna()
	profile_mle = profile.idxmin()
	profile_mle_loss = profile[profile_mle]

	plt.plot(list(profile.index), list(profile.values))
	plt.axvline(mle_eff, color='red', label=f"Estm MLE ({mle_eff:.3f})")
	plt.axvline(profile_mle, color='red', linestyle="--", label=f"Prof ({profile_mle:.3f}) - {profile_mle_loss:.2f}")

	if CI:
		plt.axvline(CI[0], color='blue', linestyle="--", label=f"Lower CI ({CI[0]:.3f})")
		plt.axvline(CI[1], color='blue', linestyle="--", label=f"Upper CI ({CI[1]:.3f})")

	if not np.isclose(mle_eff, profile_mle):
		flag = " - Unequal MLEs"
	else:
		flag = ""

	plt.xlabel(site)
	plt.ylabel('likelihood')
	plt.title(f'Likelihood surface around MLE of {site} effect{flag}')
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_dir / f'{site}.png')

	if flag:
		plt.savefig(out_dir / f'! {site}.png')
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

def score(var, i, p, name, data, iterative_pE, fit_model_params, reg_params):
	fp = fit_model_params
	fp[var]["value"][i] = p

	fit_model = BirthSamplingSite(
		data=data,
		iterative_pE=iterative_pE,
		fit_model_params=fp,
		)
	
	phylo_loss = PhyloLossIterative(graph=False, **reg_params) if iterative_pE else PhyloLoss(graph=False, **reg_params)

	c = fit_model.call()
	poss_weights = [tf.reshape(v, [-1]) for v in fit_model.trainable_variables if v.name in fit_model.penalize]
	weights = tf.cond(len(poss_weights) > 0, lambda: tf.concat(poss_weights, axis=-1), lambda: np.array([1.00]))
	loss = phylo_loss.call(c.__dict__, weights=weights).numpy()

	return {"name": name, "value": p, "loss": loss}

def wrap_score(args):
	return score(*args)

def make_profiles(results_obj, result_key, n_threads, plot_effect_profiles=False):
	analysis = json.loads((results_obj.folder / result_key / "validation.json").read_text())
	reg_params = analysis["h_combo"]

	data = results_obj.loadDataByIdx(results_obj.train_idx)

	# Replace starting parameter values with best-fitting estimates
	fit_params = results_obj.results_dict[result_key]["fit_model_params"]
	fit_params["rho"] = 0
	fit_params["gamma"] = 0
	fit_params["brownian_motion"]["info"] = fit_params["brownian_motion"]["full"]["train"]

	del results_obj.results_dict

	for k, v in analysis['estimates'].items():
		fit_params[k]["value"] = v

	for var, var_estimate in analysis['estimates'].items():
		profile_args = []
		names = fit_params[var]["names"]
		
		# Too many branch effects -- just want non-dropped-out
		if var == "brownian_motion":
			idx =  np.where(~np.isclose(var_estimate, 1, rtol=1e-03))[0].tolist()
			var_estimate = np.take(var_estimate, idx)
			names = np.take(names, idx)

		print(f"\n{var}")
		print("--------------------------------")
		print(f"Aggregating likelihood profile args")
		for i, mle in enumerate(var_estimate):
			profile_vals = np.linspace(mle - .75, mle + .75, 200, endpoint=False)
			profile_args += [[var, i, p, names[i], data, analysis['iterative_pE'], fit_params, reg_params] for p in profile_vals]

		print(f"Calculating likelihood profiles")
		if n_threads > 0:
			with Pool(int(n_threads)) as pool:
				profile_results_list = list(
					tqdm.tqdm(
						pool.imap(wrap_score, profile_args),
						total=len(profile_args)
						)
					)
		else:
			profile_results_list = []
			for profile_arg in profile_args:
				profile_results_list.append(wrap_score(profile_arg))

		profile_df = pd.DataFrame(profile_results_list).pivot(index="value", columns="name", values="loss").sort_index()
		profile_df.to_csv(results_obj.folder / result_key / f"{var}_likelihood_profile.csv")

def is_significant(row):
	c_min = row['lower_CI']
	c_max = row['upper_CI']
	if (c_min > 1) and (c_max > 1):
		return True
	elif (c_min < 1) and (c_max < 1):
		return True
	else:
		return False

def get_CIs(analysis_dir, plot_CI=True):
	"""
	Given likelihood profiles, calculate confidence intervals
	"""

	for like_profile in Path(analysis_dir).glob("*_likelihood_profile.csv"):
		var = like_profile.name.split("_likelihood_profile.csv")[0]

		estimates = json.loads((analysis_dir / f"{var}_estimates.json").read_text())
		like_profile = pd.read_csv(like_profile, index_col=0)

		plot_out_dir = analysis_dir / "figures" / "likelihood_profiles" / var
		plot_out_dir.mkdir(exist_ok=True, parents=True)

		ci_dict = {}

		for f, c in enumerate(like_profile.columns):
			flag = []

			# Get MLE estimate
			mle = estimates[c]

			# Get values where this feature was estimated
			feature_profile = like_profile.loc[like_profile[c].isna()==False, c]

			# Get profile MLE
			profile_mle = feature_profile.idxmin()

			if not np.isclose(profile_mle, mle):
				flag.append(f"differing MLE: {profile_mle} != {mle}")

			# Transform losses
			L = feature_profile.values * -1

			# Get value with lowest loss, calculate distance between
			# loss at MLE and loss at other values
			mle_index = np.argmax(L)
			deltaL = L - L[mle_index]

			# Get 95% CI upper and lower indices
			if len(deltaL[:mle_index]) == 0:
				# instance where likelihood is nan below mle
				lower_CI = profile_mle
				flag.append("Error in lower CI")

			else:
				lower_index = np.argmin(np.abs(deltaL[:mle_index] + 1.92))
				lower_CI = feature_profile.index[lower_index]
			
			upper_index = mle_index + np.argmin(np.abs(deltaL[mle_index:] + 1.92))
			upper_CI = feature_profile.index[upper_index]

			lower_CI_delta = np.abs(lower_CI - mle)
			upper_CI_delta = np.abs(upper_CI - mle)

			ci_dict[c] = dict(
				initial_mle=mle,
				mle=profile_mle,
				upper_CI=upper_CI,
				lower_CI=lower_CI,
				lower_CI_delta=lower_CI_delta,
				upper_CI_delta=upper_CI_delta,
				flag=flag,
			)

			if plot_CI:
				plot_effect_profile(c, like_profile, mle, [lower_CI, upper_CI], plot_out_dir)

		df = pd.DataFrame(ci_dict).T

		df["included"] = df['mle'].between(.999, 1.001) == False
		df["significant"] = df.apply(lambda row: is_significant(row), axis=1)
		df.to_csv(analysis_dir / f"{var}_profile_CIs.csv")

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

def do_box_plots(analysis_dir, out_dir, feature_info_file, extra_plots=True):
	"""
	Output box plots of feature estimates with 95% CIs
	"""
	analysis_dir = Path(analysis_dir)
	out_dir = Path(out_dir)
	out_dir.mkdir(exist_ok=True, parents=True)

	# -----------------------------------------------------
	# Load confidence interval estimates
	# -----------------------------------------------------
	profiles = analysis_dir.glob("*_profile_CIs.csv")
	est_dfs = []
	for p in profiles:
		pdf = pd.read_csv(p, index_col=0)
		pdf['variable_type'] = p.name.split("_profile")[0]
		est_dfs.append(pdf)

	est_df = pd.concat(est_dfs, axis=0)
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
	# Format categories for display
	# -----------------------------------------------------
	info_df = df.loc[['flag', 'included', 'significant', 'variable_type'], :].T
	info_df["Variable Type"] = info_df['variable_type'].str.replace("_", " ").str.replace("features", "feature").str.title()
	info_df["Category"] = info_df["Variable Type"]

	# -----------------------------------------------------
	# Re-format birth feature names, and update with 
	# more specific categories if we have them
	# -----------------------------------------------------
	if feature_info_file:
		feature_info = pd.read_csv(feature_info_file)
		cat_info = 	feature_info[['feature_group', 'feature_group_long', 'feature_group_category', 'feature_group_category_short', 'pastml']].drop_duplicates()
		cat_info = cat_info.dropna(subset="pastml").set_index("pastml")
		cat_info["New Category"] = cat_info["feature_group_category"].apply(lambda k: f"Birth Feature ({k})")

		birth_features = info_df[info_df["Variable Type"] == "Birth Feature"].index
		info_df.loc[birth_features, "Category"] = cat_info.loc[birth_features, "New Category"]
		
		df = df.rename(columns = {c: cat_info.loc[c, "feature_group"] for c in birth_features})
		info_df = info_df.rename(index = {c: cat_info.loc[c, "feature_group"] for c in birth_features})

	categories = sorted(info_df["Category"].unique().tolist())
	color_list = sns.color_palette("Set2") + [(0.522, 0.451, 0.631, 1)]
	colors = {cat: color for cat, color in zip(categories, color_list)}
	
	# -----------------------------------------------------
	# Set up colors
	# -----------------------------------------------------
	def get_color(row, colors):
		shades = sns.light_palette(colors[row["Category"]], n_colors=12, as_cmap=False)
		if row["significant"]:
			return shades[-1]
		else:
			return shades[2]

	category_palette = {f: get_color(row, colors) for f, row in info_df.iterrows()}
	
	order = df.loc['mle', :].sort_values(ascending=True).index.to_list()

	# -----------------------------------------------------
	# Get non-dropped features, significant features
	# -----------------------------------------------------
	included_features = info_df[info_df["included"] == True].index.to_list()
	sig_features = info_df[info_df['significant'] == True].index.to_list()
	unflagged_features = info_df[info_df['flag'] == '[]'].index.to_list()

	# non_bg_features = [n for n in display_df[display_df['Display Type'] != 'Background'].index if n in df.columns]

	df = df.drop(index=['included', 'significant', 'variable_type', 'included', 'flag', 'lower_CI_delta', 'upper_CI_delta'])

	# -----------------------------------------------------
	# Box plot of all significant, non-background effects
	# -----------------------------------------------------
	for cat, cdf in info_df.groupby("variable_type"):
		fl = cdf[cdf['flag'] != "[]"]
		box_plot(df[fl.index], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_{cat}_flag.png")

		unfl = cdf[cdf['flag'] == "[]"]
		box_plot(df[unfl.index], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_{cat}_all_noflag.png")

		sig = unfl[unfl['significant'] == True]

		box_plot(df[sig.index], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_{cat}_sig_noflag.png")

	# wanted_features = list(set(sig_features).intersection(unflagged_features).intersection(birth_features))
	# box_plot(df[wanted_features], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_sig_noflag.png")



	# -----------------------------------------------------
	# Box plot of all significant, non-background effects
	# -----------------------------------------------------
	# wanted_features = list(set(sig_features).intersection(non_bg_features))
	# box_plot(df[wanted_features], category_palette, colors, order, out_dir / f"Figure-4_profile_CIs_boxplot_non-background_sig.png")

	# if extra_plots:
	# 	# -----------------------------------------------------
	# 	# Box plot of all effects
	# 	# -----------------------------------------------------
	# 	box_plot(df, category_palette, colors, order, out_dir / f"profile_CIs_boxplot.png")

	# 	# -----------------------------------------------------
	# 	# Box plot of all non-1 (not dropped out) effects
	# 	# -----------------------------------------------------
	# 	box_plot(df[included_features], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_non1.png")

	# 	# -----------------------------------------------------
	# 	# Box plot of all significant effects
	# 	# -----------------------------------------------------
	# 	box_plot(df[sig_features], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_all_sig.png")
		
	# 	# -----------------------------------------------------
	# 	# Box plot of all non-background effects
	# 	# -----------------------------------------------------
	# 	box_plot(df[non_bg_features], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_non-Background.png")

	# 	# -----------------------------------------------------
	# 	# Box plot of all non-1, non-background effects
	# 	# -----------------------------------------------------
	# 	wanted_features = list(set(included_features).intersection(non_bg_features))
	# 	box_plot(df[wanted_features], category_palette, colors, order, out_dir / f"profile_CIs_boxplot_non-background_non1.png")

def get_named(results_obj):
	for result_key, results_dict in results_obj.results_dict.items():
		results_obj.results_dict[result_key]["full"]["named_estimates"] = {}
		for var, param_dict in results_dict['fit_model_params'].items():
			if isinstance(param_dict, dict) and param_dict.get('estimate', False):
				if 'features' in var:
					columns = pd.read_csv(param_dict["states"], index_col=0).columns

				elif var == 'branch_effects':
					columns = [b[0] for b in sorted(param_dict['branch_dict'].items(), key=lambda b: b[1])]

				elif "background" in var:
					columns = [str(c) for c in param_dict['changepoints']]

				est = results_dict["full"]["estimates"][var]
				if len(est) == 1:
					est = est[0]

				param_dict["names"] = [c for c in columns]

				results_obj.results_dict[result_key]["full"]["named_estimates"][var] = {k: v for k, v in zip(columns, est)}
		
	results_obj.save()

@click.command()
@click.argument("command")
@click.option("--analysis_dir", "-a", default="data_new/analysis/three_sampling_intervals")
@click.option("--result_key", "-k", default="full_model_birth_features+brownian_motion+sampling_background_TV+sampling_features")
@click.option("--n_threads", "-n", default=6)
@click.option("--debug", "-d", is_flag=True)
def main(command, analysis_dir, result_key, n_threads, debug):
	RO = ResultsObj(Path(analysis_dir))

	if debug:
		n_threads = 0

	if command == "profile":
		make_profiles(results_obj=RO, result_key=result_key, n_threads=n_threads, plot_effect_profiles=True)
	elif command == "ci":
		get_CIs(RO.folder / result_key)
	elif command == "plot":
		do_box_plots(RO.folder / result_key, RO.folder / result_key / "figures" / "box_plots", "data_new/final_feature_info.csv", extra_plots=True)

if __name__ == "__main__":
	main()