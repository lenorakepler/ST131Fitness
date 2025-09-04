from pathlib import Path
import pandas as pd
import numpy as np
import re
from yaml import CDumper as Dumper, CLoader as Loader, load, dump
import json
from pandas.io.formats.style import Styler

def return_styler(df):
	props = 'font-family: "Helvetica, Arial", Helvetica, sans-serif; font-size: 0.95em; border: 1px solid black;'
	html = Styler(df, uuid_len=0, cell_ids=False)
	html.set_table_styles([
		{'selector': '', 'props': "border-collapse: collapse"},
		{'selector': '*', 'props': props},
		{'selector': 'td, th', 'props': "padding: .3em .3em .1em;"},
		{'selector': 'th.row_heading', 'props': "font-weight: normal;"},
		{'selector': 'td', 'props': "font-style: italic;"},
		])
	return html

def to_html(df, out_file):
	out = ""
	for cat, gdf in df.groupby("Category"):
		gdf = gdf.set_index(["Feature Group", 'Functional Feature Group', "Feature Name"]).sort_index()
		gdf = gdf.drop(columns=["Category"])
		gdf = gdf.replace("N/A", np.nan)
		gdf = gdf.dropna(axis=1)

		out += f"<h2>{cat}</h2>"
		html = return_styler(gdf)
		out += html.to_html(table_id="table")
	(Path(out_file)).write_text(out)

def print_dict(d):
	print(json.dumps(d, indent=4))

all_feature_info = pd.read_csv("all_feature_info_man-group.csv")
functional_corr_groups = load(Path("data_new/new_groups.yml").read_text(), Loader=Loader)
colinear_corr_groups = load(Path("data_new/functional_groups_corr/groups_0.95.yml").read_text(), Loader=Loader)
tip_final_features = pd.read_csv("data_new/functional_groups_corr/corr_grouped_binary_features_0.005_for-pastml.csv", index_col=0)
pastml_dict = load(Path("data_new/functional_groups_corr/corr_grouped_binary_features_pastml-dict.yml").read_text(), Loader=Loader)

# Flatten yaml for multi-index DataFrame
# =======================================
groups_flat = []
for group, group_dict in functional_corr_groups.items():
	group_info = {k: v for k, v in group_dict.items() if "members" not in k}
	group_info["Functional Feature Group"] = group_info.pop("display name")
	for member, member_long in zip(group_dict["members"], group_dict["members_long"]):
		groups_flat.append({'Functional Feature Group Long': group, "Feature Name": member, "Feature Name Long": member_long, **group_info})

df = pd.DataFrame(groups_flat)

def get_short_category(long_cat):
	if long_cat == "AMR":
		return "AMR"
	elif long_cat == "Virulence":
		return "VIR"
	elif long_cat == "Plasmid":
		return "PLASMID"
	elif long_cat == "Stress":
		return "STRESS"

# Create index denoting colinear (final) group
# =============================================
df["Feature Group"] = df["Functional Feature Group"]
df["Feature Group Long"] = df["Functional Feature Group Long"]
df["Feature Group Category"] = df["category"]
df["Category Short"] = df["category"].apply(lambda k: get_short_category(k))
df["Feature Group Category Short"] = df["Category Short"]

df = df.set_index("Functional Feature Group Long")
for feature_group, member_list in colinear_corr_groups.items():
	long_cat = '/'.join(sorted(list(set(df.loc[member_list, "category"]))))
	df.loc[member_list, "Feature Group Category"] = long_cat

	short_cat = '/'.join(sorted(list(set(df.loc[member_list, "Category Short"]))))
	df.loc[member_list, "Feature Group Category Short"] = short_cat
	df.loc[member_list, "Feature Group Long"] = f"{feature_group}_{short_cat}"

	df.loc[member_list, "Feature Group"] = feature_group
df = df.reset_index()

# Get updated feature counts
# =============================================
amr = pd.read_csv("_analysis/out/amr_gr90.csv", index_col=0)
plas = pd.read_csv("_analysis/out/plasmid_gr90.csv", index_col=0)
raw_feat = pd.concat([amr, plas], axis=1)
n_samples = len(raw_feat)
feat_counts = raw_feat.sum(axis=0)
feat_counts_str = {f: f"{c:.0f} ({c/n_samples*100:.1f})" for f, c in feat_counts.items()}

df["Count (%)"] = [feat_counts_str.get(f, '0 (0)') for f in df["Feature Name Long"]]
df = df.set_index("Feature Name")


# Update with info from all_feature_info
# =============================================
all_feature_info = all_feature_info.set_index("Feature Name")
df = pd.concat([df, all_feature_info[["Resistance Class", "Sequence Name", "AMRFinder+ Type"]]], axis=1).reset_index()
df = df.set_index(["Feature Group", "Functional Feature Group", "Feature Name"]).sort_index().reset_index()
df = df.rename(columns=lambda c: c.title())

# Get PastmL name
# =============================================
df["pastml"] = [pastml_dict.get(f, float("nan")) for f in df["Feature Group Long"]]
df["Count Cutoff Pass"] = df["pastml"].isna() == False

# To HTML
# =============================================
wanted_cols = ['Feature Group', 'Functional Feature Group', 'Feature Name', 'Category', 'Resistance', 'Type', 'Count (%)', 'Sequence Name']
to_html(df[wanted_cols], "data_new/feature_info.html")
to_html(df.loc[df["Count Cutoff Pass"] == True, wanted_cols], "data_new/feature_info_analyzed_only.html")
to_html(df.loc[df["Count Cutoff Pass"] == False, wanted_cols], "data_new/feature_info_below_threshold.html")

# To CSV
# =============================================
df = df.rename(columns=lambda c: c.lower().replace(" ", "_"))
df.to_csv("data_new/final_feature_info.csv")
