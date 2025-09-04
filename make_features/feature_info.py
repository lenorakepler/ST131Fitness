from pathlib import Path
import pandas as pd
import numpy as np
from yaml import CLoader as Loader, CDumper as Dumper
import yaml
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

feat = yaml.load(Path("data/group_short_name_to_display_manual.yml").read_text(), Loader=Loader)
amr = pd.read_csv("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/amrfinder/amrfinder_features.csv", index_col=0)
amr = amr.drop(columns=["Reference sequence length", "Accession of closest sequence"])

for f, row in amr.iterrows():
	for grp, gdict in feat.items():
		if "members" in gdict:
			if isinstance(gdict["members"], list):
				gdict["members"] = {m: {} for m in gdict["members"]}
			
			if f in gdict["members"]:
				seq_name = row["Sequence name"]
				if row["Name of closest sequence"] != seq_name:
					seq_name += f" ({row['Name of closest sequence']})"

				kind = row["Element type"]
				if (sub:=row["Element subtype"]) != kind:
					if not pd.isna(sub):
						kind += f" ({sub.title()})"
					kind = kind.title()

				clss = row["Class"]
				if not pd.isna(clss):
					if row["Subclass"] != clss:
						clss += f" ({row['Subclass'].title()})"
					clss = clss.title()
				else:
					clss = "N/A"

				gdict["members"][f] = {
					# 'AMRFinder+ Scope': row['Scope'],
					'AMRFinder+ Type': kind,
					'Resistance Class': clss,
					'Sequence Name': seq_name,
					}

print(yaml.dump(feat, default_flow_style=False, sort_keys=False))

# star = pd.read_csv("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/staramr/summary.tsv", sep="\t", index_col=0)
# sens = star[["Predicted Phenotype"]]
# sens["Sensitive"] = sens["Predicted Phenotype"] == "Sensitive"
# sens = sens.drop(columns=["Predicted Phenotype"])
# sens.to_csv("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/staramr/sensitive.csv")

fd = []
for grp, gdict in feat.items():
	if "members" in gdict:
		cat = gdict['category']

		if cat == 'Vir':
			cat = "Virulence"

		# if gdict['type']:
		# 	cat += f" ({gdict['type']})"

		# else:
		# 	if cat == 'Amr':
		# 		cat += f" (Gene)"

		fg = gdict['display_name']

		if ' + ' in fg:
			fg += " (*)"

		for f, fdict in gdict['members'].items():
			fdict['Feature Name'] = f
			fdict['Category'] = cat
			fdict['Feature Group'] = fg
			fd.append(fdict)
	# else:
	# 	print(grp)
	
df = pd.DataFrame(fd)
df = df.replace("Amr", "AMR", regex=True)
df = df.replace("VIRULENCE", "Virulence", regex=True)

out = ""

# Ancestral states file
all_anc = pd.read_csv("data/combined_ancestral_states_binary.csv", index_col=0)

# Samples only
so = all_anc.loc[[i for i in all_anc.index if 'SAMN' in i], :]
so = so.rename(columns=lambda c: c.rsplit("_", maxsplit=1)[0])

# Count
count = so.sum(axis=0)

pastml_names = [f.replace("(", "").replace(")", "").replace("/", "").replace("-", "").replace("\'", "") for f in df['Feature Name']]
df.insert(1, "Count (%)", [f'{count[f]:.0f} ({(count[f]/len(so))*100:.1f})' for f in pastml_names])

for cat, gdf in df.groupby("Category"):
	print(cat)

	gdf = gdf.sort_values(by=["Feature Group", "Feature Name"])
	gdf = gdf.set_index(["Feature Group", "Feature Name"])
	gdf = gdf.drop(columns=["Category"])
	gdf = gdf.replace("N/A", np.nan)
	gdf = gdf.dropna(axis=1)

	out += f"<h2>{cat}</h2>"
	html = return_styler(gdf)
	out += html.to_html(table_id="table")
	(Path("feature_info.html")).write_text(out)

df = df.sort_values(by=["Category", "Feature Group", "Feature Name"])
df = df.set_index(["Category", "Feature Group", "Feature Name"])
df.to_csv(f"all_feature_info.csv")

# group_counts = df.groupby('Feature Group').count()['Feature Name']
# to_drop = group_counts[group_counts <= 1].index
# group_only = df[df['Feature Group'].isin(to_drop)==False]

# out = ""

# for cat, gdf in group_only.groupby("Category"):
# 	print(cat)

# 	gdf = gdf.sort_values(by=["Feature Group", "Feature Name"])
# 	gdf = gdf.set_index(["Feature Group", "Feature Name"])
# 	gdf = gdf.drop(columns=["Category"])
# 	gdf = gdf.replace("N/A", np.nan)
# 	gdf = gdf.dropna(axis=1)

# 	out += f"<h2>{cat}</h2>"
# 	html = return_styler(gdf)
# 	out += html.to_html(table_id="table")
# 	(Path("feature_info_grouped_only.html")).write_text(out)
