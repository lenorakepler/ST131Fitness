from pathlib import Path
import pandas as pd
import numpy as np
from _analysis.plot_correlation import plot_correlation_structured, plot_full_correlation, corr_heatmap
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import analysis.plot_phylo_standalone as pp
import seaborn as sns
import matplotlib.pyplot as plt

all_anc = pd.read_csv("data/combined_ancestral_states_binary.csv", index_col=0)

qrdr = all_anc[[c for c in all_anc.columns if 'par' in c or 'gyr' in c]]
qrdr = qrdr.rename(columns=lambda x: x.replace("_AMR", "")).astype("int")
qrdr_features = qrdr.columns.to_list()

# plot_full_correlation(qrdr, filename="../data/gyr_states.csv")

corr_file = Path("data/gyr_states_full-corr.csv")
# plot_correlation_structured(corr_file, qrdr, sep=",", methods=["average"], filename="qrdr.csv")

# Do with only sampled tips
so = qrdr.loc[[i for i in qrdr.index if 'SAMN' in i], :]
# plot_correlation_structured(corr_file, so, sep=",", methods=["average"], filename="qrdr_so.csv")

# Do with only sampled tips, qrdr that meet threshold
N = len(so)
n = N * .005
print(f"0.5% threshold is n={n}. Removing features with < {n} count or >= {N - n}")

count_gr = so.sum(axis=0) > n
count_lt = so.sum(axis=0) <= (N - n)
count = count_gr & count_lt

qrdr = qrdr.loc[:, count]
# plot_correlation_structured(corr_file, qrdr, sep=",", methods=["average"], filename="qrdr_diverse.csv")

qrdr.to_csv("qrdr_only.csv")

# Only correlation is parC_S80I gyrA_D87N
qrdr["parC_S80I+gyrA_D87N"] = ((qrdr["parC_S80I"] + qrdr["gyrA_D87N"]) > 1).astype(int)
qrdr = qrdr.drop(columns=["parC_S80I", "gyrA_D87N"])
qrdr = qrdr[[c for c in qrdr.columns if 'parE' not in c]]
qrdr_features = qrdr.columns.to_list()

# Get unique combinations
qrdr["haplo"] = qrdr.apply(lambda x: ' + '.join(str(qrdr_features[i]) for i, b in enumerate(x) if b == 1), axis=1)
unique = qrdr["haplo"].unique().tolist()

# Plot
tt = pp.loadTree(
	"data/named.tree_lsd.date.noref.pruned_unannotated.nwk",
	internal=True,
	abs_time=2023,
)

color_list = sns.color_palette("Paired")
colors, c_func = pp.categoricalFunc(trait_dict=qrdr['haplo'].to_dict(), trait="name", legend=True, null_color="#eeeeee", color_list=color_list)

fig, ax = plt.subplots(figsize=(12, 40))
ax = pp.plotTraitAx(
	ax,
	tt,
	edge_c_func=c_func,
	node_c_func=c_func,
	tip_names=False,
	zoom=None,
	title="QRDR Mutations",
)


all_change = []
df = pd.read_csv("qrdr_only.csv", index_col=0)
df = df[[c for c in df.columns if 'parE' not in c]]
for feature in df.columns:
	change_file = f"/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/feature_changepoints/{feature}_AMR.csv"
	changepoints = pd.read_csv(change_file)
	changepoints.set_index("parent_name", inplace=True)
	changepoints = changepoints.loc[~changepoints.index.duplicated()]
	changepoints['feature'] = feature

	all_change.append(changepoints)

changepoints = pd.concat(all_change, axis=0)
changepoints = changepoints.loc[changepoints["child_name"].str.contains('SAMN') == False]

string_dict = {}
for i, gdf in changepoints.groupby("change_date"):
	print(i)
	print(gdf)
	chngstr = ''
	for j, row in gdf.iterrows():
		if row['child_clade'] == 1:
			chngstr += f"+{row['feature']}\n"
		else:
			chngstr += f"-{row['feature']}\n"
	
	rep = gdf.index[0]
	string_dict[rep] = chngstr

breakpoint()

# Set x and y text coordinates, positioning
text_x_attr = lambda k: k.absoluteTime - 2
text_y_attr = lambda k: k.y - 5
kwargs = {'va': 'top', 'ha': 'right', 'size': 14}

gain = changepoints[changepoints["child_clade"] == 1]
loss = changepoints[changepoints["child_clade"] == 0]

# Annotate acquisition events
target_func = lambda k: k.traits['name'] in string_dict
text_func = lambda k: f"{string_dict[k.traits['name']]}"
tt.addText(ax, x_attr=text_x_attr, y_attr=text_y_attr, target=target_func, text=text_func, **kwargs)
tt.plotPoints(ax, x_attr=lambda k: k.absoluteTime, y_attr=lambda k: k.y, target=target_func, size=36, colour="black")

# # Annotate loss events
# kwargs["color"] = "firebrick"
# target_func = lambda k: k.traits['name'] in loss.index.to_list()
# text_func = lambda k: f"{feature}\n{loss.loc[k.traits['name'], 'change_date']:.1f}"
# tt.addText(ax, x_attr=text_x_attr, y_attr=text_y_attr, target=target_func, text=text_func, **kwargs)
# tt.plotPoints(ax, x_attr=lambda k: k.absoluteTime, y_attr=lambda k: k.y, target=target_func, size=36, colour="firebrick")

ax = pp.add_legend(colors, ax, "lower left")
# pp.add_cmap_colorbar(fig, ax, cmap, norm=norm)
plt.tight_layout()
plt.savefig("amr_gyr_haplo_noParE.png", dpi=300)