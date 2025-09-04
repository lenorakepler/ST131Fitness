from pathlib import Path
import pandas as pd
import numpy as np
from plot_correlation import plot_correlation_structured, plot_full_correlation, corr_heatmap
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

dir = Path("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/amrfinder")
dfs = [pd.read_csv(f, sep="\t") for f in dir.glob("*_amrfinder.tsv")]
df = pd.concat(dfs, axis=0)
df.to_csv(dir / "all.csv", index=False)

df = df.set_index("Gene symbol")
feat = df.loc[~df.index.duplicated(), :]

feat = feat.drop(columns=['Contig id', 'Start', 'Stop', 'Strand', 'Method', 'Target length', '% Coverage of reference sequence', '% Identity to reference sequence', 'Alignment length'])
feat = feat.dropna(how="all", axis=1)
feat.to_csv(dir / "amrfinder_features.csv")
breakpoint()