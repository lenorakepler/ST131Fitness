from pathlib import Path
import pandas as pd
import re
import numpy as np

dir = Path("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/data_filters")

summ = pd.read_csv(dir / "initial_qc_summary.csv", index_col=0)

qc_files = [
	"assembly_success_summary.csv", 
	"confirmed_st_summary.csv", 
	"long_branch_summary.csv", 
	"long_branch_fail.txt",
	"post_reconstruction_bioproject_fail.txt"
	]

for file in qc_files:
	metric = file.split(".")[0]

	if "csv" in file:
		df = pd.read_csv(dir / file, index_col=0)

		if 'long_branch' in df.columns:
			df["long_branch"] = df["long_branch"] == False
			df = df.rename(columns={'long_branch': 'long_branch_a'})

		summ = pd.concat([summ, df[[c for c in df.columns if c != "pass"]]], axis=1)

	else:
		metric = metric.replace("_fail", "")
		fail_samples = (dir / file).read_text().splitlines()
		pass_samples = (dir / file.replace("fail", "pass")).read_text().splitlines()

		new_col = pd.Series({**{s: False for s in fail_samples}, **{s: True for s in pass_samples}})
		new_col.name = metric

		summ = pd.concat([summ, new_col], axis=1)

summ.loc[summ["st"].isna() == False, "st"] = summ.loc[summ["st"].isna() == False, "st"] == "131"
summ = summ.drop(columns=["pass"])

previous = summ.index.to_list()
for col in summ.columns:
	pass_samples = summ[summ[col] == True].index.to_list()
	print(f"\n{col}")
	print([s for s in pass_samples if s not in previous])

	previous = pass_samples

# config['pastml_dir'] + "/named.tree_lsd.date.noref.pruned.nwk"

# breakpoint()