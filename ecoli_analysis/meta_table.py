from pathlib import Path
import pandas as pd
import numpy as np

def master_qc_list(meta_sheet, data_dir, out_dir):
	# Get all USA clinical ST131 samples
	# ----------------------------------
	meta = pd.read_csv(meta_sheet, index_col=0)

	# Create table with whether not excluded, exclusion reasons
	# ----------------------------------
	df = pd.DataFrame(index=meta.index, columns=["included_in_analysis", "exclusion_reason"])
	df['included_in_analysis'] = True
	
	pre_qc = pd.read_csv(data_dir / "initial_qc_summary.tsv", index_col=0)

	for col, fail_reason in [('date', 'Absent or Invalid Date'), ('reads', 'No Reads Available'), ('bioproject_num', 'Too Few in Bioproject')]:
		failed = pre_qc[pre_qc[col]==False].index
		df.loc[failed, ["included_in_analysis", "exclusion_reason"]] = [False, fail_reason]

	filters = [
		("assembly_success", "Failed Assembly"), 
		("confirmed_st", "Unconfirmed Serotype"),
		("long_branch", "Long Branch"),
		("post_reconstruction_bioproject", 'Too Few in Bioproject Post-QC')
		]

	for filter, fail_reason in filters:
		failed = (data_dir / f"{filter}_fail.txt").read_text().splitlines()
		df.loc[failed, ["included_in_analysis", "exclusion_reason"]] = [False, fail_reason]

	# Add back in other wanted columns and save
	# --------------------------------------------
	wanted_columns=['bioproject_id', 'assembly_id', 'isolate_id', 'read_illumina_id', 'read_nanopore_id', 'read_pacbio_id', 'location', 'specimen_type', 'collection_date']
	df = pd.concat([df, meta[wanted_columns]], axis=1)
	
	df.to_csv(out_dir / "sample_info.csv")

	df[df['included_in_analysis'] == True].to_csv(out_dir / "sample_info_included.csv")
	df[df['included_in_analysis'] == False].to_csv(out_dir / "sample_info_excluded.csv")

	# Save dataframe of bioprojects included in final analysis
	# ----------------------------------------------------------
	bioprojects = df.loc[df["included_in_analysis"]==True, "bioproject_id"].value_counts().to_frame(name="isolates_included")
	bioprojects.index.name = "bioproject_id"

	bioprojects.to_csv(out_dir / "bioprojects_included.csv")

	excl = df.loc[df['bioproject_id'].isin(bioprojects.index) == False, "bioproject_id"].value_counts().to_frame(name="num_isolates")
	excl.index.name = "bioproject_id"
	excl.to_csv(out_dir / "bioprojects_excluded.csv")

if __name__ == "__main__":
	if Path("/home/lenora/Dropbox").exists():
		dropbox_dir = Path("/home/lenora/Dropbox")

	else:
		dropbox_dir = Path("/Users/lenorakepler/Dropbox")

	ncbi = dropbox_dir / "NCSU/Lab/ESBL-HAI/NCBI_Dataset"
	dir = ncbi / "final"

	name = "3-interval_constrained-sampling"
	analysis_dir = dir / "analysis" / name

	data_dir = dir / "data_filters"

	master_qc_list(dir / "meta" / "combined_meta.csv", data_dir, out_dir=dir)





