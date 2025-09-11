from pathlib import Path
import pandas as pd

dir = Path("/Users/lenorakepler/Dropbox/NCSU/Lab/ESBL-HAI/NCBI_Dataset/final/")
samples = (dir / "data_filters" / "post_reconstruction_bioproject_pass.txt").read_text().splitlines()
amr_dir = dir / "amrfinder"

amr_csvs = {sample: amr_dir / f"{sample}_amrfinder.tsv" for sample in samples}

def get_pres_abs(files_dict, output_file, threshold, type="AMRFinder"):
    all_presence = []
    results_dict = {}
    for sample, file in files_dict.items():
        res = pd.read_csv(file, sep="\t")
        res = res[(res["% Coverage of reference sequence"] >= threshold) & (res["% Identity to reference sequence"] >= threshold) & (res["Method"] != "INTERNAL_STOP")]

        res.loc[:, 'name'] = res[["Gene symbol", "Element type"]].apply("_".join, axis=1)
        
        # flag = res[res['Method'].isin(['PARTIALX', 'PARTIAL_CONTIG_ENDX'])]
        # if len(flag) > 0:
        #     print(flag.loc[:, ['name', "Contig id", 'Method', "% Coverage of reference sequence", "% Identity to reference sequence"]])

        # Some samples have more than one of the same gene, etc. but we only want one
        sample_present = list(set(res['name'].to_list()))
        # if len(list(set(sample_present))) != len(sample_present):
        #     seen = set()
        #     dupes = [x for x in sample_present if x in seen or seen.add(x)]
        #     for dupe in dupes:
        #         print(res.loc[res['name'] == dupe, ['name', "Contig id", "% Coverage of reference sequence", "% Identity to reference sequence"]])
        
        all_presence += sample_present
        results_dict[sample] = sample_present

    all_genes = list(set(all_presence))

    df = pd.DataFrame(columns=all_genes, index=results_dict.keys())
    for k, v in results_dict.items():
        df.loc[k, v] = 1

    df.fillna(0, inplace=True)
    df.rename(columns={c: c.replace("VIRULENCE", "VIR") for c in df.columns if "VIRULENCE" in c}, inplace=True)
    df.to_csv(output_file)

def get_pres_abs_plasmid(file, output_file, threshold):
    all_presence = []
    results_dict = {}

    res = pd.read_csv(file, sep="\t")
    res = res[(res["%Identity"] >= threshold) & (res["%Overlap"] >= threshold)]

    res['Plasmid'] = res['Plasmid'].apply(lambda k: k + "_PLASMID")

    for sample, sdf in res.groupby('Isolate ID'):
        sample_present = list(set(sdf['Plasmid'].to_list()))
        all_presence += sample_present
        results_dict[sample] = sample_present

    all_genes = list(set(all_presence))

    df = pd.DataFrame(columns=all_genes, index=results_dict.keys())
    for k, v in results_dict.items():
        df.loc[k, v] = 1

    df.fillna(0, inplace=True)
    df.to_csv(output_file)

# get_pres_abs(amr_csvs, Path("_analysis") / "out" / "amr_gr90.csv", 90)
# get_pres_abs(amr_csvs, Path("_analysis") / "out" / "amr_gr0.csv", 0)

get_pres_abs_plasmid(dir / "staramr" / "plasmidfinder.tsv", Path("_analysis") / "out" / "plasmid_gr0.csv", 0)
get_pres_abs_plasmid(dir / "staramr" / "plasmidfinder.tsv", Path("_analysis") / "out" / "plasmid_gr90.csv", 90)

gr90 = pd.read_csv(Path("_analysis") / "out" / "plasmid_gr90.csv", index_col=0)
gr0 = pd.read_csv(Path("_analysis") / "out" / "plasmid_gr0.csv", index_col=0)

gr90[[c for c in gr0.columns if c not in gr90.columns]] = 0

diff = ((gr0.sum(axis=0) - gr90.sum(axis=0))).to_frame(name="count_diff")
diff['percent_diff'] = diff / len(gr90.index)
diff['0_thresh_count'] = gr0.sum(axis=0)
diff['90_thresh_count'] = gr90.sum(axis=0)
diff = diff.sort_values(by="count_diff", ascending=False)
diff.to_csv(Path("_analysis") / "out" / "plasmid_threshold_diff.csv")

res = pd.read_csv(dir / "staramr" / "plasmidfinder.tsv", sep="\t")
res = res[(res["%Identity"] < 90) | (res["%Overlap"] < 90)]
fib = res[res["Plasmid"] == "IncFIB(AP001918)"]

q1 = res[res["Plasmid"] == "IncQ1"]

breakpoint()

# diff = pd.read_csv(Path("_analysis") / "out" / "threshold_diff.csv", index_col=0)
# concat = pd.concat([pd.read_csv(a, sep="\t") for a in amr_csvs.values()], axis=0)
# concat.loc[:, 'name'] = concat[["Gene symbol", "Element type"]].apply("_".join, axis=1).replace("VIRULENCE", "VIR", regex=True)

# cols = ['name', "Contig id", 'Method', "% Coverage of reference sequence", "% Identity to reference sequence"]

# for feat in diff.iloc[0:5, :].index:
#     print(f"\n{feat}=====")
#     print("(% cov, % ident, method)")
#     feat_rows = concat.loc[concat['name'] == feat, cols]
#     combs = pd.Series([tuple(r) for r in feat_rows[["% Coverage of reference sequence", '% Identity to reference sequence', 'Method']].values])
#     print(combs.value_counts())

