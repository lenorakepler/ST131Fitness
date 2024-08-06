# ST131Fitness

Data, code, and instructions for the analysis presented in "Quantifying the genomic determinants of fitness in *E. coli* ST131 using phylodynamics" (https://www.biorxiv.org/content/10.1101/2024.06.10.598183v1).

For questions or more information, contact lkepler at ncsu.edu!

## Setting up the virtual environment
From the base directory of the folder, run:

```
mamba env create -f environment.yml
```
Note: I *highly* recommend using `mamba`  rather than `conda` to resolve package dependencies as it is much quicker, but `conda` should also work.

## Using the Code
For convenience, two snakemake files have been created: one that runs through the steps of the actual analysis (`analysis.smk`) and one that generates the figures (`make_figures.smk`). Both use a config file (`config.yaml`) that can be used as-is to use the same parameters as the paper, or can be modified for alternate analyses.

To use, run one of the below commands from within the base directory of the folder.

### analysis.smk

Performs model fitting, residual model fitting, and fitness decomposition.

```
snakemake -s workflow/analysis.smk --cores 4 --configfile config.yaml
```

### make_figures.smk

Makes the figures included in the paper.

```
snakemake -s workflow/make_figures.smk --cores 4 --configfile config.yaml
```





