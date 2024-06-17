# ST131Fitness

Data, code, and instructions for the analysis presented in "Quantifying the genomic determinants of fitness in *E. coli* ST131 using phylodynamics" (https://www.biorxiv.org/content/10.1101/2024.06.10.598183v1).

For questions or more information, contact lkepler at ncsu.edu!



## Using the Code

**\*Note: An environment configuration file for setting up necessary packages will be added shortly, as will a brief documentation of the workflow, file structure, and snakemake pipeline\*** 

For convenience, two snakemake files have been created: one that runs through the steps of the actual analysis (`analysis.smk`) and one that generates the figures (`make_figures.smk`). Both use a config file (`config.yaml`) that can be used as-is to use the same parameters as the paper, or can be modified for alternate analyses.



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





