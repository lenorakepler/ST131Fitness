# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Set up the conda/mamba environment with:
```bash
mamba env create -f environment.yml
```

Activate the environment:
```bash
conda activate ecolienv
```

## Main Analysis Commands

### Run Full Analysis Pipeline
```bash
snakemake -s workflow/analysis.smk --cores 4 --configfile config.yaml
```

### Generate Figures
```bash
snakemake -s workflow/make_figures.smk --cores 4 --configfile config.yaml
```

### Direct Python Execution
The main entry point is `run.py`, which can be executed directly or imported for programmatic use.

## Architecture Overview

This is a phylodynamics analysis codebase for quantifying genomic determinants of fitness in E. coli ST131. The system implements:

### Core Components

1. **Phylogenetic Analysis Framework** (`analysis/` directory):
   - `phylo_obj.py`: Core phylogenetic object handling
   - `fitness_model.py`: TensorFlow-based birth-sampling-site model (`BirthSamplingSite`)
   - `phylo_loss.py`: Loss function implementations for phylodynamic modeling
   - `optimizer.py`: Model optimization routines
   - `do_model_fit.py`: Main model fitting orchestration with cross-validation

2. **E. coli Specific Analysis** (`ecoli_analysis/` directory):
   - `branch_fitness.py`: Branch-specific fitness calculations
   - `feature_matrix.py`: Genomic feature matrix handling
   - `fitness_decomp.py`: Fitness decomposition analysis
   - `likelihood_profile.py`: Likelihood profiling and confidence intervals

3. **Data Processing**:
   - Uses ancestral state reconstruction data in `data_new/`
   - Handles phylogenetic trees with interval-based sampling
   - Feature correlation analysis and grouping

### Key Technical Details

- **TensorFlow-based modeling**: Uses custom Keras models for phylodynamic inference
- **Cross-validation**: Implements k-fold CV with hyperparameter optimization
- **Multi-threading**: Supports parallel processing for parameter searches
- **Configuration-driven**: YAML-based configuration system with template processing (`yte`)

### Workflow Structure

The Snakemake workflows orchestrate:
1. Interval tree creation from phylogenetic data
2. Model fitting with hyperparameter optimization
3. Residual analysis and fitness decomposition
4. Figure generation for publication

### Data Organization

- `data_new/`: Main data directory with phylogenetic trees and feature matrices
- `configs/`: Multiple configuration files for different analysis scenarios
- Analysis outputs are organized by analysis name specified in config

### Important Configuration

- Main config: `config.yaml` - controls all analysis parameters
- Multiple model parameter configs in `configs/` for different fitting scenarios
- Bioproject sampling windows and changepoints are configurable

## Testing

The codebase includes test files in `_analysis/` but no formal testing framework is configured. Tests appear to be ad-hoc Python scripts for specific functionality validation.

## Dependencies

Key dependencies include:
- TensorFlow 2.15.0 for machine learning models
- BioPython for phylogenetic data handling
- Snakemake for workflow management
- Standard scientific Python stack (pandas, numpy, scipy, matplotlib, seaborn)
- Specialized packages: dendropy, ete3, baltic for phylogenetics