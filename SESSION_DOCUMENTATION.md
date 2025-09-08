# ST131Fitness Codebase Analysis Session Documentation

This document contains comprehensive analysis and documentation generated during a Claude Code session for the ST131Fitness phylodynamics research codebase.

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [CLAUDE.md Creation](#claudemd-creation)
3. [Dependency Analysis](#dependency-analysis)
4. [Function Documentation](#function-documentation)

---

## Repository Overview

### Project Description
Data, code, and instructions for the analysis presented in "Quantifying the genomic determinants of fitness in *E. coli* ST131 using phylodynamics" (https://www.biorxiv.org/content/10.1101/2024.06.10.598183v1).

### Key Architecture Components

**Core Analysis Framework** (`analysis/` directory):
- `phylo_obj.py`: Core phylogenetic object handling
- `fitness_model.py`: TensorFlow-based birth-sampling-site model (`BirthSamplingSite`)
- `phylo_loss.py`: Loss function implementations for phylodynamic modeling
- `optimizer.py`: Model optimization routines
- `do_model_fit.py`: Main model fitting orchestration with cross-validation

**E. coli Specific Analysis** (`ecoli_analysis/` directory):
- `branch_fitness.py`: Branch-specific fitness calculations
- `feature_matrix.py`: Genomic feature matrix handling
- `fitness_decomp.py`: Fitness decomposition analysis
- `likelihood_profile.py`: Likelihood profiling and confidence intervals

**Technical Details:**
- **TensorFlow-based modeling**: Uses custom Keras models for phylodynamic inference
- **Cross-validation**: Implements k-fold CV with hyperparameter optimization
- **Multi-threading**: Supports parallel processing for parameter searches
- **Configuration-driven**: YAML-based configuration system with template processing (`yte`)

---

## CLAUDE.md Creation

The following CLAUDE.md file was created to provide guidance for future Claude Code instances:

```markdown
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
```

---

## Dependency Analysis

### Command Analyzed
```bash
python -m analysis.analyze_fit "single" "full-intercept-tvb" -k "lamb=1_reg_type=l1_sigma=0.05_lr=5e-05_n_epochs=80000_fold-2"
```

### Required Files (Keep These - 16 files)

**Core analysis module:**
- `analysis/__init__.py`
- `analysis/analyze_fit.py` (entry point)
- `analysis/do_model_fit.py` (ResultsObj)
- `analysis/fitness_model.py` (BirthSamplingSite)
- `analysis/arrayer.py` (PhyloDataFile, PhyloArrayer)
- `analysis/phylo_loss.py` (PhyloLoss, PhyloLossIterative) 
- `analysis/phylo_obj.py` (used by do_model_fit)
- `analysis/optimizer.py` (used by do_model_fit, arrayer)
- `analysis/plot_phylo_standalone.py` (used by branch_fitness, feature_matrix)

**Required ecoli_analysis files:**
- `ecoli_analysis/__init__.py`
- `ecoli_analysis/branch_fitness.py` (color_tree function)
- `ecoli_analysis/feature_matrix.py` (load_info function)
- `ecoli_analysis/utils.py` (used by feature_matrix)

### Removable Files (35+ files)

**analysis module (2 files):**
- `analysis/random_effects.py`
- `analysis/random_fill.py`

**analysis_old module (3 files):**
- `analysis_old/__init__.py`
- `analysis_old/param_model.py`  
- `analysis_old/pm2.py`

**_analysis module (16 files):**
- `_analysis/__init__.py`
- `_analysis/check_coverage_identity.py`
- `_analysis/corr_group.py`
- `_analysis/data_filters.py`
- `_analysis/feature_info.py`
- `_analysis/feature_table.py`
- `_analysis/feature_thresholds.py`
- `_analysis/gyr_par_reparam.py`
- `_analysis/pastml.py`
- `_analysis/plot_ancestral.py`
- `_analysis/plot_correlation.py`
- `_analysis/re_param.py`
- `_analysis/redo_test.py`
- `_analysis/test_parallel.py`
- `_analysis/test_prob.py`
- `_analysis/test_sigma_sign.py`

**phyloTF2 module (6 files):**
- `phyloTF2/__init__.py`
- `phyloTF2/CovidFullModelGenSpaceTimeWrapper.py`
- `phyloTF2/GenSpaceTimeTensorTree.py`
- `phyloTF2/TensorTree.py`
- `phyloTF2/TreeUtils.py`
- `phyloTF2/test.py`

**ecoli_analysis unused files (3 files):**
- `ecoli_analysis/fitness_decomp.py`
- `ecoli_analysis/likelihood_profile.py`
- `ecoli_analysis/plot_clades.py`

**Root directory files (5+ files):**
- `run.py`
- `test.py`
- `scratch.py`
- `check_overfitting.py`
- `walk_code.py`

**configs/old/ and ecoli_external_workflow/:**
All Python files in these directories can be removed as they're legacy/experimental code or external workflow scripts.

### Summary
- **Keep: 16 Python files** (essential for the command)
- **Remove: 35+ Python files** (unused by this specific command)

---

## Function Documentation

### Command Execution Flow

**Command execution path:**
1. `main()` → parses arguments and routes to `old_vs_new_single()`
2. `old_vs_new_single()` → loads data, compares models, creates visualizations
3. Helper functions: `lj()`, `simple_scatter_with_xy()`, `color_tree()`
4. Class instantiations: `PhyloDataFile()`, `color_tree()` via imported modules

### Main Entry Point Functions

#### `main(command, model, combo_key="")`
**Location:** `analysis/analyze_fit.py:452`

**Purpose:** CLI entry point that parses command-line arguments and routes execution to appropriate analysis functions.

**Parameters:**
- `command` (str): The analysis command to execute (e.g., "single", "plot", "compare_old")
- `model` (str): Model type identifier (e.g., "full-intercept-tvb") 
- `combo_key` (str, optional): Specific parameter combination key for single analysis

**Behavior:**
- Maps model aliases to full result keys:
  - `"full-intercept-tvb"` → `"full_model_birth_background_TV+birth_features+brownian_motion+sampling_background_TV+sampling_features"`
- Constructs results directory path: `"data_new/analysis/three_sampling_intervals/{result_key}"`
- Routes to specific analysis function based on command argument
- For `command="single"`: calls `old_vs_new_single(results_dir, combo_key)`

**Returns:** None (void function that executes analysis)

### Core Analysis Functions

#### `old_vs_new_single(results_dir, combo_key)`
**Location:** `analysis/analyze_fit.py:220`

**Purpose:** Compares phylogenetic random effects estimates between old and new model implementations for a specific parameter combination.

**Parameters:**
- `results_dir` (Path): Directory containing analysis results 
- `combo_key` (str): Parameter combination identifier (e.g., "lamb=1_reg_type=l1_sigma=0.05_lr=5e-05_n_epochs=80000_fold-2")

**Workflow:**
1. **Load Results Data:**
   - Loads combo-specific results: `{combo_key}.json`
   - Loads model parameters: `fit_model_params.json`  
   - Loads analysis parameters: `params.json`

2. **Process New Estimates:**
   - Creates DataFrame indexed by brownian motion branch names
   - Extracts new brownian motion estimates from results

3. **Load Legacy Data:**
   - Reads old random effects from hard-coded path: `/Users/lenorakepler/Documents/GitHub/ST131Fitness/data/analysis/3-interval_constrained-sampling/Est-Random_Fixed-BetaSite/edge_random_effects_all.csv`
   - Cleans branch names by removing "_interval" suffix
   - Removes duplicate entries

4. **Generate Scatter Plot:**
   - Combines old and new estimates
   - Creates scatter plot comparing old vs new random fitness values
   - Saves plot as: `old_v_new_random_{combo_key}.png`

5. **Create Phylogenetic Visualization:**
   - Loads phylogenetic array data using `PhyloDataFile`
   - Maps branch names to fitness values
   - Generates colored phylogenetic tree using `color_tree()`
   - Saves as: `phylo_fitness_random__{combo_key}.png`

**Returns:** None (generates files and prints DataFrames)

### Helper Functions

#### `lj(file)`
**Location:** `analysis/analyze_fit.py:25`

**Purpose:** Convenience function for loading JSON files.

**Parameters:**
- `file` (str/Path): Path to JSON file

**Returns:** Parsed JSON data as Python object

**Implementation:**
```python
return json.loads(Path(file).read_text())
```

#### `simple_scatter_with_xy(x_vals, y_vals, title, out_file)`
**Location:** `analysis/analyze_fit.py:124`

**Purpose:** Creates a scatter plot with diagonal reference line for comparing two datasets.

**Parameters:**
- `x_vals` (array-like): X-axis values
- `y_vals` (array-like): Y-axis values  
- `title` (str): Plot title
- `out_file` (str/Path): Output file path

**Behavior:**
- Creates scatter plot of x_vals vs y_vals
- Adds diagonal line (y=x) for reference
- Automatically scales axes to show full data range
- Saves plot at 300 DPI resolution
- Closes all matplotlib figures after saving

**Returns:** None (saves plot file)

#### `color_tree(tree_file, fit_dict, out_file_phylo, center=False, null_color="white")`
**Location:** `ecoli_analysis/branch_fitness.py:74`

**Purpose:** Creates a phylogenetic tree visualization with branches colored by fitness values.

**Parameters:**
- `tree_file` (str/Path): Path to phylogenetic tree file
- `fit_dict` (dict): Mapping of branch names to fitness values
- `out_file_phylo` (str/Path): Output file path for tree image
- `center` (bool): Whether to center colormap around zero
- `null_color` (str): Color for branches with no data

**Workflow:**
1. Loads phylogenetic tree using `analysis.plot_phylo_standalone.loadTree()`
2. Creates continuous color function with 'coolwarm' colormap
3. Generates tree plot with colored branches and nodes
4. Adds colorbar legend
5. Saves high-resolution figure (12x30 inches)

**Returns:** None (saves tree visualization)

### Class Documentation

#### `PhyloDataFile`
**Location:** `analysis/arrayer.py:491`
**Inherits from:** `PhyloData`

**Purpose:** File-based phylogenetic data loader that can read numpy arrays and parameter dictionaries from disk.

##### `__init__(self, **kwargs)`
**Parameters:**
- `data_params_file` (str/Path, optional): Pickle file containing data parameters
- `array_file` (str/Path, optional): Numpy array file path
- `array` (numpy.ndarray, optional): Direct array data

**Behavior:**
- If `data_params_file` provided: loads parameters from pickle and merges with kwargs
- If `array_file` provided: loads numpy array using `np.load()` with `allow_pickle=True`
- If `array` provided: uses array directly
- Calls parent `PhyloData.__init__()` with loaded array and parameters

##### Key Attributes:
- `self.array`: Structured numpy array containing phylogenetic event data with fields:
  - `name`: Branch/event identifiers
  - `event`: Event type codes (1=birth, 2=sampling, 4=edge, etc.)
  - `time_step`: Time duration values
  - Additional parameter fields added dynamically

#### `PhyloData`
**Location:** `analysis/arrayer.py:398`
**Inherits from:** `PhyloArrayer`

**Purpose:** Core data structure for phylogenetic analysis containing structured arrays of phylogenetic events.

##### `__init__(self, array, **kwargs)`
**Parameters:**
- `array` (numpy.ndarray): Structured array of phylogenetic data
- `**kwargs`: Additional parameters set as object attributes

**Key Methods Used:**
- `getDataParams()`: Returns dictionary of object parameters for copying
- `returnCopy()`: Creates deep copy of the object
- `getSubArraySpecific(indices)`: Returns subset of data for specific indices

#### `ResultsObj`
**Location:** `analysis/do_model_fit.py:25`

**Purpose:** Analysis results management class that handles loading/saving of model fitting results, cross-validation indices, and analysis parameters.

##### `__init__(self, folder, verbose=True)`
**Parameters:**
- `folder` (str/Path): Analysis directory path
- `verbose` (bool): Enable verbose output

**Key Attributes:**
- `self.folder`: Analysis directory path
- `self.data`: PhyloData object
- `self.results_dict`: Dictionary of analysis results
- `self.params`: Analysis parameters
- `self.success`: Dictionary tracking successful loading of components

**Workflow in Command Context:**
- Created by `old_vs_new_single()` but not extensively used
- Primarily serves as data container for the analysis directory structure

### Data Flow Summary

**For the specific command:**

1. **Input Processing:**
   - CLI args: `command="single"`, `model="full-intercept-tvb"`, `combo_key="lamb=1_reg_type=l1_sigma=0.05_lr=5e-05_n_epochs=80000_fold-2"`
   - Maps to results directory: `data_new/analysis/three_sampling_intervals/full_model_birth_background_TV+birth_features+brownian_motion+sampling_background_TV+sampling_features/`

2. **Data Loading:**
   - JSON files: combo-specific results, model parameters, analysis parameters
   - CSV file: legacy random effects estimates
   - Numpy files: phylogenetic array data and parameters

3. **Analysis:**
   - Extract and compare brownian motion estimates
   - Generate comparative scatter plot
   - Create phylogenetic tree visualization with fitness coloring

4. **Outputs:**
   - Console: DataFrame printouts of comparison results
   - Files: Two PNG visualization files (scatter plot and phylogenetic tree)

This command specifically performs a single-case comparison between old and new model implementations for a specific hyperparameter combination, focusing on the brownian motion (random effects) component of the phylodynamic model.

---

## Summary

This session provided comprehensive analysis of the ST131Fitness codebase including:

1. **Repository Architecture Understanding**: Identified core components and their relationships
2. **Documentation Creation**: Generated CLAUDE.md for future development guidance
3. **Dependency Analysis**: Traced import dependencies and identified removable code (35+ unused files)
4. **Function Documentation**: Detailed documentation of all functions used in specific command execution

The analysis focused on a phylodynamics research codebase using TensorFlow for modeling E. coli fitness determinants, with Snakemake workflows for analysis pipeline orchestration.