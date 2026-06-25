
"""
Gemini, thinking mode:
I have a table with various runs with various hyperparameters, and estimated values of each variable. 
I have a separate json file that categorizes the variables. I would like to make a facet plot in python 
where each row is represents a combination of ["model_key", "lamb", "sigma"] and where each column is 
one of the variable categories in the json file. I would like to have a density plot for each row and 
column that shows the distribution of estimated values for that ["model_key", "lamb", "sigma"] and variable 
category. I am attaching the table as a csv file and the categories as a json file.
--> Can you also standardize y axis scales across model runs
--> Can you add a horizontal line to separate the different models? And can you make each variable category a different color? 

+ good amount of revision
"""

import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import lines
from natsort import natsorted

def plot_densities(df, var_names, out_dir, group_vars=["model", "lamb"], compare_var="sigma", model_display_names="", pre_avg=False, query=""):
    save_str = f"fitness_category_density_group-{'-'.join(group_vars)}_compare-{compare_var}{'_pre_avg' if pre_avg else ''}{'_query-' + query if query else ''}"

    # reverse dictionary of categories with variables
    var_to_cat = {}
    for cat, var_list in var_names.items():
        for var in var_list:
            var_to_cat[var] = cat

    df = df.rename(columns={"model_key": "model"})

    if model_display_names:
        df["model"] = df["model"].apply(lambda v: model_display_names[v])

    groupby_vars = group_vars + [compare_var]
    id_vars = ["model", "lamb", "sigma", "lr", "n_epochs", "fold"]
    avg_over_vars = [v for v in id_vars if v not in groupby_vars]
    
    if query:
        pre_avg = False

    # Average dataframe over extraneous variables (e.g. n_epochs, fold)
    # ==============================================================================
    if pre_avg:
        # group by wanted variables and get mean of each variable
        dfm = df.groupby(groupby_vars).mean(numeric_only=True).reset_index()

    else:
        dfm = df

    # Create the longform dataframe that will be input into sns.displot
    # ==============================================================================
    # make data long
    dfl = dfm.melt(id_vars=id_vars, var_name="variable", value_name="estimate")

    # set row labels
    def get_row_label(row):
        return ' | '.join([f"{var[0].upper()}={row[var]}" for var in group_vars])

    dfl['row_label'] = dfl.apply(lambda x: get_row_label(x), axis=1)

    # set row category
    dfl["category"] = dfl["variable"].map(var_to_cat)

    # drop rows with nan estimates (e.g. where model wasn't estimating)
    dfl = dfl.dropna(subset="estimate")
    dfl = dfl.query(query)
    dfl.to_csv(out_dir / f"{save_str}.csv")

    # Make this seaborn displot
    # ==============================================================================
    order_df = dfl[group_vars + ['row_label']].drop_duplicates().sort_values(group_vars)

    # Set up the plot with standardized y-axis across model runs (rows) for each category (column)
    # sharey='col' means: within each column, all rows share the same y-axis.
    g = sns.displot(
        data=dfl,
        x="estimate",
        row="row_label",
        col="category",
        hue=compare_var,
        kind="hist",
        rug=True,
        height=2.5,
        aspect=1.5,
        row_order=order_df["row_label"],
        # col_order=col_order,
        palette=sns.color_palette("magma", as_cmap=False, n_colors=len(df[compare_var].unique())),
        facet_kws={'sharex': 'col', 'sharey': False},
        # facet_kws={'sharex': 'col', 'sharey': False, 'margin_titles': True},
        common_norm=False,
        element="step",
        fill=False,
        common_bins=True,
        # lw=0,
        # warn_singular=False,
    )

    g.tick_params(axis='x', labelbottom=True)
    g.tick_params(axis='y', labelleft=False)

    # Adjust titles
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # Add lines of separation between models
    # ==============================================================================
    # Draw separator lines
    # Find indices where model changes
    separator_indices = []
    for i in range(len(order_df) - 1):
        if order_df.iloc[i]["model"] != order_df.iloc[i+1]["model"]:
            separator_indices.append(i)

    # Get figure and axes
    fig = g.fig
    axes = g.axes

    # We want to draw a line across the entire figure width at the y-position between rows
    # The grid is regular. We can use the axes positions.
    # For each separator index `idx`, the line is between row `idx` and `row `idx+1`.
    # We can check the y0 of row `idx` and y1 of row `idx+1`?
    # No, row indices go top to bottom. Row 0 is top.
    # So row `idx` is above row `idx+1`.
    # We need the bottom of row `idx` and top of row `idx+1`.

    for idx in separator_indices:
        # Get the axes of the last column for row `idx` and row `idx+1` to be safe, or just first column.
        # We need the y position.
        # Let's take the first column ax.
        ax_upper = axes[idx, 0]
        ax_lower = axes[idx+1, 0]
        
        # Get bounding boxes in figure coordinates
        bbox_upper = ax_upper.get_position()
        bbox_lower = ax_lower.get_position()
        
        # Calculate mid y
        y_line = (bbox_upper.y0 + bbox_lower.y1) / 2
        
        # Draw line
        # x range: from left of first column to right of last column
        ax_first_col = axes[idx, 0]
        ax_last_col = axes[idx, -1]
        
        x_start = ax_first_col.get_position().x0
        x_end = ax_last_col.get_position().x1
        
        x_tot = x_end - x_start
        x_start -= (x_tot * .05)
        x_end += (x_tot * .05)
        line = lines.Line2D([x_start, x_end], [y_line, y_line], transform=fig.transFigure, color='black', linewidth=2, linestyle='--')
        fig.add_artist(line)

    # plt.tight_layout()
    g.savefig(out_dir / f"{save_str}.png", dpi=300)

def plot_densities_2(df, var_names, out_dir, group_vars=["model", "lamb"], compare_var="sigma", model_display_names="", pre_avg=False):
    # reverse dictionary of categories with variables
    var_to_cat = {}
    for cat, var_list in var_names.items():
        for var in var_list:
            var_to_cat[var] = cat

    df = df.rename(columns={"model_key": "model"})
    if model_display_names:
        df["model"] = df["model"].apply(lambda v: model_display_names[v])

    groupby_vars = group_vars + [compare_var]
    id_vars = ["model", "lamb", "sigma", "reg_type", "lr", "n_epochs", "fold"]
    avg_over_vars = [v for v in id_vars if v not in groupby_vars]
    
    # drop columns that won't be used in grouping
    df = df.drop(columns=avg_over_vars)
    
    # Average dataframe over extraneous variables (e.g. n_epochs, fold)
    # ==============================================================================
    if pre_avg:
        # group by wanted variables and get mean of each variable
        dfm = df.groupby(groupby_vars).agg("mean").reset_index()

    else:
        dfm = df

    # Create the longform dataframe that will be input into sns.displot
    # ==============================================================================
    # make data long
    dfl = dfm.melt(id_vars=groupby_vars, var_name="variable", value_name="estimate")

    # set row labels
    def get_row_label(row):
        return ' | '.join([f"{var[0].upper()}={row[var]}" for var in group_vars])

    dfl['row_label'] = dfl.apply(lambda x: get_row_label(x), axis=1)

    # set row category
    dfl["category"] = dfl["variable"].map(var_to_cat)

    # drop rows with nan estimates (e.g. where model wasn't estimating)
    dfl = dfl.dropna(subset="estimate")

    # Make this seaborn displot
    # ==============================================================================
    order_df = dfl[group_vars + ['row_label']].drop_duplicates().sort_values(group_vars)

    # Set up the plot with standardized y-axis across model runs (rows) for each category (column)
    # sharey='col' means: within each column, all rows share the same y-axis.
    g = sns.displot(
        data=dfl,
        x="estimate",
        row="row_label",
        col="category",
        hue=compare_var,
        kind="kde",
        rug=True,
        height=2.5,
        aspect=1.5,
        row_order=order_df["row_label"],
        # col_order=col_order,
        palette=sns.color_palette("magma", as_cmap=False, n_colors=len(df[compare_var].unique())),
        facet_kws={'sharex': 'col', 'sharey': False},
        # facet_kws={'sharex': 'col', 'sharey': False, 'margin_titles': True},
        common_bins=False,
        common_norm=False,
        warn_singular=False,
    )

    g.tick_params(axis='x', labelbottom=True)
    g.tick_params(axis='y', labelleft=False)

    # Adjust titles
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # Add lines of separation between models
    # ==============================================================================
    # Draw separator lines
    # Find indices where model changes
    separator_indices = []
    for i in range(len(order_df) - 1):
        if order_df.iloc[i]["model"] != order_df.iloc[i+1]["model"]:
            separator_indices.append(i)

    # Get figure and axes
    fig = g.fig
    axes = g.axes

    # We want to draw a line across the entire figure width at the y-position between rows
    # The grid is regular. We can use the axes positions.
    # For each separator index `idx`, the line is between row `idx` and `row `idx+1`.
    # We can check the y0 of row `idx` and y1 of row `idx+1`?
    # No, row indices go top to bottom. Row 0 is top.
    # So row `idx` is above row `idx+1`.
    # We need the bottom of row `idx` and top of row `idx+1`.

    for idx in separator_indices:
        # Get the axes of the last column for row `idx` and row `idx+1` to be safe, or just first column.
        # We need the y position.
        # Let's take the first column ax.
        ax_upper = axes[idx, 0]
        ax_lower = axes[idx+1, 0]
        
        # Get bounding boxes in figure coordinates
        bbox_upper = ax_upper.get_position()
        bbox_lower = ax_lower.get_position()
        
        # Calculate mid y
        y_line = (bbox_upper.y0 + bbox_lower.y1) / 2
        
        # Draw line
        # x range: from left of first column to right of last column
        ax_first_col = axes[idx, 0]
        ax_last_col = axes[idx, -1]
        
        x_start = ax_first_col.get_position().x0
        x_end = ax_last_col.get_position().x1
        
        x_tot = x_end - x_start
        x_start -= (x_tot * .05)
        x_end += (x_tot * .05)
        line = lines.Line2D([x_start, x_end], [y_line, y_line], transform=fig.transFigure, color='black', linewidth=2, linestyle='--')
        fig.add_artist(line)

    # plt.tight_layout()
    g.savefig(out_dir / f"fitness_category_density_group-{'-'.join(group_vars)}_compare-{compare_var}{'_pre_avg' if pre_avg else ''}.png", dpi=300)
   
def plot_densities_orig(df, var_names, out_dir, group_vars=["model_short", "lamb"], compare_var="sigma"):
    # Prepare reverse mapping for categories
    var_to_cat = {}
    for cat, vars_list in var_names.items():
        for var in vars_list:
            var_to_cat[var] = cat

    # Filter relevant variables
    relevant_vars = list(set(var_to_cat.keys()).intersection(set(df.columns)))

    # Melt
    id_vars = ["model_key", "lamb", "sigma", "reg_type", "lr", "n_epochs", "fold"]
    df_long = df.melt(id_vars=id_vars, value_vars=relevant_vars, var_name="variable", value_name="estimate")

    # Add category
    df_long['category'] = df_long['variable'].map(var_to_cat)

    # Create a row label
    unique_models = df_long['model_key'].unique()
    model_map = {m: f"M{i}" for i, m in enumerate(unique_models)}
    df_long['model_short'] = df_long['model_key'].map(model_map)

    def get_row_label(row):
        return ' | '.join([f"{var[0]}={row[var]}" for var in group_vars])

    df_long['row_label'] = df_long.apply(lambda x: get_row_label(x), axis=1)

    # Drop NaNs
    df_clean = df_long.dropna(subset=['estimate'])

    # Define sort order for row labels
    id_vals_df = df_clean[group_vars + ['row_label']].drop_duplicates().sort_values(group_vars)
    row_order = id_vals_df['row_label'].tolist()
    col_order = list(var_names.keys())



    # Set up the plot with standardized y-axis across model runs (rows) for each category (column)
    # sharey='col' means: within each column, all rows share the same y-axis.
    g = sns.displot(
        data=df_clean,
        x="estimate",
        row="row_label",
        col="category",
        hue=compare_var,
        hue_order=sorted(df_clean[compare_var].unique().tolist()),
        kind="kde",
        # fill=True,
        height=2.5,
        aspect=1.5,
        row_order=row_order,
        col_order=col_order,
        palette=sns.color_palette("magma", as_cmap=False),
        facet_kws={'sharex': 'col', 'sharey': False},
        common_norm=False,
        # legend=False
    )

    g.tick_params(axis='x', labelbottom=True)

    # Adjust titles
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    plt.subplots_adjust(top=0.98, hspace=0.3) # hspace to make room for lines if needed, or just separation
    
    # Draw separator lines
    # Find indices where model changes
    separator_indices = []
    for i in range(len(row_order) - 1):
        if row_order[i][0:4] != row_order[i+1][0:4]:
            separator_indices.append(i)

    print("Separator indices (0-based row index after which to draw line):", separator_indices)

    # Get figure and axes
    fig = g.fig
    axes = g.axes

    # We want to draw a line across the entire figure width at the y-position between rows
    # The grid is regular. We can use the axes positions.
    # For each separator index `idx`, the line is between row `idx` and `row `idx+1`.
    # We can check the y0 of row `idx` and y1 of row `idx+1`?
    # No, row indices go top to bottom. Row 0 is top.
    # So row `idx` is above row `idx+1`.
    # We need the bottom of row `idx` and top of row `idx+1`.

    for idx in separator_indices:
        # Get the axes of the last column for row `idx` and row `idx+1` to be safe, or just first column.
        # We need the y position.
        # Let's take the first column ax.
        ax_upper = axes[idx, 0]
        ax_lower = axes[idx+1, 0]
        
        # Get bounding boxes in figure coordinates
        bbox_upper = ax_upper.get_position()
        bbox_lower = ax_lower.get_position()
        
        # Calculate mid y
        y_line = (bbox_upper.y0 + bbox_lower.y1) / 2
        
        # Draw line
        # x range: from left of first column to right of last column
        ax_first_col = axes[idx, 0]
        ax_last_col = axes[idx, -1]
        
        x_start = ax_first_col.get_position().x0
        x_end = ax_last_col.get_position().x1
        
        x_tot = x_end - x_start
        x_start -= (x_tot * .1)
        x_end += (x_tot * .1)
        line = lines.Line2D([x_start, x_end], [y_line, y_line], transform=fig.transFigure, color='black', linewidth=2, linestyle='--')
        fig.add_artist(line)

    g.savefig(out_dir / f"fitness_category_density_group-{'-'.join(group_vars)}_compare-{compare_var}.png", dpi=300)

