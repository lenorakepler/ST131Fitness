
"""
Gemini, thinking mode:
I have a table with various runs with various hyperparameters, and estimated values of each variable. 
I have a separate json file that categorizes the variables. I would like to make a facet plot in python 
where each row is represents a combination of ["model_key", "lamb", "sigma"] and where each column is 
one of the variable categories in the json file. I would like to have a density plot for each row and 
column that shows the distribution of estimated values for that ["model_key", "lamb", "sigma"] and variable 
category. I am attaching the table as a csv file and the categories as a json file.
--> Can you also standardize y axis scales across model runs
"""

import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

def plot_densities(df, var_names, out_file):

# Prep
var_to_cat = {}
for cat, vars_list in var_names.items():
    for var in vars_list:
        var_to_cat[var] = cat

relevant_vars = list(set(var_to_cat.keys()).intersection(set(df.columns)))
id_vars = ["model_key", "lamb", "sigma", "fold"]
df_long = df.melt(id_vars=id_vars, value_vars=relevant_vars, var_name="variable", value_name="estimate")
df_long['category'] = df_long['variable'].map(var_to_cat)

unique_models = df_long['model_key'].unique()
model_map = {m: f"M{i}" for i, m in enumerate(unique_models)}
df_long['model_short'] = df_long['model_key'].map(model_map)
df_long['row_label'] = df_long.apply(lambda x: f"{x['model_short']} | L={x['lamb']} | S={x['sigma']}", axis=1)

df_clean = df_long.dropna(subset=['estimate'])

row_tuples = df_clean[['model_short', 'lamb', 'sigma', 'row_label']].drop_duplicates()
row_tuples = row_tuples.sort_values(by=['model_short', 'lamb', 'sigma'])
row_order = row_tuples['row_label'].tolist()
col_order = list(var_names.keys())

# Plot
# Use hue="category" to color each column differently.
# palette="deep" or similar.
# legend=False because columns distinguish categories.
g = sns.displot(
    data=df_clean,
    x="estimate",
    row="row_label",
    col="category",
    hue="category",
    kind="kde",
    fill=True,
    height=2.0, # Reduced height slightly to make it more manageable
    aspect=1.5,
    row_order=row_order,
    col_order=col_order,
    palette="tab10",
    facet_kws={'sharex': 'col', 'sharey': 'col'},
    legend=False
)

g.set_titles(row_template="{row_name}", col_template="{col_name}")
plt.subplots_adjust(top=0.98, hspace=0.3) # hspace to make room for lines if needed, or just separation

# Draw separator lines
# Find indices where model changes
models_in_order = row_tuples['model_short'].tolist()
separator_indices = []
for i in range(len(models_in_order) - 1):
    if models_in_order[i] != models_in_order[i+1]:
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
    
    line = lines.Line2D([x_start, x_end], [y_line, y_line], transform=fig.transFigure, color='black', linewidth=2, linestyle='--')
    fig.add_artist(line)

g.savefig("facet_plot_final.png")