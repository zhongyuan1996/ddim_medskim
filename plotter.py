# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:30:20 2023

@author: yfz5556
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

column_names = ['Kidney', 'COPD', 'Amnesia', 'Heart Failure', 'MIMIC']
row_names = ['with', 'wo']

data = [[[0.6674, 0.6107], [0.5604, 0.5534], [0.6377, 0.5536], [0.6040, 0.5498] ,[0.6120, 0.5943]],
        [[0.6701, 0.6533], [0.6057, 0.5870], [0.6449, 0.5804], [0.6191, 0.5680], [0.5719, 0.5856]],
        [[0.6967, 0.5781], [0.5914, 0.5356], [0.6365, 0.5604], [0.6270, 0.5390], [0.6120, 0.5989]],
        [[0.6120, 0.5465], [0.5118, 0.5170], [0.5424, 0.5151], [0.5289, 0.5374], [0.5892, 0.5470]],
        [[0.7012, 0.6952], [0.6165, 0.6050], [0.6182, 0.5968], [0.6220, 0.6074], [0.6126, 0.6242]],
        [[0.7256, 0.7320], [0.6020, 0.6384], [0.6860, 0.6816], [0.6772, 0.6824], [0.6607, 0.7007]],
        [[0.7172, 0.6957], [0.6148, 0.6052], [0.6356, 0.6344], [0.6258, 0.6018], [0.6198, 0.6152]],
        [[0.6649, 0.6807], [0.5701, 0.5486], [0.5999, 0.5646], [0.6033, 0.6018], [0.6388, 0.6545]],
        [[0.7272, 0.6940], [0.6548, 0.6862], [0.6377, 0.6319], [0.6346, 0.6466], [0.6113, 0.6193]],
        [[0.7616, 0.7554], [0.6840, 0.6846], [0.7152, 0.7080], [0.6696, 0.6756], [0.6116, 0.6044]],
        [[0.7474, 0.7631], [0.7049, 0.6932], [0.7413, 0.7309], [0.7178, 0.7238], [0.5850, 0.6220]]]


# Define model and dataset names.
model_names = ['LSTM', 'Dipole', 'Retain', 'SAnD', 'Adacare', 'LSAN', 'RetainEX', 'Timeline', 'T-LSTM', 'HiTANet', 'MedSkim']
dataset_names = ['Kidney', 'COPD', 'Amnesia', 'Heart Failure', 'MIMIC']
methods = ['w', 'wo']

# Define a list to store each row of the DataFrame.
rows = []

# Iterate over each model.
for i, model in enumerate(model_names):
    # Iterate over each dataset.
    for j, dataset in enumerate(dataset_names):
        # Iterate over each method.
        for k, method in enumerate(methods):
            # Append a new row to the list.
            rows.append([model, dataset, method, data[i][j][k]])

# Create a DataFrame from the list of rows.
df = pd.DataFrame(rows, columns=['Model', 'Dataset', 'Method', 'Output'])

print(df)
# Set the aesthetic theme
sns.set_theme(style="whitegrid")

plt.rcParams['font.family'] = 'Times New Roman'
# Increase the font size and line width
sns.set_context("notebook", font_scale=3, rc={"lines.linewidth": 2.5})

# Define the models to exclude
exclude_models = []

# Create a function to format the y-ticks
formatter = ticker.FuncFormatter(lambda x, pos: '{:.0f}%'.format(x * 100))

# Modify the DataFrame directly to change the names of the datasets and 'wo' to 'w/o'
df['Dataset'] = df['Dataset'].replace({'Kidney': 'Ki', 'COPD': 'CO', 'Amnesia': 'Am', 'Heart Failure': 'HF', 'MIMIC': 'MM'})
df['Method'] = df['Method'].replace({'wo': 'w/o'})

max_outputs = df.groupby('Model')['Output'].max()
# Set the color palette to red and blue
palette = [ "#e74c3c", "#3498db"]

# Iterate over each unique model
for i, model in enumerate(df['Model'].unique()):
    if model not in exclude_models:
        # Filter the DataFrame for the current model
        df_filtered = df[df['Model'] == model]
        max_output = max_outputs[model]
        # Create a bar plot for the filtered DataFrame
        g = sns.catplot(data=df_filtered, kind="bar", x='Dataset', y='Output', hue='Method', palette=palette, alpha=1, height=6)
        g.fig.set_size_inches(10, 10)

        # Set the title
        g.fig.suptitle(model, y = 0.88)

        # Remove the y-axis label
        g.set_ylabels("")

        # Format the y-ticks to display two decimal places
        g.ax.yaxis.set_major_formatter(formatter)


        # Set the y-axis limits
        g.set(ylim=(0.4, max_output + 0.01))  # Adding 0.1 to max_output for a bit of margin at the top

        # Only draw the legend for the first plot
        # if i != 0:
        g._legend.remove()
        # else:
        #     # Move the legend to the left of the plot for the first plot
        #     g._legend.set_bbox_to_anchor((0.9, 0.9))

        # Show the plot
        plt.tight_layout()
        plt.savefig('figure/'+model+'.png', dpi=600)
        plt.show()