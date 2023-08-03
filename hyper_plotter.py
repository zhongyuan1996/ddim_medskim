import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os
import matplotlib.ticker as ticker
from matplotlib import cm

def save_plot(title):
    # Format the title to remove spaces and special characters
    filename = ''.join(e for e in title if e.isalnum())

    # Save the plot
    plt.savefig(f"{filename}.png")
    plt.close()

# # Path to CSV files
# path_to_csv = "C:/Users/yfz5556/PycharmProjects/ddim_medskim/saved_models"
#
# # Read all csv files in the specified directory
# all_files = glob.glob(os.path.join(path_to_csv, "*.csv"))
#
# dataframes = []
#
# for file in all_files:
#     # Parse filename to get LD, LS, dataset, and seed
#     filename = os.path.basename(file).split('_')
#
#     # Extract LD and LS based on their known positions and formats
#     LD = float(filename[1][2:])
#     LS = float(filename[2][2:])
#
#     # Extract seed from the last part of the filename
#     seed = filename[-1].split('.')[0]
#
#     # Ignore the first element (which is 'medDiff') and assume that everything else is part of the dataset name
#     dataset_name = '_'.join(filename[3:-1])
#
#     df = pd.read_csv(file, header=None, names=['PR', 'F1', 'Kappa'])
#     df['LD'] = LD
#     df['LS'] = LS
#     df['Dataset'] = dataset_name
#     df['Seed'] = seed
#
#     dataframes.append(df)
#
# # Combine all dataframes into a single DataFrame
# df = pd.concat(dataframes, ignore_index=True)

# Path where the "hyper_data.csv" file will be saved
save_path = "C:/Users/yfz5556/PycharmProjects/ddim_medskim/"
#
# # Existing data
# LD = np.array([0.1, 0.25, 0.5, 0.75, 1.0])  # Add 1.0 to the existing values
# LS = np.array([0.1, 0.25, 0.5, 0.75, 1.0])  # Add 1.0 to the existing values
#
# PR = np.array([[0.76, 0.7358, 0.7834, 0.7281, 0],  # Add placeholder for the new data point
#                 [0.7609, 0.7385, 0.7247, 0.7421, 0],
#                 [0.7729, 0.7717, 0.7715, 0.7695, 0],
#                 [0.77, 0.7749, 0.7759, 0.7663, 0],
#                 [0, 0, 0, 0, 0]])
#
# F1 = np.array([[0.6972, 0.6873, 0.7112, 0.6713, 0],
#                 [0.6946, 0.6925, 0.6935, 0.6941, 0],
#                 [0.6942, 0.701, 0.701, 0.7051, 0],
#                 [0.6942, 0.673, 0.6989, 0.7065, 0],
#                 [0, 0, 0, 0, 0]])
#
# Kappa = np.array([[0.5776, 0.5781, 0.6109, 0.5599, 0],
#                 [0.5713, 0.5679, 0.5752, 0.5787, 0],
#                 [0.5799, 0.5933, 0.5966, 0.601, 0],
#                 [0.5873, 0.5653, 0.6075, 0.602, 0],
#                 [0, 0, 0, 0, 0]])
#
# # Create a DataFrame for the new data
# new_data = []
#
# for i in range(LD.shape[0]):
#     for j in range(LS.shape[0]):
#         new_data.append({
#             'LD': LD[i],
#             'LS': LS[j],
#             'PR': PR[i, j],
#             'F1': F1[i, j],
#             'Kappa': Kappa[i, j],
#             'Dataset': 'Kidney',
#             'Seed': '1234'  # Assuming you also want a placeholder for Seed
#         })
#
# df_new = pd.DataFrame(new_data)
#
# # Append new data to existing DataFrame
# df = pd.concat([df, df_new], ignore_index=True)
#
# # Save the updated DataFrame into a CSV file
# df.to_csv(os.path.join(save_path, "hyper_data.csv"), index=False)

##########################################################################################################################

# Increase default font size
plt.rcParams['font.size'] = 18

# Load the combined data from the csv file
df = pd.read_csv(os.path.join(save_path, "hyper_data.csv"))

# Get unique datasets
datasets = df['Dataset'].unique()

# For each unique dataset
for dataset in datasets:
    # Get data for the current dataset
    df_dataset = df[df['Dataset'] == dataset]

    fig, axs = plt.subplots(1, 3, figsize=(21, 7), subplot_kw={'projection': '3d'})

    # Set overall title and font size
    if dataset == 'Heart_failure':
        fig.suptitle('Heart Failure', fontsize=20)
    elif dataset == 'mimic':
        fig.suptitle('MIMIC', fontsize=20)
    else:
        fig.suptitle(dataset, fontsize=20)

    metrics = ['PR', 'F1', 'Kappa']
    titles = ['PR-AUC', 'F1', 'Kappa']

    for ax, metric, title in zip(axs, metrics, titles):

        X = df_dataset['LD']
        Y = df_dataset['LS']
        Z = df_dataset[metric]

        # Interpolate unstructured D-dimensional data with linear method
        grid_x, grid_y = np.mgrid[0.1:1.0:0.05, 0.1:1.0:0.05]
        grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
        # surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='bwr', edgecolor='none')

        # Change z axis to 2 decimal places
        ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        # Create colorbar
        m = cm.ScalarMappable(cmap=cm.viridis)
        m.set_array(grid_z)
        m.set_clim(np.nanmin(grid_z), np.nanmax(grid_z))
        cbar = plt.colorbar(m, ax=ax, shrink=0.5, aspect=10)
        cbar.formatter = ticker.FormatStrFormatter('%.2f')
        cbar.update_ticks()

        # Set x and y ticks
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])

        # Set tick label size
        ax.tick_params(axis='both', labelsize=16)

        # Set axis label size and position
        ax.set_xlabel('LD', fontsize=20, labelpad=10)
        ax.set_ylabel('LS', fontsize=20, labelpad=10)
        # ax.set_zlabel(metric, fontsize=18, labelpad=10)

        # Set titles
        ax.set_title(title, fontsize=20)

        # Adjust distance between axis labels and ticks
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20

        ax.zaxis.set_major_locator(plt.MaxNLocator(5))
        ax.view_init(elev=25, azim=-130)

    fig.savefig(os.path.join(save_path, dataset + '_hyper.png'))