"""
This script contains all necessary functions for the training pipeline.
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import rasterio
import pickle
import matplotlib.pyplot as plt  # to plot kmeans splits


def load_object(filename):
    """
    This function loads an object from a pickle file.

    Args:
        filename (str): Name of the file to be loaded, with extension, without path unless a subfolder is desired.

    Returns:
            obj (object): Loaded object.
    """
    filename = r"../Models/" + filename + ".pkl"
    with open(filename, "rb") as inp:
        obj = pickle.load(inp)
    return obj




#############################################
# Model comparisons:
#############################################


def model_comparison_table(model_list):
    """
    This function creates a table with the results of the models in the model_list.

    Args:
        model_list (list): List of models to be compared.

    Returns:
        table (pd.DataFrame): Table with the results of the models in the model_list.
    """
    table = pd.concat([i.get_results() for i in model_list])
    set_index = table.pop("Set")
    multiindex = [[model.name for model in model_list for _ in range(2)], set_index]
    table.index = multiindex
    table.index.names = ["Model", "Set"]
    return table


def model_comparison_plot(model_list, metric="RMSE"):
    """
    Creates a bar plot comparing the train and test metric values for a list of models.

    Args:
        model_list (list): A list of Model objects.

        metric (str): The metric to plot. 'RMSE' or 'R2'.
    """
    table = model_comparison_table(model_list).reset_index()
    model_names = table["Model"].unique()

    # Set the figure size and create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    train_color = "steelblue"  # Choose a color for the train set bars
    test_color = "darkorange"  # Choose a color for the test set bars
    for i, model in enumerate(model_names):
        # Get the train and test metric values for the current model
        train_val = table[(table["Model"] == model) & (table["Set"] == "Train")][metric].values[0]
        test_val = table[(table["Model"] == model) & (table["Set"] == "Test")][metric].values[0]
        # Get the train and test metric standard deviation values for the current model
        train_val_std = table[(table["Model"] == model) & (table["Set"] == "Train")][metric + "_std"].values[0]
        test_val_std = table[(table["Model"] == model) & (table["Set"] == "Test")][metric + "_std"].values[0]
        # Create the bar plot
        ax.bar(i - width / 2, train_val, width, yerr=train_val_std, label=None, capsize=10, color=train_color)
        ax.bar(i + width / 2, test_val, width, yerr=test_val_std, label=None, capsize=10, color=test_color)

    # Add the legend
    ax.legend(["Train", "Test"], loc="upper right")

    # Set the axis labels
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)

    # Show the plot
    plt.show()


#############################################
# Model predictions:
#############################################


def mean_predict(model, data):
    """
    This function calculates mean predictions, std of predictions and error of predictions.

    Args:
        model (object): model object

        data (dataframe): dataframe with data (should include all columns, both trained on and not).

    Returns:
        df_results (dataframe): dataframe with results (mean predictions, std and error of predictions).
    """
    columns = model.columns
    x_test = data[columns]
    y_test = data["opt_value"]
    all_predictions = []
    for i in range(len(model.cv_model_list)):
        predictions_one_model = model.cv_model_list[i].predict(x_test)
        all_predictions.append(predictions_one_model)
    mean_prediction = np.mean(all_predictions, axis=0)
    std_prediction = np.std(all_predictions, axis=0)
    error_prediction = np.abs(mean_prediction - y_test)
    df_results = pd.DataFrame(
        {
            "row": data["row"],
            "col": data["col"],
            "mean_prediction": mean_prediction,
            "std_prediction": std_prediction,
            "error_prediction": error_prediction,
        }
    )

    return df_results


#############################################
# Save to tif:
#############################################


def save_prediction_tif(df_predictions, metric, path_out):
    """
    Function to write predictions to .tif.

    Args:
        df_predictions (dataframe): dataframe with predictions

        metric_to_save (str): metric to save (mean, std or error)

        path_out (str): path to save .tif file, path should include file name and extension
    """
    # original matrix shape:
    nan_matrix = np.full((2663, 1462), np.nan)

    for row in tqdm(df_predictions.iterrows()):  # fix progress bar?
        row_index = int(row[1]["row"])
        col_index = int(row[1]["col"])
        pred_val = row[1][metric + "_prediction"]
        nan_matrix[row_index][col_index] = pred_val

    convert_to_tif(nan_matrix, path_out)
    return


def convert_to_tif(data, path_out):
    """
    Function to convert data to tif file.

    Args:
        data (numpy array or xarray): data to convert to tif

        path_out (str): path to save .tif file, path should include file name and extension

    Returns:
        .tif file
    """
    path_file_metadata = r"../Data/microwave-rs/mw_interpolated/2019-07-01_mw.tif"

    with rasterio.open(path_file_metadata) as src:
        kwargs1 = src.meta.copy()

    with rasterio.open(path_out, "w", **kwargs1) as dst:
        dst.write_band(1, data)  # numpy array or xarray
    return
