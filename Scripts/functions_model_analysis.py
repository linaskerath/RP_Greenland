"""
This script contains all necessary functions for the training pipeline.
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import rasterio
import pickle
import matplotlib.pyplot as plt

# for feature importance:
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor


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

#############################################
# Feature importance:
#############################################

def feature_importance_dict(model, columns):
    """
    Function to plot feature importance.
    """
    # columns = model.columns
    # model = model.cv_model_list[0]

    if isinstance(model, DecisionTreeRegressor):
        feature_importance = model.feature_importances_
    elif isinstance(model, RandomForestRegressor):
        feature_importance = model.feature_importances_
    elif isinstance(model, GradientBoostingRegressor):
        feature_importance = model.feature_importances_
    elif isinstance(model, LinearRegression):
        feature_importance = model.coef_
    elif isinstance(model, Ridge):
        feature_importance = np.abs(model.coef_)
    elif isinstance(model, Lasso):
        feature_importance = np.abs(model.coef_)
    elif isinstance(model, ElasticNet):
        feature_importance = np.abs(model.coef_)
    else:
        print("model not supported")
        assert False
    feature_importance_dict = dict(zip(columns, feature_importance))
    return feature_importance_dict


def plot_feature_importance(model):
    """ Plot mean feature importance over 5 cv models with std.
    """
    feature_importance_df = []
    for mod in model.cv_model_list:
        feature_importance = feature_importance_dict(mod, model.columns)
        if len(feature_importance_df) == 0:
            feature_importance_df = pd.DataFrame(feature_importance, index=[0])
        else:
            feature_importance_df = pd.concat([feature_importance_df, pd.DataFrame([feature_importance])], ignore_index=True)

    # sort features by mean importance in descending order by absolute value
    mean_importances = feature_importance_df.mean()
    mean_importances_abs = np.abs(mean_importances)
    sorted_index = mean_importances_abs.sort_values(ascending=False).index
    mean_importances = mean_importances[sorted_index]
    feature_names = mean_importances.index

    # assign colors to positive and negative features
    colors = ['red' if imp < 0 else 'green' for imp in mean_importances]

    # plot mean feature importance as bar plot with std
    fig, ax = plt.subplots()
    ax.bar(feature_names, mean_importances, yerr=feature_importance_df[sorted_index].std(), capsize=5, color=colors)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Mean Importance')
    ax.set_title('Feature Importance')
    ax.tick_params(axis='x', rotation=90)

    plt.show()


    ############
    # to get model params:
    # model.cv_model_list[0].get_params()