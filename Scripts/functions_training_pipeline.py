"""
This script contains all necessary functions for the training pipeline.
"""

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def import_data(date_from: str, date_to: str, df_path: str):
    """
    Imports data and merges into one dataframe.

    Args:
        date_from (format: 'yyyy-mm-dd'): Period starting date (included).

        date_to (format: 'yyyy-mm-dd'): Period end date (included).

        df_path: Path to folder with daily data files.

    Returns:
        pandas.DataFrame: Dataframe with all merged data.
    """

    date_range = pd.date_range(date_from, date_to)  # both ends included
    date_range = [str(day.date()) for day in date_range]
    df_list = []

    for melt_date in tqdm(date_range):
        try:  # bc some days are empty
            file = pd.read_parquet(
                df_path + "melt_" + melt_date + "_extended.parquet.gzip", index=False
            )
            df_list.append(file)  # list of df
        except:
            continue

    df = pd.concat(df_list, axis=0)  # concat af df to one
    del df_list
    return df


def data_prep(df, removeMaskedClouds=True):
    """
    Removes missing mw and opt data from dataframe.
    Used for training and testing, not predicting.

    Args:
        df (pandas.DataFrame): Full train/ test dataframe.

        removeMaskedClouds (bool): True for train and test data, removes masked data from opt data.
                                   False for predicting data, keeps masked opt data.

    Returns:
        pandas.DataFrame: The same dataframe with removed water (and masked data).
    """
    df = df[df["mw_value"] != -1]

    if removeMaskedClouds == True:
        df = df[df["opt_value"] != -1]

    # remove bare ice?
    # check if all aggregations are num (not nan)
    return df


def data_normalization(df, feature):
    """
    Normalizes data with min-max (linear) or Z-score normalization depending on feature.

    Args:
        df (pandas.DataFrame): Full train/ test dataframe.

        feature (string): Name of feature to be normalized.

    Returns:
        pandas.DataFrame: The same dataframe with normalized feature.
    """
    # TODO add / correct opt_value and elevation_data

    minmax_features = [
        "col",
        "row",
        "mean_3",
        "mean_9",
        "sum_5",
        "mw_value_yesterday",
        "mw_value_7_day_average",
        "hours_of_daylight",
        "slope_data",
        "aspect_data",
        "distance_to_margin",
    ]
    zscore_features = ["opt_value", "elevation_data"]

    if feature in minmax_features:
        if feature == "col":
            min, max = 0, 1461
        elif feature == "row":
            min, max = 0, 2662
        elif feature == "mean_3":
            min, max = 0, 1
        elif feature == "mean_9":
            min, max = 0, 1
        elif feature == "sum_5":
            min, max = 0, 25
        elif feature == "mw_value_yesterday":
            min, max = 0, 1
        elif feature == "mw_value_7_day_average":
            min, max = 0, 1
        elif feature == "hours_of_daylight":
            min, max = 0, 24
        elif feature == "slope_data":
            min, max = 0, 90
        elif feature == "aspect_data":
            min, max = -1, 1
        else:
            min, max = 1, 500

        df[feature] = (df[feature] - min) / (max - min)

    elif feature in zscore_features:
        scaler = StandardScaler()
        df[feature] = scaler.fit_transform(df[[feature]])
    else:
        print("Feature not found.")

    return df


def cross_validation(df, columns, train_func, n_splits=5, hyperparameters=None):
    """
    Cross-validation with TimeSeriesSplit.

    Args:
        df (pandas.DataFrame): Full train/ test dataframe.

        columns (list of strings): List of columns to be used in training.

        train_func (function): Custom defined function for training and evaluating model.
                                E.g.: model_decisionTree()

        n_splits (int): Number of cv splits.

        hyperparameters (dict, optional): Dictionary with hyperparameters for model.

    Returns:
        list: Two list with <n_splits> RMSE scores for train and test data.
    """

    df.sort_values(by=["date"], inplace=True)  # sort df by time
    X = df[columns]
    y = df[["opt_value"]]

    rmse_train_list = []
    rmse_test_list = []
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_index, test_index in tqdm(tscv.split(X)):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        y_predicted_train, y_predicted_test = train_func(
            X_train, y_train, X_test, y_test, hyperparameters
        )

        rmse_train = get_rmse(y_train, y_predicted_train)
        rmse_test = get_rmse(y_test, y_predicted_test)

        rmse_train_list.append(rmse_train)
        rmse_test_list.append(rmse_test)

    return rmse_train_list, rmse_test_list


def get_rmse(y_real, y_predicted):
    """
    Calculates RMSE score.

    Args:
        y_real (): real target values.

        y_predicted (): model predicted target values.

    Returns:
        float: RMSE score.
    """

    return np.sqrt(mean_squared_error(y_real, y_predicted))


def model_decisionTree(X_train, y_train, X_test, y_test, hyperparameters=None):
    """
    Trains model and predicts target values.

    Args:
        X_train (pandas.DataFrame): Dataframe with train data.

        y_train (pandas.DataFrame): Dataframe with train labels, one column.

        X_test (pandas.DataFrame): Dataframe with test data.

        y_test (pandas.DataFrame): Dataframe with test labels, one column.

        hyperparameters (dict, optional): Dictionary with model parameters.

    Returns:
        list: Two lists with predicted values for train and test set.
    """

    if hyperparameters:
        regressor = DecisionTreeRegressor(**hyperparameters)
    else:
        regressor = DecisionTreeRegressor(random_state=0)

    regressor.fit(X_train, y_train)
    y_predicted_train = regressor.predict(X_train)
    y_predicted_test = regressor.predict(X_test)

    return y_predicted_train, y_predicted_test


def model_meanBenchmark(y_train, y_test):
    """
    Creates predictions for mean benchmark.

    Args:
        y_train (pandas.DataFrame): Dataframe with train labels, one column.

        y_test (pandas.DataFrame): Dataframe with test labels, one column.

    Returns:
        list: Lists with predicted values for test set.
    """

    y_predicted = np.full((1, len(y_test)), y_train.mean())[0]

    return y_predicted


def model_mwBenchmark(X_test):
    """
    Creates predictions for microwave benchmark by comparing the mw and opt datasets directly.

    Args:
        X_test (pandas.DataFrame): Dataframe with test data.

    Returns:
        list: Lists with predicted values for test set.
    """

    y_predicted = X_test["mw_value"]

    return y_predicted


def hyperparameter_tune():
    # define grid (if grid)
    # do cv for each??? - maybe less splits?
    # define hyperparameters as a dictionary eg: dt_params = {'max_depth':7, 'criterion': 'squared_error'}
    return


def plot_cv_results():
    return
