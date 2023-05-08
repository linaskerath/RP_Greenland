"""
This script contains all necessary functions for the training pipeline.
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
#import polars as pl
#import pyarrow
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt  # to plot kmeans splits
from sklearn.model_selection import ParameterGrid


#############################################
# Data preparation functions
#############################################


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
    df = pd.DataFrame()

    for melt_date in tqdm(date_range):
        # print(melt_date)
        try:  # bc some days are empty
            file = pd.read_parquet(df_path + "melt_" + melt_date + "_extended.parquet.gzip")
            # drop columns row, col, date as not needed
            file = file.drop(columns=["row", "col", "date"], axis=1)
            # remove masked data
            file = remove_data(file, removeMaskedClouds=True, removeNoMelt=True)

            df = pd.concat([df, file], axis=0)
        except:
            continue

    # df = df.to_pandas()

    return df


def remove_data(df, removeMaskedClouds=True, removeNoMelt=True):
    """
    Removes missing/masked mw and opt data from dataframe.
    Used for training and testing, not predicting.

    Args:
        df (pandas.DataFrame): Full train/ test dataframe.

        removeMaskedClouds (bool): True for train and test data, removes masked data from opt data.
                                   False for predicting data, keeps masked opt data.

        removeNoMelt (bool): True for train and test data, removes non-melt areas from mw data.
                             False for predicting data, keeps non-melt areas.
    Returns:
        pandas.DataFrame: The same dataframe with removed water (and masked data).
    """

    if removeMaskedClouds == True:
        df = df[df["opt_value"] != -1]

    if removeNoMelt == True:
        melt = pd.read_parquet(r"/mnt/volume/AWS_Data/Data/split_indexes/noMelt_indexes.parquet")
        df = df.merge(melt, how="left", on=["y", "x"])
        df = df[df["melt"] == 1]
        df.pop("melt")
    return df


def data_normalization(df):
    """
    Normalizes data with min-max (linear) or Z-score normalization depending on feature.

    Args:
        df (pandas.DataFrame): Full train/ test dataframe.

    Returns:
        pandas.DataFrame: The same dataframe with normalized features.
    """
    features = df.columns

    minmax_features = [
        # "col",
        # "row",
        "x",
        "y",
        "mean_3",
        "mean_9",
        "sum_5",
        "mw_value_7_day_average",
        "hours_of_daylight",
        "slope_data",
        "aspect_data",
        "elevation_data",
        "distance_to_margin",
    ]
    log_feature = ["opt_value"]

    for feature in features:
        if feature in minmax_features:
            # if feature == "col":
            #     min, max = 0, 1461
            # elif feature == "row":
            #     min, max = 0, 2662
            if feature == "x":
                min, max = -636500.0, 824500.0
            elif feature == "y":
                min, max = -3324500.0, -662500.0
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
            elif feature == "elevation_data":
                min, max = 0, 3694
            else:
                min, max = 1, 500

            df[feature] = (df[feature] - min) / (max - min)

        elif feature in log_feature:
            # add a constant to avoid log(0)
            df[feature] = np.log(1 + df[feature])
        else:
            print(f"Not applicable for feature'{feature}'.")

    return df


#############################################
# Benchmark functions
#############################################


# def model_meanBenchmark(y_train, y_test):
#     """
#     Creates predictions for mean benchmark.

#     Args:
#         y_train (pandas.DataFrame): Dataframe with train labels, one column.

#         y_test (pandas.DataFrame): Dataframe with test labels, one column.

#     Returns:
#         list: Lists with predicted values for test set.
#     """
#     y_predicted = np.full((1, len(y_test)), y_train.mean())[0]

#     return y_predicted


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


#############################################
# Model training CV
#############################################


class Model:
    """
    This class contains models.
    After training it also contains performance scores and the hyperparameters used to train it.
    """

    def __init__(self, model, name):
        self.model = model
        self.hyperparameters = []  # list of dictionaries with hyperparameters
        self.name = name

    def create_hyperparameter_grid(self, hyperparameters):
        """
        This function creates a grid of hyperparameters.

        Args:
            hyperparameters (dict): Dictionary with hyperparameters.

        Returns:
            list: List of dictionaries with hyperparameters.
        """
        return list(ParameterGrid(hyperparameters))

    def __kmeans_split(self, df, split_variable_name, plot=False, verbose=False):
        """
        This function splits the data into 5 areas based on the kmeans algorithm.

        Args:
            df (pandas.DataFrame): Dataframe with data.

            split_variable_name (str): name for the column with the kmeans split. Eg. 'inner_area' or 'outer_area' loop.

            plot (bool): If True, plots the kmeans split.

            verbose (bool): If True, prints the number of observations in each split.

        Returns:
            pandas.DataFrame: Dataframe with added column with kmeans split.
        """
        kmeans = KMeans(n_clusters=5, n_init="auto").fit(df[["x", "y"]])  #  random_state=0,
        df[split_variable_name] = kmeans.labels_

        if verbose == True:
            print(df[split_variable_name].value_counts())

        if plot == True:
            plt.scatter(df["x"], df["y"], c=df[split_variable_name], edgecolor="none", s=0.05)
            plt.show()
        return df

    def __train_test_split(self, df, columns, split_variable_name, split_index):
        """
        This function splits the data into train and test set.

        Args:
            df (pandas.DataFrame): Dataframe with data.

            columns (list): List with column names to be used in the model.

            split_variable_name (str): name of column with kmeans split.

            split_index (int): Index of the split (loop).

        Returns:
            pandas.DataFrame: Dataframe with added column with kmeans split.
        """
        train = df[df[split_variable_name] != split_index]
        test = df[df[split_variable_name] == split_index]
        train_X = train[columns]
        train_y = train["opt_value"].values.ravel()
        test_X = test[columns]
        test_y = test["opt_value"].values.ravel()
        return train_X, train_y, test_X, test_y

    def __tune_hyperparameters(self, df, columns, split_variable_name):
        """
        This function performs hyperparameter tuning in (inner loop of nested cross-validation).

        Args:
            df (pandas.DataFrame): Dataframe with data.

            columns (list): List with column names to be used in the model.

            split_variable_name (str): name of column with kmeans split.

        Returns:
            dict: Dictionary with best hyperparameters.
        """
        all_hyperparameter_scores = []
        for split in df[split_variable_name].unique():
            train_X, train_y, test_X, test_y = self.__train_test_split(df, columns, split_variable_name, split)
            one_loop_hyperparameter_scores = []
            if isinstance(self.hyperparameters, list):
                for hyperparams in self.hyperparameters:
                    regressor = self.model(**hyperparams).fit(train_X, train_y)
                    y_predicted_test = regressor.predict(test_X)
                    one_loop_hyperparameter_scores.append(mean_squared_error(test_y, y_predicted_test, squared=False))
            else:
                print("hyperparameters must be a list")
            all_hyperparameter_scores.append(one_loop_hyperparameter_scores)
        mean_hyperparameters = np.mean(all_hyperparameter_scores, axis=0)
        best_hyperparameters = self.hyperparameters[
            np.argmin(mean_hyperparameters)
        ]  # not argmax because we want to minimize the error
        return best_hyperparameters

    def __save_dates(self, df):
        """
        This function saves the dates of the train and test set.

        Args:
            df (pandas.DataFrame): Dataframe with data.

        Returns:
            list of dates used in training/cv.
        """
        return list((df["date"].min(), df["date"].max()))

    def __check_columns(self, columns):
        for col in columns:
            if col in ["row", "col", "date", "opt_value"]:
                print(f"Column {col} should not be included")
                assert False

    def spatial_cv(self, df, columns):
        """
        This function performs spatial cross-validation.

        Args:
            df (pandas.DataFrame): Dataframe with data (should include all columns, both used and not).

            columns (list): List with column names to be used in the model.

        Returns:
            Nothing. But it assigns the RMSE and R2 scores for the train and test set to the model object.
                     It also assigns the best hyperparameters, predicted and real values of each outer split to the model object.
        """
        self.__check_columns(columns)
        # self.dates = self.__save_dates(df)
        self.columns = columns

        rmse_list_train = []
        rmse_list_test = []
        r2_list_train = []
        r2_list_test = []
        # self.best_hyperparameter_list = []
        # self.feature_importance_list = []
        self.cv_model_list = []

        # split the data into outer folds:
        df = self.__kmeans_split(df, "outer_area")
        # for each outer fold:
        for outer_split in df["outer_area"].unique():
            print("Spatial CV, outer split: ", outer_split)
            # define only train set (to be used in inner loop of nested cross-validation)
            train = df[df["outer_area"] != outer_split]
            # split the data into inner folds:
            train = self.__kmeans_split(train, "inner_area")
            # tune hyperparameters (all inner loops of nested cross-validation are executed in this function):
            best_hyperparam = self.__tune_hyperparameters(train, columns, split_variable_name="inner_area")
            # self.best_hyperparameter_list.append(best_hyperparam)

            # with the best hyperparameters, train the model on the outer fold:
            train_X, train_y, test_X, test_y = self.__train_test_split(
                df, columns, split_variable_name="outer_area", split_index=outer_split
            )
            regressor = self.model(**best_hyperparam).fit(train_X, train_y)
            # self.feature_importance_list.append(self.get_feature_importance(regressor, columns))
            self.cv_model_list.append(regressor)

            train_y_predicted = regressor.predict(train_X)
            test_y_predicted = regressor.predict(test_X)

            train_y_predicted = np.exp(train_y_predicted) - 1
            test_y_predicted = np.exp(test_y_predicted) - 1

            rmse_list_train.append(mean_squared_error(train_y, train_y_predicted))
            rmse_list_test.append(mean_squared_error(test_y, test_y_predicted))
            r2_list_train.append(r2_score(train_y, train_y_predicted))
            r2_list_test.append(r2_score(test_y, test_y_predicted))

        # results:
        self.rmse_train = np.mean(rmse_list_train)
        self.rmse_std_train = np.std(rmse_list_train)
        self.rmse_test = np.mean(rmse_list_test)
        self.rmse_std_test = np.std(rmse_list_test)
        self.r2_train = np.mean(r2_list_train)
        self.r2_std_train = np.std(r2_list_train)
        self.r2_test = np.mean(r2_list_test)
        self.r2_std_test = np.std(r2_list_test)

        # find best hyperparameters in for the WHOLE dataset (instaed of only one fold at a time):
        # (this trained final model is mainly used for feature importance)
        df = self.__kmeans_split(df, "final_split_areas")
        for split in df["final_split_areas"].unique():
            print("Spatial CV, final split: ", split)
            final_hyperparameters = self.__tune_hyperparameters(df, columns, split_variable_name="final_split_areas")
        # fit final model:
        self.final_model = self.model(**final_hyperparameters).fit(df[columns], df["opt_value"])
        # self.final_feature_importance = self.get_feature_importance(self.final_model, columns)

        return

    def spatial_cv_mean_benchmark(self, df, columns):
        rmse_list_train = []
        rmse_list_test = []
        r2_list_train = []
        r2_list_test = []

        self.cv_mean_list = []

        # split the data into outer folds:
        df = self.__kmeans_split(df, "outer_area")
        # for each outer fold:
        for outer_split in df["outer_area"].unique():
            train_X, train_y, test_X, test_y = self.__train_test_split(
                df, columns, split_variable_name="outer_area", split_index=outer_split
            )

            mean_ = train_y.mean()
            train_y_predicted = np.full((1, len(train_y)), mean_)[0]
            test_y_predicted = np.full((1, len(test_y)), mean_)[0]

            self.cv_mean_list.append(mean_)

            train_y_predicted = np.exp(train_y_predicted) - 1
            test_y_predicted = np.exp(test_y_predicted) - 1

            rmse_list_train.append(mean_squared_error(train_y, train_y_predicted))
            rmse_list_test.append(mean_squared_error(test_y, test_y_predicted))
            r2_list_train.append(r2_score(train_y, train_y_predicted))
            r2_list_test.append(r2_score(test_y, test_y_predicted))

        # results:
        self.rmse_train = np.mean(rmse_list_train)
        self.rmse_std_train = np.std(rmse_list_train)
        self.rmse_test = np.mean(rmse_list_test)
        self.rmse_std_test = np.std(rmse_list_test)
        self.r2_train = np.mean(r2_list_train)
        self.r2_std_train = np.std(r2_list_train)
        self.r2_test = np.mean(r2_list_test)
        self.r2_std_test = np.std(r2_list_test)

        print(f"Microwave benchmark RMSE on test set: {self.rmse_test}")
        print(f"Microwave benchmark R2 on test set: {self.r2_test}")
        print(f"Means: {self.cv_mean_list}")
        return

    def get_results(self):
        """
        This function prints the results of the model in a table.
        """
        results = pd.DataFrame(
            {
                "Set": ["Train", "Test"],
                "RMSE": [self.rmse_train, self.rmse_test],
                "RMSE_std": [self.rmse_std_train, self.rmse_std_test],
                "R2": [self.r2_train, self.r2_test],
                "R2_std": [self.r2_std_train, self.r2_std_test],
            }
        )
        return results

    def get_attributes(self):
        """
        This function prints the attributes of the model.
        """
        for attribute, value in self.__dict__.items():
            print(attribute, "=", value)
        return


#############################################


def save_object(obj):
    """
    This function saves an object to a pickle file.

    Args:
        obj (object): Object to be saved.

        filename (str): Name of the file to be saved, with extension, without path unless a subfolder is desired.
    """
    filename = r"/mnt/volume/AWS_Data/Models/" + obj.name + ".pkl"
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """
    This function loads an object from a pickle file.

    Args:
        filename (str): Name of the file to be loaded, with extension, without path unless a subfolder is desired.

    Returns:
            obj (object): Loaded object.
    """
    filename = r"/mnt/volume/AWS_Data/Models/" + filename + ".pkl"
    with open(filename, "rb") as inp:
        obj = pickle.load(inp)
    return obj
