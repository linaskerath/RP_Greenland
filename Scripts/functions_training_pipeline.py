"""
This script contains all necessary functions for the training pipeline.
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt # to plot kmeans splits
from sklearn.model_selection import ParameterGrid

from sklearn.tree import DecisionTreeRegressor


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


def remove_data(df, removeMaskedClouds = True, removeNoMelt = True):
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
    # df = df[df['mw_value'] != -1] 
    
    if removeMaskedClouds == True:
        df = df[df["opt_value"] != -1]

    if removeNoMelt == True:
        melt = pd.read_parquet(r"../Data/split_indexes/noMelt_indexes.parquet", index= False)
        df = df.merge(melt, how = 'left', on = ["y",'x'])
        df = df[df['melt'] == 1]

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

#############################################
# Benchmark functions
#############################################


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


#############################################
# Model training CV
#############################################
import matplotlib.pyplot as plt
import pickle

class Model:
    """ 
    This class contains models. 
    After training it also contains performance scores and the hyperparameters used to train it.
    """

    def __init__(self, model, name):  
        self.model = model
        self.hyperparameters = [] # list of dictionaries with hyperparameters
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


    def __kmeans_split(self, df, split_variable_name, plot = False, verbose = False):
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
        kmeans = KMeans(n_clusters=5, n_init="auto").fit(df[['x','y']]) #  random_state=0,
        df[split_variable_name] = kmeans.labels_

        if verbose == True:
            print(df[split_variable_name].value_counts())

        if plot == True:
            plt.scatter(df['x'], df['y'], c=df[split_variable_name], edgecolor='none', s = 0.05)
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
        test  = df[df[split_variable_name] == split_index]
        train_X = train[columns]
        train_y = train[["opt_value"]]
        test_X = test[columns]
        test_y = test[["opt_value"]] 
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
        all_hyperparameter_scores= []
        for split in df[split_variable_name].unique():
            train_X, train_y, test_X, test_y = self.__train_test_split(df, columns, split_variable_name, split)                
            one_loop_hyperparameter_scores = []
            if isinstance(self.hyperparameters, list):
                for hyperparams in self.hyperparameters:
                    regressor = self.model(random_state=0, **hyperparams).fit(train_X, train_y)
                    y_predicted_test = regressor.predict(test_X)
                    one_loop_hyperparameter_scores.append(mean_squared_error(test_y, y_predicted_test, squared=False))
            else:
                print('hyperparameters must be a list')
            all_hyperparameter_scores.append(one_loop_hyperparameter_scores)
        mean_hyperparameters = np.mean(all_hyperparameter_scores, axis=0)
        best_hyperparameters = self.hyperparameters[np.argmin(mean_hyperparameters)] # not argmax because we want to minimize the error
        return best_hyperparameters


    def __save_dates(self, df):
        """ 
        This function saves the dates of the train and test set.
        
        Args:
            df (pandas.DataFrame): Dataframe with data.

        Returns:
            list of dates used in training/cv.
        """
        return list(df['date'].unique())
    
    def get_feature_importance(self, model, columns):
        """
        This function returns the feature importance of a model.

        Args:
            model (sklearn model): Model to get feature importance from.

            columns (list): List with column names used in the model.

        Returns:
            dict: Dictionary with feature importance.
        """
        if isinstance(model, DecisionTreeRegressor):
            feature_importance = model.feature_importances_
        elif isinstance(model, RandomForestRegressor):
            feature_importance = model.feature_importances_
        elif isinstance(model, GradientBoostingRegressor):
            feature_importance = model.feature_importances_
        elif isinstance(model, LinearRegression):
            feature_importance = np.abs(model.coef_[0])
        elif isinstance(model, Ridge):
            feature_importance = np.abs(model.coef_[0])
        elif isinstance(model, Lasso):
            feature_importance = np.abs(model.coef_[0])
        elif isinstance(model, ElasticNet):
            feature_importance = np.abs(model.coef_[0])
        else:
            print('model not supported')

        feature_importance_dict = dict(zip(columns, feature_importance))
        return feature_importance_dict


    def spatial_cv(self, df, columns):
        """ 
        This function performs spatial cross-validation.
        
        Args:
            df (pandas.DataFrame): Dataframe with data.
            
            columns (list): List with column names to be used in the model.

        Returns:
            Nothing. But it assigns the RMSE and R2 scores for the train and test set to the model object.
                     It also assigns the best hyperparameters, predicted and real values of each outer split to the model object.
        """

        self.dates = self.__save_dates(df)
        #columns = list(df.columns[df.columns != 'opt_value']) # and date ( and x and Y???)

        rmse_list_train = []
        rmse_list_test = []
        r2_list_train = []
        r2_list_test = []
        self.best_hyperparameter_list = []
        self.feature_importance_list = []
        
        df = self.__kmeans_split(df, 'outer_area')
        for outer_split in df['outer_area'].unique():
            train = df[df['outer_area'] != outer_split]
            train = self.__kmeans_split(train, 'inner_area')
            best_hyperparam = self.__tune_hyperparameters(train, columns, split_variable_name = 'inner_area')
            self.best_hyperparameter_list.append(best_hyperparam)
            
            train_X, train_y, test_X, test_y = self.__train_test_split(df, columns, split_variable_name = 'outer_area', split_index = outer_split)
            regressor = self.model(random_state=0, **best_hyperparam).fit(train_X, train_y)
            self.feature_importance_list.append(self.get_feature_importance(regressor, columns))

            train_y_predicted = regressor.predict(train_X)
            test_y_predicted  = regressor.predict(test_X )

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
        

        df = self.__kmeans_split(df, 'final_split_areas')
        for split in df['final_split_areas'].unique():
            self.final_hyperparameters = self.__tune_hyperparameters(df, columns, split_variable_name = 'final_split_areas')

        self.final_model = self.model(random_state=0, **self.final_hyperparameters).fit(df[columns], df['opt_value']) 
        self.final_feature_importance = self.get_feature_importance(self.final_model, columns)
        
        return


        
    def get_results(self):
        """ 
        This function prints the results of the model in a table.
        """
        results = pd.DataFrame({'Metric': ['RMSE', 'RMSE_std', 'R2', 'R2_std'],
                                'Train': [self.rmse_train, self.rmse_std_train ,self.r2_train, self.r2_std_train],
                                'Test': [self.rmse_test, self.rmse_std_test, self.r2_test, self.r2_std_test]})
        print(results)
        return 
    
    def get_attributes(self):
        """ 
        This function prints the attributes of the model.
        """
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value)
        return
    

#############################################

def save_object(obj):
    """ 
    This function saves an object to a pickle file.

    Args:
        obj (object): Object to be saved.

        filename (str): Name of the file to be saved, with extension, without path unless a subfolder is desired.
    """
    filename = r'../Models/' + obj.name + '.pkl'
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """ 
    This function loads an object from a pickle file.
    
    Args:
        filename (str): Name of the file to be loaded, with extension, without path unless a subfolder is desired.
        
    Returns:
            obj (object): Loaded object.
    """
    filename = r'../Models/' + filename + '.pkl'
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj

#############################################
