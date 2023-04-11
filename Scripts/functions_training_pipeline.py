"""
This script contains all necessary functions for the training pipeline.
"""

import pandas as pd
from tqdm import tqdm
#from sklearn.model_selection import TimeSeriesSplit
#from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt # to plot kmeans splits


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
# Model training class, contains CV
#############################################
import matplotlib.pyplot as plt
import pickle

class Model:
    """ 
    This class contains models. 
    After training it also contains performance scores and the hyperparameters used to train it.
    """

    def __init__(self, model):  
        self.model = model
        self.hyperparameters = [] # list of dictionaries with hyperparameters
    
    def create_hyperparameter_grid(self, hyperparameters):
        """
        Creates a grid with all possible combinations of hyperparameters.

        Args:
            hyperparameters (dict): Dictionary with hyperparameters.

        Returns:
            list: List with dictionaries with all possible combinations of hyperparameters.
        """
        hyperparameter_grid = []
        for i in range(len(hyperparameters)):
            hyperparameter_grid.append(list(hyperparameters.values())[i])
        hyperparameter_grid = list(itertools.product(*hyperparameter_grid))
        hyperparameter_grid = [dict(zip(hyperparameters.keys(), values)) for values in hyperparameter_grid]
        return hyperparameter_grid


    def __kmeans_split(self, df, loop, plot = False):
        """ 
        This function splits the data into 5 areas based on the kmeans algorithm.

        Args:
            df (pandas.DataFrame): Dataframe with data.

            loop (str): 'inner' or 'outer' loop.

            plot (bool): If True, plots the kmeans split.

        Returns:
            pandas.DataFrame: Dataframe with added column with kmeans split.
        
        """
        kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(df[['x','y']])
        if loop == 'inner':
            df['inner_area'] = kmeans.labels_
        elif loop == 'outer':
            df['outer_area'] = kmeans.labels_

        if plot == True:
            print(df[loop+'_area'].value_counts())
            plt.scatter(df['x'], df['y'], c=df[loop+'_area'], edgecolor='none', s = 0.05)
            plt.show()
        return df
    
    def __train_test_split(self, df, columns, split_index):
        """ 
        This function splits the data into train and test set.

        Args:
            df (pandas.DataFrame): Dataframe with data.

            columns (list): List with column names to be used in the model.

            split_index (int): Index of the split (loop).

        Returns:
            pandas.DataFrame: Dataframe with added column with kmeans split.
        """
        inner_train = df[df['inner_area'] != split_index]
        inner_test  = df[df['inner_area'] == split_index]
        train_X = inner_train[columns]
        train_y = inner_train[["opt_value"]]
        test_X = inner_test[columns]
        test_y = inner_test[["opt_value"]] 
        return train_X, train_y, test_X, test_y

    def __inner_loop_tune_hyperparameters(self, df, columns):
        """ 
        This function performs hyperparameter tuning in (inner loop of nested cross-validation).
        
        Args:
            df (pandas.DataFrame): Dataframe with data.

            columns (list): List with column names to be used in the model.

        Returns:
            dict: Dictionary with best hyperparameters.
        """
        all_inner_loops_hyperparameter_scores= []
        for inner_split in df['inner_area'].unique():
            inner_train_X, inner_train_y, inner_test_X, inner_test_y = self.__train_test_split(df, columns, inner_split)                
            hyperparameter_scores = []
            if isinstance(self.hyperparameters, list):
                for hyperparams in self.hyperparameters:
                    regressor = self.model(random_state=0, **hyperparams).fit(inner_train_X, inner_train_y)
                    y_predicted_test = regressor.predict(inner_test_X)
                    hyperparameter_scores.append(mean_squared_error(inner_test_y, y_predicted_test, squared=False))
            else:
                print('hyperparameters must be a list')
            all_inner_loops_hyperparameter_scores.append(hyperparameter_scores)
        mean_hyperparameters = np.mean(all_inner_loops_hyperparameter_scores, axis=0)
        best_inner_hyperparameters = self.hyperparameters[np.argmin(mean_hyperparameters)] # not argmax because we want to minimize the error
        return best_inner_hyperparameters
    
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
        rmse_list_train = []
        rmse_list_test = []
        r2_list_train = []
        r2_list_test = []
        self.best_hyperparameters = []
        predictions_train = []
        predictions_test = []
        real_values_train = []
        real_values_test = []
        
        df = self.__kmeans_split(df, 'outer') #, plot = True #df = self.__cv_split_outer_loop(df)
        for outer_split in df['outer_area'].unique():
            #if outer_split == 1: # remove
            train = df[df['outer_area'] != outer_split]
            train = self.__kmeans_split(train, 'inner') #train = self.__cv_split_inner_loop(train)
            best_hyperparam= self.__inner_loop_tune_hyperparameters(train, columns)
            self.best_hyperparameters.append(best_hyperparam)
            
            train_X, train_y, test_X, test_y = self.__train_test_split(train, columns, outer_split)
            print(f'length train_X: {len(train_X)}, length train_y: {len(train_y)}, length test_X: {len(test_X)}, length test_y: {len(test_y)}')
            regressor = self.model(random_state=0, **best_hyperparam).fit(train_X, train_y)
            train_y_predicted = regressor.predict(train_X)
            test_y_predicted  = regressor.predict(test_X )
            predictions_train.append(train_y_predicted)
            predictions_test.append(test_y_predicted)
            real_values_train.append(train_y)
            real_values_test.append(test_y)

            rmse_list_train.append(mean_squared_error(train_y, train_y_predicted))
            rmse_list_test.append(mean_squared_error(test_y, test_y_predicted))
            r2_list_train.append(r2_score(train_y, train_y_predicted))
            r2_list_test.append(r2_score(test_y, test_y_predicted))

            # else: # tb removed
            #     continue
        # results:
        self.rmse_train = np.mean(rmse_list_train)
        self.rmse_std_train = np.std(rmse_list_train)
        self.rmse_test = np.mean(rmse_list_test)
        self.rmse_std_test = np.std(rmse_list_test)
        self.r2_train = np.mean(r2_list_train)
        self.r2_std_train = np.std(r2_list_train)
        self.r2_test = np.mean(r2_list_test)
        self.r2_std_test = np.std(r2_list_test)
        
        self.outer_loop_results = {'rmse_list_train': rmse_list_train,
                                   'rmse_list_test' : rmse_list_test,
                                   'r2_list_train'  : r2_list_train,
                                   'r2_list_test'   : r2_list_test}
        
        self.outer_loop_predictions = {'train_y_predicted': predictions_train,
                                       'test_y_predicted' : predictions_test}
        self.outer_loop_real_values = {'train_y': real_values_train,
                                        'test_y' : real_values_test}
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
    


def save_object(obj, filename):
    """ 
    This function saves an object to a pickle file.

    Args:
        obj (object): Object to be saved.

        filename (str): Name of the file to be saved, with extension, without path unless a subfolder is desired.
    """
    filename = r'../Models/' + filename + '.pkl'
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