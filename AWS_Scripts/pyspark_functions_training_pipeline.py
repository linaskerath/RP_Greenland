from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle

from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.param import Param
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, exp, log
from pyspark.ml.evaluation import RegressionEvaluator


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
        pyspark.sql.DataFrame: Dataframe with all merged data.
    """

    # Initialize SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Create date range
    start_date = datetime.strptime(date_from, "%Y-%m-%d").date()
    end_date = datetime.strptime(date_to, "%Y-%m-%d").date()
    date_range = [str(start_date + timedelta(days=x)) for x in range((end_date - start_date).days + 1)]

    dfs = []
    for melt_date in date_range:
        print(melt_date)
        # try:
        file = spark.read.parquet(df_path + "melt_" + melt_date + "_extended.parquet.gzip")
        file = file.drop("row", "col", "date")
        file = remove_data(file, removeMaskedClouds=True, removeNoMelt=True, removeLittleMelt=True)
        dfs.append(file)
        # except:
        #    continue

    # Merge dataframes
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = df.union(dfs[i])

    return df


def remove_data(df, removeMaskedClouds=True, removeNoMelt=True, removeLittleMelt=True):
    """
    Removes missing/masked mw and opt data from dataframe.
    Used for training and testing, not predicting.

    Args:
        df (pandas.DataFrame): Full train/ test dataframe.

        removeMaskedClouds (bool): True for train and test data, removes masked data from opt data.
                                   False for predicting data, keeps masked opt data.

        removeNoMelt (bool): True for train and test data, removes non-melt areas from mw data.
                             False for predicting data, keeps non-melt areas.

        removeLittleMelt (bool): True for train and test data, removes areas with little melt from mw data.
                                 False for predicting data, keeps areas with little melt.

    Returns:
        pandas.DataFrame: The same dataframe with removed water (and masked data).
    """
    spark = SparkSession.builder.getOrCreate()

    if removeMaskedClouds:
        df = df.filter(col("opt_value") != -1)

    if removeNoMelt and removeLittleMelt:
        melt_noMelt = spark.read.parquet(r"../AWS_Data/Data/split_indexes/noMelt_indexes.parquet")
        # melt_noMelt = spark.read.parquet(r"/mnt/volume/AWS_Data/Data/split_indexes/noMelt_indexes.parquet")
        melt_littleMelt = spark.read.parquet(r"../AWS_Data/Data/split_indexes/littleMelt_indexes.parquet")
        # melt_littleMelt = pd.read_parquet(r"/mnt/volume/AWS_Data/Data/split_indexes/littleMelt_indexes.parquet")

        df = df.join(melt_noMelt, on=["y", "x"], how="left")
        # rename melt column to melt_noMelt
        df = df.withColumnRenamed("melt", "melt_noMelt")
        df = df.join(melt_littleMelt, on=["y", "x"], how="left")

        df = df.filter((col("melt_noMelt") == 1) & (col("melt") == 1))
        df = df.drop("melt_noMelt", "melt")

    elif removeNoMelt:
        # melt = pd.read_parquet(r"/mnt/volume/AWS_Data/Data/split_indexes/noMelt_indexes.parquet")
        melt = spark.read.parquet(r"../AWS_Data/Data/split_indexes/noMelt_indexes.parquet")
        df = df.join(melt, on=["y", "x"], how="left")
        df = df.filter(col("melt") == 1)
        df = df.drop("melt")

    elif removeLittleMelt:
        # melt = pd.read_parquet(r"/mnt/volume/AWS_Data/Data/split_indexes/littleMelt_indexes.parquet")
        melt = spark.read.parquet(r"../AWS_Data/Data/split_indexes/littleMelt_indexes.parquet")
        df = df.join(melt, on=["y", "x"], how="left")
        df = df.filter(col("melt") == 1)
        df = df.drop("melt")

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
            if feature == "x":
                min_val, max_val = -636500.0, 824500.0
            elif feature == "y":
                min_val, max_val = -3324500.0, -662500.0
            elif feature == "mean_3":
                min_val, max_val = 0, 1
            elif feature == "mean_9":
                min_val, max_val = 0, 1
            elif feature == "sum_5":
                min_val, max_val = 0, 25
            elif feature == "mw_value_7_day_average":
                min_val, max_val = 0, 1
            elif feature == "hours_of_daylight":
                min_val, max_val = 0, 24
            elif feature == "slope_data":
                min_val, max_val = 0, 90
            elif feature == "aspect_data":
                min_val, max_val = -1, 1
            elif feature == "elevation_data":
                min_val, max_val = 0, 3694
            else:
                min_val, max_val = 1, 500

            df = df.withColumn(feature, (col(feature) - min_val) / (max_val - min_val))

        elif feature in log_feature:
            df = df.withColumn(feature, log(1 + col(feature)))
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
        param_grid = ParamGridBuilder()
        for param_name, param_values in hyperparameters.items():
            param = Param(self.model, param_name, "")
            param_grid = param_grid.addGrid(param, param_values)
        return param_grid.build()

    def __kmeans_split(self, df, split_variable_name, plot=False, verbose=False):
        """
        This function splits the data into 5 areas based on the kmeans algorithm.

        Args:
            df (DataFrame): DataFrame with data.

            split_variable_name (str): Name for the column with the k-means split. Eg. 'inner_area' or 'outer_area' loop.

            plot (bool): If True, plots the k-means split.

            verbose (bool): If True, prints the number of observations in each split.

        Returns:
            DataFrame: DataFrame with added column with k-means split.
        """
        kmeans = KMeans(k=2, seed=0)  # TODO: change k to 5
        assembler = VectorAssembler(inputCols=["x", "y"], outputCol="features")
        df = assembler.transform(df)

        model = kmeans.fit(df.select("features"))
        df = model.transform(df)

        df = df.withColumnRenamed("prediction", split_variable_name)
        df = df.drop("features")

        if verbose:
            df.groupBy(split_variable_name).count().show()

        if plot:
            import matplotlib.pyplot as plt

            pdf = df.select("x", "y", split_variable_name).toPandas()
            plt.scatter(pdf["x"], pdf["y"], c=pdf[split_variable_name], edgecolor="none", s=0.05)
            plt.show()

        return df

    def __train_test_split(self, df, columns, split_variable_name, split_index):
        """
        This function splits the data into train and test sets.

        Args:
            df (DataFrame): DataFrame with data.

            columns (list): List with column names to be used in the model.

            split_variable_name (str): Name of the column with the split variable.

            split_index (int): Index of the split (loop).

        Returns:
            DataFrame, DataFrame, DataFrame, DataFrame: Train and test sets for features and target.
        """
        train = df.filter(col(split_variable_name) != split_index)
        test = df.filter(col(split_variable_name) == split_index)

        assembler = VectorAssembler(inputCols=columns, outputCol="features")

        train_data = assembler.transform(train.select(columns)).select(col("features")).join(train.select("opt_value"))
        train_data = train_data.withColumnRenamed("opt_value", "label")

        test_data = assembler.transform(test.select(columns)).select(col("features")).join(test.select("opt_value"))
        test_data = test_data.withColumnRenamed("opt_value", "label")

        return train_data, test_data

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
        unique_splits = df.select(split_variable_name).distinct().collect()
        for row in unique_splits:
            split = row[split_variable_name]
            if split_variable_name == "inner_area":
                print("Inner Split: ", split)
            else:
                print("Final Split: ", split)
            train_data, test_data = self.__train_test_split(df, columns, split_variable_name, split)

            one_loop_hyperparameter_scores = []
            if isinstance(self.hyperparameters, list):
                # Set the parameters in the model
                for param_dict in self.hyperparameters:
                    for param, value in param_dict.items():
                        name = param.name
                        self.model.setParams(**{name: value})
                    # Fit the model
                    regressor = self.model.fit(train_data)
                    # Predict
                    y_predicted_test = regressor.transform(test_data.select("features")).select("prediction")
                    # Calculate and append MSE
                    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")
                    mse = evaluator.evaluate(y_predicted_test.join(test_data.select("label")))
                    one_loop_hyperparameter_scores.append(mse)
            else:
                print("hyperparameters must be a list")
            all_hyperparameter_scores.append(one_loop_hyperparameter_scores)
        mean_hyperparameters = np.mean(all_hyperparameter_scores, axis=0)
        best_hyperparameters = self.hyperparameters[np.argmin(mean_hyperparameters)]
        return best_hyperparameters

    def __check_columns(self, columns):
        for col in columns:
            if col in ["row", "col", "date", "opt_value"]:
                raise ValueError(f"Column {col} should not be included")

    def spatial_cv(self, df, columns, target_normalized):
        """
        This function performs spatial cross-validation.

        Args:
            df (DataFrame): DataFrame with data (should include all columns, both used and not).

            columns (list): List with column names to be used in the model.

            target_normalized (bool): If True, the target variable is normalized and will be transformed back.

        Returns:
            Nothing. But it assigns the RMSE and R2 scores for the train and test set to the model object.
                     It also assigns the best hyperparameters, predicted and real values of each outer split to the model object.
        """
        self.__check_columns(columns)
        self.columns = columns

        rmse_list_train = []
        rmse_list_test = []
        r2_list_train = []
        r2_list_test = []
        self.cv_model_list = []

        # split the data into outer folds:
        df = self.__kmeans_split(df, "outer_area")
        unique_outer_splits = df.select("outer_area").distinct().collect()
        print("--- Spatial CV ---\n")
        for row in unique_outer_splits:
            outer_split = row["outer_area"]
            print("Outer split: ", outer_split)

            # Define only the train set (to be used in the inner loop of nested cross-validation)
            train = df.filter(df["outer_area"] != outer_split)

            # Split the data into inner folds
            train = self.__kmeans_split(train, "inner_area")

            # tune hyperparameters (all inner loops of nested cross-validation are executed in this function):
            best_hyperparam = self.__tune_hyperparameters(train, columns, split_variable_name="inner_area")

            # Define the train and test sets for the outer loop of nested cross-validation
            train_data, test_data = self.__train_test_split(
                df, columns, split_variable_name="outer_area", split_index=outer_split
            )
            # Set the parameters in the model
            for param, value in best_hyperparam.items():
                name = param.name
                self.model.setParams(**{name: value})
            # Fit the model
            regressor = self.model.fit(train_data)

            self.cv_model_list.append(regressor)

            train_y_predicted = regressor.transform(train_data.select("features")).select("prediction")
            test_y_predicted = regressor.transform(test_data.select("features")).select("prediction")

            if target_normalized:
                train_y_predicted = train_y_predicted.withColumn("prediction", exp(col("prediction")) - 1)
                test_y_predicted = test_y_predicted.withColumn("prediction", exp(col("prediction")) - 1)

            evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
            rmse_list_train.append(evaluator_rmse.evaluate(train_y_predicted.join(train_data.select("label"))))
            rmse_list_test.append(evaluator_rmse.evaluate(test_y_predicted.join(test_data.select("label"))))

            evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
            r2_list_train.append(evaluator_rmse.evaluate(train_y_predicted.join(train_data.select("label"))))
            r2_list_test.append(evaluator_rmse.evaluate(test_y_predicted.join(test_data.select("label"))))

            print()

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
        final_hyperparameters = self.__tune_hyperparameters(df, columns, split_variable_name="final_split_areas")
        # Set the parameters in the final model
        for param, value in final_hyperparameters.items():
            name = param.name
            self.model.setParams(**{name: value})
        # Fit final model
        assembler = VectorAssembler(inputCols=columns, outputCol="features")
        final_train_data = assembler.transform(df).select(col("features"), col("opt_value"))
        final_train_data = final_train_data.withColumnRenamed("opt_value", "label")
        self.final_model = self.model.fit(final_train_data)

        return

    """ # TODO: DOES IT NEED TO BE ADAPTED ?
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

            rmse_list_train.append(mean_squared_error(train_y, train_y_predicted), squared=False)
            rmse_list_test.append(mean_squared_error(test_y, test_y_predicted), squared=False)
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
    """

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
