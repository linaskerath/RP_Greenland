from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit
import matplotlib.pyplot as plt
import numpy as np


class Model:
    def __init__(self, model, name):
        self.model = model
        self.hyperparameters = []  # list of dictionaries with hyperparameters
        self.name = name

    def create_hyperparameter_grid(self, hyperparameters):
        return ParamGridBuilder().build()

    def __kmeans_split(self, df, split_variable_name, plot=False, verbose=False):
        kmeans = KMeans(k=5, seed=0).fit(df.select("x", "y"))
        df = df.withColumn(split_variable_name, kmeans.transform(df.select("x", "y")))

        if verbose:
            df.groupBy(split_variable_name).count().show()

        if plot:
            # Convert Spark DataFrame to Pandas DataFrame for plotting
            pandas_df = df.select("x", "y", split_variable_name).toPandas()
            plt.scatter(pandas_df["x"], pandas_df["y"], c=pandas_df[split_variable_name], edgecolor="none", s=0.05)
            plt.show()
        return df

    def __train_test_split(self, df, columns, split_variable_name, split_index):
        train = df.filter(df[split_variable_name] != split_index)
        test = df.filter(df[split_variable_name] == split_index)
        train_X = train.select(columns)
        train_y = train.select("opt_value").rdd.flatMap(lambda x: x).collect()
        test_X = test.select(columns)
        test_y = test.select("opt_value").rdd.flatMap(lambda x: x).collect()
        return train_X, train_y, test_X, test_y

    def __tune_hyperparameters(self, df, columns, split_variable_name):
        """
        This function performs hyperparameter tuning in the inner loop of nested cross-validation.

        Args:
            df (pyspark.sql.DataFrame): DataFrame with data.

            columns (list): List with column names to be used in the model.

            split_variable_name (str): Name of column with k-means split.

        Returns:
            dict: Dictionary with best hyperparameters.
        """
        all_hyperparameter_scores = []
        for split in df.select(split_variable_name).distinct().rdd.flatMap(lambda x: x).collect():
            train_X, train_y, test_X, test_y = self.__train_test_split(df, columns, split_variable_name, split)
            one_loop_hyperparameter_scores = []
            if isinstance(self.hyperparameters, list):
                for hyperparams in self.hyperparameters:
                    regressor = self.model(**hyperparams)
                    evaluator = RegressionEvaluator(metricName="rmse")
                    param_grid = ParamGridBuilder().build()
                    crossval = CrossValidator(estimator=regressor, estimatorParamMaps=param_grid, evaluator=evaluator)
                    cv_model = crossval.fit(train_X)
                    predictions = cv_model.transform(test_X)
                    rmse = evaluator.evaluate(predictions)
                    one_loop_hyperparameter_scores.append(rmse)
            else:
                print("hyperparameters must be a list")
            all_hyperparameter_scores.append(one_loop_hyperparameter_scores)

        mean_hyperparameters = np.mean(all_hyperparameter_scores, axis=0)
        best_hyperparameters = self.hyperparameters[np.argmin(mean_hyperparameters)]
        return best_hyperparameters

    def __check_columns(self, columns):
        for col in columns:
            if col in ["row", "col", "date", "opt_value"]:
                print(f"Column {col} should not be included")
                assert False

    def spatial_cv(self, df, columns, target_normalized):
        self.__check_columns(columns)
        self.columns = columns

        rmse_list_train = []
        rmse_list_test = []
        r2_list_train = []
        r2_list_test = []
        self.cv_model_list = []

        # Split the data into outer folds
        df = self.__kmeans_split(df, "outer_area")

        # Create a SparkSession
        spark = SparkSession.builder.getOrCreate()

        # For each outer fold
        for outer_split in df.select("outer_area").distinct().rdd.flatMap(lambda x: x).collect():
            print("Spatial CV, outer split:", outer_split)

            # Define only train set (to be used in the inner loop of nested cross-validation)
            train = df.filter(df["outer_area"] != outer_split)

            # Split the data into inner folds
            train = self.__kmeans_split(train, "inner_area")

            # Tune hyperparameters (all inner loops of nested cross-validation are executed in this function)
            best_hyperparam = self.__tune_hyperparameters(train, columns, split_variable_name="inner_area")

            # With the best hyperparameters, train the model on the outer fold
            train_X, train_y, test_X, test_y = self.__train_test_split(
                df, columns, split_variable_name="outer_area", split_index=outer_split
            )

            # Convert train and test data to Spark DataFrame
            train_data = spark.createDataFrame(train_X.rdd.zipWithIndex().map(lambda x: (x[1],) + x[0]).toDF())
            train_data = train_data.toDF(*["index"] + train_X.columns)
            train_data = train_data.withColumn("opt_value", lit(train_y))

            test_data = spark.createDataFrame(test_X.rdd.zipWithIndex().map(lambda x: (x[1],) + x[0]).toDF())
            test_data = test_data.toDF(*["index"] + test_X.columns)
            test_data = test_data.withColumn("opt_value", lit(test_y))

            # Create the model with the best hyperparameters
            regressor = self.model(**best_hyperparam)
            model = regressor.fit(train_data)
            self.cv_model_list.append(model)

            # Make predictions on train and test data
            train_y_predicted = model.transform(train_data).select("prediction").rdd.flatMap(lambda x: x).collect()
            test_y_predicted = model.transform(test_data).select("prediction").rdd.flatMap(lambda x: x).collect()

            if target_normalized:
                train_y_predicted = [np.exp(y) - 1 for y in train_y_predicted]
                test_y_predicted = [np.exp(y) - 1 for y in test_y_predicted]

            evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
            rmse_list_train = evaluator.evaluate(train_y_predicted)
            rmse_list_test = evaluator.evaluate(test_y_predicted)
            evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
            r2_list_train = evaluator.evaluate(train_y_predicted)
            r2_list_test = evaluator.evaluate(test_y_predicted)

        # Results
        self.rmse_train = np.mean(rmse_list_train)
        self.rmse_std_train = np.std(rmse_list_train)
        self.rmse_test = np.mean(rmse_list_test)
        self.rmse_std_test = np.std(rmse_list_test)
        self.r2_train = np.mean(r2_list_train)
        self.r2_std_train = np.std(r2_list_train)
        self.r2_test = np.mean(r2_list_test)
        self.r2_std_test = np.std(r2_list_test)

        # Find best hyperparameters for the WHOLE dataset (instead of only one fold at a time)
        df = self.__kmeans_split(df, "final_split_areas")
        for split in df.select("final_split_areas").distinct().rdd.flatMap(lambda x: x).collect():
            print("Spatial CV, final split:", split)
            final_hyperparameters = self.__tune_hyperparameters(df, columns, split_variable_name="final_split_areas")

        # Fit final model
        self.final_model = self.model(**final_hyperparameters)
        final_data = spark.createDataFrame(df.rdd.zipWithIndex().map(lambda x: (x[1],) + x[0]).toDF())
        final_data = final_data.toDF(*["index"] + df.columns)
        final_data = final_data.withColumn("opt_value", lit(df["opt_value"]))
        self.final_model = self.final_model.fit(final_data)

        return


def save_object(obj):
    """
    This function saves an object to a file using PySpark.

    Args:
        obj (object): Object to be saved.
    """
    filename = "/mnt/volume/AWS_Data/Models/" + obj.name + ".parquet"
    spark = SparkSession.builder.getOrCreate()
    obj_df = spark.createDataFrame([obj])
    obj_df.write.parquet(filename)
