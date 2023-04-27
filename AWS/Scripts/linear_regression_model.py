import pandas as pd
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f


df_path = r"../Data/dataframe_extended/"

date_from = "2019-06-10"
date_to = "2019-06-11"

data = f.import_data(date_from, date_to, df_path)
data = f.remove_data(data, removeMaskedClouds=True, removeNoMelt=True)
data = f.data_normalization(data)

columns = data.columns.drop(["date", "row", "col", "opt_value"])

lr = f.Model(model=LinearRegression, name="LinearRegression")
hyperparameters_for_grid = {"fit_intercept": [True]}
lr.hyperparameters = lr.create_hyperparameter_grid(hyperparameters_for_grid)

lr.spatial_cv(data, columns)

f.save_object(lr)
