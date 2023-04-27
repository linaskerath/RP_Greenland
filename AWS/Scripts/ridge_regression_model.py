import pandas as pd
from sklearn.linear_model import Ridge

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f


df_path = r"../Data/dataframe_extended/"

date_from = "2019-06-10"
date_to = "2019-06-11"

data = f.import_data(date_from, date_to, df_path)
data = f.remove_data(data, removeMaskedClouds=True, removeNoMelt=True)
data = f.data_normalization(data)

columns = data.columns.drop(["date", "row", "col", "opt_value"])

ridge = f.Model(model=Ridge, name="RidgeRegression")
hyperparameters_for_grid = {"fit_intercept": [True]}
ridge.hyperparameters = ridge.create_hyperparameter_grid(hyperparameters_for_grid)

ridge.spatial_cv(data, columns)

f.save_object(ridge)
