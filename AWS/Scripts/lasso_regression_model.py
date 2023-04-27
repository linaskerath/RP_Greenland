import pandas as pd
from sklearn.linear_model import Lasso

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f


df_path = r"../Data/dataframe_extended/"

date_from = "2019-06-10"
date_to = "2019-06-11"

data = f.import_data(date_from, date_to, df_path)
data = f.remove_data(data, removeMaskedClouds=True, removeNoMelt=True)
data = f.data_normalization(data)

columns = data.columns.drop(["date", "row", "col", "opt_value"])

lasso = f.Model(model=Lasso, name="LassoRegression")
hyperparameters_for_grid = {"fit_intercept": [True], "alpha": [0.1]}
lasso.hyperparameters = lasso.create_hyperparameter_grid(hyperparameters_for_grid)

lasso.spatial_cv(data, columns)

f.save_object(lasso)
