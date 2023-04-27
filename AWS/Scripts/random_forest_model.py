import pandas as pd
from sklearn.linear_model import RandomForestRegressor

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f


df_path = r"../Data/dataframe_extended/"

date_from = "2019-06-10"
date_to = "2019-06-11"

data = f.import_data(date_from, date_to, df_path)
data = f.remove_data(data, removeMaskedClouds=True, removeNoMelt=True)
data = f.data_normalization(data)

columns = data.columns.drop(["date", "row", "col", "opt_value"])

rf = f.Model(model=RandomForestRegressor, name="RandomForest")
hyperparameters_for_grid = {"n_estimators": [10, 20], "max_depth": [4]}
rf.hyperparameters = rf.create_hyperparameter_grid(hyperparameters_for_grid)

rf.spatial_cv(data, columns)

f.save_object(rf)
