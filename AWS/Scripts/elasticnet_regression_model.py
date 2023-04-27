import pandas as pd
from sklearn.linear_model import ElasticNet

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f


df_path = r"../Data/dataframe_extended/"

date_from = "2017-05-01"
date_to = "2019-07-31"

data = f.import_data(date_from, date_to, df_path)
data = f.remove_data(data, removeMaskedClouds=True, removeNoMelt=True)
data = f.data_normalization(data)

columns = data.columns.drop(["date", "row", "col", "opt_value"])

elasticnet = f.Model(model=ElasticNet, name="ElasticNetRegression")
hyperparameters_for_grid = {"alpha": [0.5, 1, 2, 5, 10], "l1_ratio": [0.2, 0.5, 1, 2, 5, 10]}
elasticnet.hyperparameters = elasticnet.create_hyperparameter_grid(hyperparameters_for_grid)

elasticnet.spatial_cv(data, columns)

f.save_object(elasticnet)
