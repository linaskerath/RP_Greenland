import pandas as pd

pd.options.mode.chained_assignment = None

import functions_training_pipeline as f

df_path = r"/mnt/volume/AWS_Data/Data/dataframe_extended/"

date_from = "2017-05-01"
date_to = "2019-07-31"

data = f.import_data(date_from, date_to, df_path)
data = f.data_normalization(data)
data.to_parquet(r"/mnt/volume/AWS_Data/Data/dataframe_model_training/training_data_NEW.parquet.gzip")
