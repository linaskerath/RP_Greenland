import pandas as pd

df_path_in = r"/mnt/volume/AWS_Data/Data/dataframe_model_training/training_data_NEW.parquet.gzip"
df_path_out = r"/mnt/volume/AWS_Data/Data/dataframe_model_training/training_data_UPPER_HALF.parquet.gzip"

#df_path_in = r'../AWS_Data/Data/dataframe_model_training/training_data_NEW.parquet.gzip'
#df_path_out = r'../AWS_Data/Data/dataframe_model_training/training_data_UPPER_HALF.parquet.gzip'

data = pd.read_parquet(df_path_in)
data = data[data['y'] > -2000000.0]
data.to_parquet(df_path_out)



