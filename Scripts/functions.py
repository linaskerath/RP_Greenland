import xarray 
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import rasterio

def read_and_prep_parquet(path, purpose):
    """
    Function to read parquet and prepare as train or test data.
    Arguments:
        path: path to file.
        purpose: {'train', 'test', 'validate', 'predict'} purpose of file.
    Returns: train/test dataset and label array (specify output datatype!)
    """
    valid = {'train', 'test', 'validate', 'predict'}
    if purpose not in valid:
        raise ValueError("Purpose must be one of %r." % valid)

    df = pd.read_parquet(path)
    if purpose in ['train', 'test', 'validate']:
        df = df.loc[df['opt_value'] != -1] # remove mask
        df = df.fillna(-1) # fill values to be able to train
        X = df[['x', 'y', 'mw_value', 'col', 'row', 'v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8', 'v9', 'mean']] # v5 is duplicated
        y = df[['opt_value']]
        return X, y
    else:
        df = df.fillna(-1) # fill values to be able to train
        X = df[['x', 'y', 'mw_value', 'col', 'row', 'v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8', 'v9', 'mean']] # v5 is duplicated
        return X


def convert_to_tif(data, path_file_metadata, path_out):
    """
    Function to convert data to tif file.
    Arguments:
        data: new file
        path_file_metadata: tif file with metadata matching expected output tif file
        path_out: output tif file destination and name path
    Returns:
        .tif file
    """
    with rasterio.open(path_file_metadata) as src:
        kwargs1 = src.meta.copy()

    with rasterio.open(path_out, 'w', **kwargs1) as dst:
        dst.write_band(1, data) # numpy array or xarray   
    return


def get_rmse(y_real, y_predicted):
    return np.sqrt(mean_squared_error(y_real, y_predicted))   