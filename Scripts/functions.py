import xarray 
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import rasterio
from tqdm import tqdm

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
        X = df[['x', 'y', 'mw_value', 'col', 'row', 'v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8', 'v9', 'mean', 'elevation_data']] # v5 is duplicated
        y = df[['opt_value']]
        return X, y
    else:
        df = df.fillna(-1) # fill values to be able to train
        X = df[['x', 'y', 'mw_value', 'col', 'row', 'v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8', 'v9', 'mean', 'elevation_data']] # v5 is duplicated
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


def information(path_to_file):
    """
    Function to print raster file metadata.
    Arguments:
        path_to_file: path to dile
    Returns:
        Only prints information, no returns.
    """
    with rasterio.open(path_to_file) as src:
        print('BOUNDS:')
        print(f'    {src.bounds}')
        print('METADATA:')
        print(f'    {src.meta}')
        #print(src.crs)

    data = xarray.open_dataarray(path_to_file)
    print('MORE CRS INFO:')
    print(f'    {data.spatial_ref.crs_wkt}')
    print('RESOLUTION:')
    print(f'    {data.rio.resolution()}')
    # include min, max, variables, examples..?
    return


def save_prediction_tif(X_pred, y_predicted, path_out):
    """
    Function to write predictions to .tif.
    Arguments:
        X_pred: data to be predicted on.
        y_predicted: predicted labels in array, or pandas series.
        path_out: path to save .tif file with file name.
    Returns: No return, writes data to path.
    """
    # join prediction and coordinates (row, col)
    X_pred['prediction'] = y_predicted 

    # original matrix shape:
    nan_matrix = np.full((2663, 1462), np.nan)

    for row in tqdm(X_pred.iterrows()): # fix progress bar?
        row_index = int(row[1]['row'])
        col_index = int(row[1]['col'])
        pred_val = row[1]['prediction']
        nan_matrix[row_index][col_index] = pred_val

    # file to take reference metadata from is interpolated transformed file
    path_metadata_reference =  r"../Data/microwave-rs/mw_interpolated/2019-07-01_mw.tif"

    convert_to_tif(nan_matrix, path_metadata_reference, path_out)

    return    
