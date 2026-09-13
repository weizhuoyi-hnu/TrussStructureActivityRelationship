import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def normalization_x(x):
    x_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Structure.csv', index_col=0)
    x_data = x_data.drop(['name', 'Number'],  axis=1)
    scale_method_x = MinMaxScaler(feature_range=(-1, 1))
    scale_method_x.fit(x_data)
    scale_x = scale_method_x.transform(x)
    return scale_x


def inv_normalization_x(x):
    x_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Structure.csv', index_col=0)
    x_data = x_data.drop(['name', 'Number'],  axis=1)
    scale_method_x = MinMaxScaler(feature_range=(-1, 1))
    scale_method_x.fit(x_data)
    original_x = scale_method_x.inverse_transform(x)
    return original_x


def norm_label_A(y):
    y_read_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Property.csv', index_col=0)
    y_data = y_read_data[['Isotropy']]
    y_data_log = np.log(y_data)
    scale_method_y = MinMaxScaler(feature_range=(-1, 1))
    scale_method_y.fit(y_data_log)
    y = np.asarray(y, dtype=np.float64)
    y_log = np.log(y)
    y_scaled = scale_method_y.transform(y_log)
    return y_scaled


def inv_norm_label_A(y):
    y_read_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Property.csv', index_col=0)
    y_data = y_read_data[['Isotropy']]
    y_data_log = np.log(y_data)
    scale_method_y = MinMaxScaler(feature_range=(-1, 1))
    scale_method_y.fit(y_data_log)
    y = np.asarray(y, dtype=np.float64)
    y_log = scale_method_y.inverse_transform(y)
    original_y = np.exp(y_log)
    return original_y
