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


def norm_label_vxy(y):
    y_read_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Property.csv', index_col=0)
    y_data = y_read_data[['vxy']]
    scale_method_y = MinMaxScaler(feature_range=(-1, 1))
    scale_method_y.fit(y_data)
    scale_y = scale_method_y.transform(y)
    return scale_y


def inv_norm_label_vxy(y):
    y_read_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Property.csv', index_col=0)
    y_data = y_read_data[['vxy']]
    scale_method_y = MinMaxScaler(feature_range=(-1, 1))
    scale_method_y.fit(y_data)
    original_y = scale_method_y.inverse_transform(y)
    return original_y
