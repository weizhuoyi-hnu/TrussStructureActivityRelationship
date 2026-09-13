from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from torch import nn
import torch
from lightgbm import LGBMRegressor


# RF
def random_forest(num_tree, depth_tree):
    model = RandomForestRegressor(n_estimators=num_tree, max_depth=depth_tree, n_jobs=10, random_state=2025)
    return model


# XGBoost
def extreme_gradient_boosting(num_tree, depth_tree, lr):
    model = xgb.XGBRegressor(n_estimators=num_tree, max_depth=depth_tree, learning_rate=lr,
                             n_jobs=10, random_state=2025)
    return model


# KNN
def k_nearest_neighbor(num_neighbor, method):
    model = KNeighborsRegressor(n_neighbors=num_neighbor, weights=method, n_jobs=8)
    return model


# LightGBM
def light_gradient_boosting(num_leaf, num_tree, depth_tree, lr):
    model = LGBMRegressor(objective='regression', metric='rmse', n_jobs=10,
                          num_leaves=num_leaf, max_depth=depth_tree, learning_rate=lr, n_estimators=num_tree)
    return model


# NN
def neural_network(input_num, hidden_arch, output_num):
    net = nn.Sequential()
    layer_input = input_num
    layer_num = 1
    for i in range(len(hidden_arch)):
        net.add_module('layer'+str(layer_num), nn.Linear(layer_input, hidden_arch[i]))
        net.add_module('activation'+str(layer_num), nn.ReLU())
        layer_input = hidden_arch[i]
        layer_num += 1
    net.add_module('layer'+str(layer_num), nn.Linear(layer_input, output_num))
    return net


# R2
def calculate_R2(pred_data, true_data):
    R2 = torch.zeros(pred_data.shape[1])
    for i in range(pred_data.shape[1]):
        y_pred = pred_data[:, i]
        y = true_data[:, i]
        SSR = torch.norm(y_pred - y)**2
        SST = torch.norm(torch.mean(y_pred) - y)**2
        R2[i] = 1.0 - SSR/SST
    return R2


# MAE
def calculate_MAE(pred_data, true_data):
    MAE = torch.zeros(pred_data.shape[1])
    for i in range(pred_data.shape[1]):
        y_pred = pred_data[:, i]
        y = true_data[:, i]
        MAE[i] = torch.norm((y_pred - y), p=1)/pred_data.shape[0]
    return MAE.double()


# MSE
def calculate_MSE(pred_data, true_data):
    MSE = torch.zeros(pred_data.shape[1])
    for i in range(pred_data.shape[1]):
        y_pred = pred_data[:, i]
        y = true_data[:, i]
        MSE[i] = torch.norm((y_pred - y))**2/pred_data.shape[0]
    return MSE.double()
