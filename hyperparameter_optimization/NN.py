import numpy as np
import pandas as pd
import torch
from torch import nn
from itertools import product
from torch.utils.data import DataLoader, TensorDataset, random_split
from normalization import *
from model import *

np.random.seed(2025)
torch.manual_seed(2025)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 1280
train_epoch = 200

feature_name = ['block_num_beam', 'block_design_area', 'block_length_mean',
                'block_length_var', 'block_length_skew', 'block_length_kurt', 'block_angle_mean', 'block_angle_var',
                'block_angle_skew', 'block_angle_kurt', 'unit_coord_diff_x', 'unit_coord_diff_y',
                'unit_design_area', 'unit_length_var', 'unit_length_skew', 'unit_length_kurt',
                'unit_angle_mean', 'unit_angle_skew', 'unit_angle_kurt',  'nearest_centroid_distance_mean',
                'nearest_centroid_distance_var', 'nearest_centroid_distance_skew', 'nearest_centroid_distance_kurt',
                'centroid_distance_mean', 'centroid_distance_var', 'centroid_distance_skew', 'centroid_distance_kurt',
                'sin_angle_mean', 'sin_angle_var', 'sin_angle_skew', 'sin_angle_kurt', 'cos_angle_mean',
                'cos_angle_var', 'cos_angle_skew', 'cos_angle_kurt', 'num_node', 'num_edge',
                'num_polygon', 'poly_num_edge_mean', 'poly_num_edge_var', 'poly_num_edge_skew', 'poly_num_edge_kurt',
                'poly_area_mean', 'poly_area_var', 'poly_area_skew', 'poly_area_kurt', 'poly_perimeter_mean',
                'poly_perimeter_var', 'poly_perimeter_skew', 'poly_perimeter_kurt', 'poly_circularity_mean',
                'poly_circularity_var', 'poly_circularity_skew', 'poly_circularity_kurt', 'poly_compactness_mean',
                'poly_compactness_var', 'poly_compactness_skew', 'poly_compactness_kurt', 'poly_eccentricity_mean',
                'poly_eccentricity_var', 'poly_eccentricity_skew', 'poly_eccentricity_kurt', 'poly_min_edge_mean',
                'poly_min_edge_var', 'poly_min_edge_skew', 'poly_min_edge_kurt', 'nodal_connectivity_mean',
                'node_connectivity_var', 'node_connectivity_skew', 'node_connectivity_kurt']
label_name = ['ey']  # ey, ex, g, vxy, vyx, Isotropy
feature_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Structure.csv', index_col=0)
label_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Property.csv', index_col=0)
feature = feature_data[feature_name]
label = label_data[label_name]

feature_scaled = normalization_x(feature)
label_scaled = norm_label_ey(label)

feature_tensor = torch.tensor(feature_scaled).to(device)
label_tensor = torch.tensor(label_scaled).to(device)

dataset = TensorDataset(feature_tensor.float(), label_tensor.float())
l_train = round(len(dataset)*0.8)
l_test = len(dataset) - l_train
print('**************************************************************')
print('train: %d, test: %d' % (l_train, l_test))
print('**************************************************************')
train_dataset, test_dataset = random_split(dataset, [l_train, l_test], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)
feature_test, label_test = next(iter(test_loader))

param_grid = {
    'num_layer': [3, 4, 5],
    'num_neuron': [1000, 2000, 3000],
    'learning_rate': [1e-3, 1e-4, 1e-5]
}

param_combinations = list(product(param_grid['num_layer'], param_grid['num_neuron'], param_grid['learning_rate']))

results_list = []

for k, (n_layer, n_neuron, lr) in enumerate(param_combinations):
    print(f"\n[{k+1}/{len(param_combinations)}] Training model with: "
          f"num_layer={n_layer}, num_neuron={n_neuron}, learning_rate={lr}")
    hidden_layer = [n_neuron // 2] + [n_neuron] * n_layer + [n_neuron // 2]
    model = neural_network(70, hidden_layer, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    print('**************************************************************')
    print('fwd_net:\n', model)
    print('**************************************************************')
    print('Training neural network model')
    print('---------------------------')
    train_history, test_history = [], []
    for epoch_iter in range(train_epoch):
        train_loss = 0.0
        for i, batch in enumerate(train_loader, 0):
            feature_train = batch[0]
            label_train = batch[1]
            model.train()
            label_train_pred = model(feature_train)
            loss = loss_function(label_train_pred, label_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
        feature_test, label_test = feature_test.to(device), label_test.to(device)
        label_test_pred = model(feature_test)
        test_loss = loss_function(label_test_pred, label_test).item()
        print('Epoch %d, train_loss: %f, test_loss: %f' % (epoch_iter + 1, train_loss, test_loss))
        train_history.append(train_loss)
        test_history.append(test_loss)
    print('---------------------------')

    with torch.no_grad():
        print('Testing neural network model')
        train_loader_all = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=0)
        feature_train_all, label_train_all = next(iter(train_loader_all))
        feature_test, label_test = feature_test.to(device), label_test.to(device)
        model.eval()
        train_label_pred = model(feature_train_all)
        test_label_pred = model(feature_test)

        train_R2 = calculate_R2(train_label_pred, label_train_all)
        train_MAE = calculate_MAE(train_label_pred, label_train_all)
        train_MSE = calculate_MSE(train_label_pred, label_train_all)
        test_R2 = calculate_R2(test_label_pred, label_test)
        test_MAE = calculate_MAE(test_label_pred, label_test)
        test_MSE = calculate_MSE(test_label_pred, label_test)
        train_R2 = train_R2.item()
        train_MAE = train_MAE.item()
        train_MSE = train_MSE.item()
        test_R2 = test_R2.item()
        test_MAE = test_MAE.item()
        test_MSE = test_MSE.item()
        print(f"training: MAE={mae_train:.2e}, MSE={mse_train:.2e}, R²={r2_train:.4f}")
        print(f"testing: MAE={mae_test:.2e}, MSE={mse_test:.2e}, R²={r2_test:.4f}")
        results_list.append({
            'num_layer': n_layer,
            'num_neuron': n_neuron,
            'learning_rate': lr,
            'train_mae': train_MAE,
            'test_mae': test_MAE,
            'train_mse': train_MSE,
            'test_mse': test_MSE,
            'train_r2': train_R2,
            'test_r2': test_R2
        })

df_results = pd.DataFrame(results_list)
df_results.sort_values(by='test_r2', inplace=True)
df_results.to_csv(r'F:\wzy\TrussStructureActivityRelationship\result\NN\ey\manual_gridsearch_results.csv', index=False)
