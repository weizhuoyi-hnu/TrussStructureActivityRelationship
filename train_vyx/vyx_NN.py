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

hidden_layer = [500, 1000, 1000, 1000, 1000, 1000, 500]
learning_rate = 1e-4
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
label_name = ['vyx']
feature_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Structure.csv', index_col=0)
label_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Property.csv', index_col=0)
feature = feature_data[feature_name]
label = label_data[label_name]

feature_scaled = normalization_x(feature)
label_scaled = norm_label_vyx(label)

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

model = neural_network(70, hidden_layer, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
torch.save(model, 'F:/wzy/TrussStructureActivityRelationship/model/NN/vyx/neural_network_vyx.pt')
pd.DataFrame(data=train_history, index=None, columns=['train_loss']).to_csv('F:/wzy/TrussStructureActivityRelationship/model/NN/vyx/train_loss_vyx.csv')
pd.DataFrame(data=test_history, index=None, columns=['test_loss']).to_csv('F:/wzy/TrussStructureActivityRelationship/model/NN/vyx/test_loss_vyx.csv')

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

    print('---------------------------\n')
    print('Train R2: ', train_R2, '\n')
    print('Test R2: ', test_R2, '\n')
    print('---------------------------\n')
    print('Train MAE: ', train_MAE, '\n')
    print('Test MAE: ', test_MAE, '\n')
    print('---------------------------\n')
    print('Train MSE: ', train_MSE, '\n')
    print('Test MSE: ', test_MSE, '\n')
    print('---------------------------\n')

    feature_train_all_np = feature_train_all.cpu().detach().numpy()
    feature_test_np = feature_test.cpu().detach().numpy()
    label_train_all_np = label_train_all.cpu().detach().numpy()
    label_test_np = label_test.cpu().detach().numpy()
    train_label_pred_np = train_label_pred.cpu().detach().numpy()
    test_label_pred_np = test_label_pred.cpu().detach().numpy()
    inv_scaled_train_label_true = inv_norm_label_vyx(label_train_all_np.reshape(-1, 1))
    inv_scaled_test_label_true = inv_norm_label_vyx(label_test_np.reshape(-1, 1))
    inv_scaled_train_label_pred = inv_norm_label_vyx(train_label_pred_np.reshape(-1, 1))
    inv_scaled_test_label_pred = inv_norm_label_vyx(test_label_pred_np.reshape(-1, 1))

    pd.DataFrame(feature_train_all_np, columns=feature_name).to_csv(r'F:/wzy/TrussStructureActivityRelationship/model/NN/vyx/feature_vyx_train.csv', index=False)
    pd.DataFrame(feature_test_np, columns=feature_name).to_csv(r'F:/wzy/TrussStructureActivityRelationship/model/NN/vyx/feature_vyx_test.csv', index=False)

    pd.DataFrame(inv_scaled_train_label_true, columns=label_name).to_csv(r'F:/wzy/TrussStructureActivityRelationship/model/NN/vyx/label_true_vyx_train.csv', index=False)
    pd.DataFrame(inv_scaled_test_label_true, columns=label_name).to_csv(r'F:/wzy/TrussStructureActivityRelationship/model/NN/vyx/label_true_vyx_test.csv', index=False)
    pd.DataFrame(inv_scaled_train_label_pred, columns=label_name).to_csv(r'F:/wzy/TrussStructureActivityRelationship/model/NN/vyx/label_pred_vyx_train.csv', index=False)
    pd.DataFrame(inv_scaled_test_label_pred, columns=label_name).to_csv(r'F:/wzy/TrussStructureActivityRelationship/model/NN/vyx/label_pred_vyx_test.csv', index=False)
