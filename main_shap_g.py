import torch
import torch.nn as nn
import shap
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

torch.manual_seed(2025)
np.random.seed(2025)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
shap.initjs()


def normalization_x(x):
    x_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Structure.csv',index_col=0)
    x_data = x_data.drop(['name', 'Number'], axis=1)
    scale_method_x = MinMaxScaler(feature_range=(-1, 1))
    scale_method_x.fit(x_data)
    scale_x = scale_method_x.transform(x)
    return scale_x


def norm_label_g(y):
    y_read_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Property.csv', index_col=0)
    y_data = y_read_data[['g']]
    y_data_log = np.log(y_data)
    scale_method_y = MinMaxScaler(feature_range=(-1, 1))
    scale_method_y.fit(y_data_log)
    y = np.asarray(y, dtype=np.float64)
    y_log = np.log(y)
    y_scaled = scale_method_y.transform(y_log)
    return y_scaled


feature_name = ['block_num_beam', 'block_design_area', 'block_length_mean',
                'block_length_var', 'block_length_skew', 'block_length_kurt', 'block_angle_mean', 'block_angle_var',
                'block_angle_skew', 'block_angle_kurt', 'unit_coord_diff_x', 'unit_coord_diff_y',
                'unit_design_area', 'unit_length_var', 'unit_length_skew', 'unit_length_kurt',
                'unit_angle_mean', 'unit_angle_skew', 'unit_angle_kurt', 'nearest_centroid_distance_mean',
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
label_name = ['g']
feature_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Structure.csv', index_col=0)
label_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Property.csv', index_col=0)
feature = feature_data[feature_name]
label = label_data[label_name]
feature_scaled = normalization_x(feature)
label_scaled = norm_label_g(label)
feature_tensor = torch.tensor(feature_scaled, dtype=torch.float32).to(device)
label_tensor = torch.tensor(label_scaled, dtype=torch.float32).to(device)
model = torch.load(r'F:\wzy\TrussStructureActivityRelationship\model\NN\g\neural_network_g.pt', map_location=device, weights_only=False)
model.eval()
background_size = 50
indices = np.random.choice(feature_tensor.shape[0], background_size, replace=False)
background = feature_tensor[indices]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer(feature_tensor)
shap_vals = shap_values.values
shap_vals = shap_vals[:, :, 0]
shap_mean = np.mean(np.abs(shap_vals), axis=0)
exp_bar = shap.Explanation(
    values=shap_mean,
    feature_names=feature_name
)
shap_df = pd.DataFrame({
    "feature": feature_name,
    "mean_abs_shap": shap_mean
})
shap_df = shap_df.sort_values(by="mean_abs_shap",ascending=False).reset_index(drop=True)
shap_df.to_csv(r'F:\wzy\TrussStructureActivityRelationship\shap_analysis\all_shap_values_g.csv', index=False)
