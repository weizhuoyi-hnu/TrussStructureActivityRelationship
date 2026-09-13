import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import *
from normalization import *
import joblib


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
feature_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Structure.csv', index_col=0)
label_data = pd.read_csv(r'F:\wzy\TrussStructureActivityRelationship\data\Property.csv', index_col=0)
X = feature_data[feature_name].reset_index(drop=True)
Y = label_data[['Isotropy']].reset_index(drop=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2025)

X_train_scaled = pd.DataFrame(normalization_x(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(normalization_x(X_test), columns=X_test.columns)
Y_train_scaled = norm_label_A(Y_train)
Y_test_scaled = norm_label_A(Y_test)
X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)
Y_train_scaled = Y_train_scaled.astype(np.float32)
Y_test_scaled = Y_test_scaled.astype(np.float32)

ML_model = extreme_gradient_boosting(num_tree=10000, depth_tree=7, lr=0.1)

ML_model.fit(X_train_scaled, Y_train_scaled.ravel())

Y_train_pred_scaled = ML_model.predict(X_train_scaled)
Y_test_pred_scaled = ML_model.predict(X_test_scaled)
Y_train_pred = inv_norm_label_A(Y_train_pred_scaled.reshape(-1, 1))
Y_test_pred = inv_norm_label_A(Y_test_pred_scaled.reshape(-1, 1))

print("Results：")
mae_train = mean_absolute_error(Y_train, Y_train_pred)
mae_test = mean_absolute_error(Y_test, Y_test_pred)
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)
print(f"ex training error: MAE = {mae_train:.2e}, MSE = {mse_train:.2e}, R² = {r2_train:.4f}")
print(f"ex testing error: MAE = {mae_test:.2e}, MSE = {mse_test:.2e}, R² = {r2_test:.4f}")

joblib.dump(ML_model, r'F:\wzy\TrussStructureActivityRelationship\model\XGBOOST\A\model_XGBOOST_A.pkl')

pd.DataFrame(X_train, columns=X.columns).to_csv(r'F:\wzy\TrussStructureActivityRelationship\model\XGBOOST\A\feature_A_train.csv', index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv(r'F:\wzy\TrussStructureActivityRelationship\model\XGBOOST\A\feature_A_test.csv', index=False)

pd.DataFrame(Y_train, columns=Y.columns).to_csv(r'F:\wzy\TrussStructureActivityRelationship\model\XGBOOST\A\label_true_A_train.csv', index=False)
pd.DataFrame(Y_test, columns=Y.columns).to_csv(r'F:\wzy\TrussStructureActivityRelationship\model\XGBOOST\A\label_true_A_test.csv', index=False)
pd.DataFrame(Y_train_pred, columns=Y.columns).to_csv(r'F:\wzy\TrussStructureActivityRelationship\model\XGBOOST\A\label_pred_A_train.csv', index=False)
pd.DataFrame(Y_test_pred, columns=Y.columns).to_csv(r'F:\wzy\TrussStructureActivityRelationship\model\XGBOOST\A\label_pred_A_test.csv', index=False)
