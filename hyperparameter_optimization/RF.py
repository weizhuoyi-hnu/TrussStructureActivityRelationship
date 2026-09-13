import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from normalization import *


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
Y = label_data[['ey']].reset_index(drop=True)  # ey, ex, g, vxy, vyx, Isotropy

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2025)

X_train_scaled = normalization_x(X_train)
X_test_scaled = normalization_x(X_test)
Y_train_scaled = norm_label_ey(Y_train)
Y_test_scaled = norm_label_ey(Y_test)
X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)
Y_train_scaled = Y_train_scaled.astype(np.float32)
Y_test_scaled = Y_test_scaled.astype(np.float32)

param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [10, 15, 20, 25, 30]
}

param_combinations = list(product(param_grid['n_estimators'], param_grid['max_depth']))

results_list = []

for i, (n_est, max_d) in enumerate(param_combinations):
    print(f"\n[{i+1}/{len(param_combinations)}] Training model with: "
          f"n_estimators={n_est}, max_depth={max_d}")

    model = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=max_d,
        random_state=2025,
        n_jobs=-1
    )

    model.fit(X_train_scaled, Y_train_scaled.ravel())

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_train_pred_inv = inv_norm_label_ey(y_train_pred.reshape(-1, 1))
    y_test_pred_inv = inv_norm_label_ey(y_test_pred.reshape(-1, 1))
    y_train_true_inv = inv_norm_label_ey(Y_train_scaled.reshape(-1, 1))
    y_test_true_inv = inv_norm_label_ey(Y_test_scaled.reshape(-1, 1))

    mae_train = mean_absolute_error(y_train_true_inv, y_train_pred_inv)
    mae_test = mean_absolute_error(y_test_true_inv, y_test_pred_inv)
    mse_train = mean_squared_error(y_train_true_inv, y_train_pred_inv)
    mse_test = mean_squared_error(y_test_true_inv, y_test_pred_inv)
    r2_train = r2_score(y_train_true_inv, y_train_pred_inv)
    r2_test = r2_score(y_test_true_inv, y_test_pred_inv)

    print(f"training: MAE={mae_train:.2e}, MSE={mse_train:.2e}, R²={r2_train:.4f}")
    print(f"testing: MAE={mae_test:.2e}, MSE={mse_test:.2e}, R²={r2_test:.4f}")

    results_list.append({
        'n_estimators': n_est,
        'max_depth': max_d,
        'train_mae': mae_train,
        'test_mae': mae_test,
        'train_mse': mse_train,
        'test_mse': mse_test,
        'train_r2': r2_train,
        'test_r2': r2_test
    })

df_results = pd.DataFrame(results_list)
df_results.sort_values(by='test_r2', inplace=True)
df_results.to_csv(r'F:\wzy\TrussStructureActivityRelationship\result\RF\ey\manual_gridsearch_results.csv', index=False)
