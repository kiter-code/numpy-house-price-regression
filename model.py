"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    # 1. Создаем копию, чтобы не менять исходный массив
    X_copy = np.array(X, dtype=float)
    
    # 2. Считаем среднее по столбцам (axis=0), игнорируя NaN
    col_means = np.nanmean(X_copy, axis=0)
    
    # 3. Если столбец был полностью из NaN, его среднее равно NaN. 
    # Заменяем такие средние на 0.0 с помощью np.where
    # (где col_means равно NaN, ставим 0.0, иначе оставляем col_means)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    
    # 4. Находим, где в данных находятся NaN
    nan_mask = np.isnan(X_copy)
    
    # 5. Заменяем NaN в данных на средние значения из col_means.
    # np.where работает так: где nan_mask == True, берем из col_means, иначе из X_copy
    result = np.where(nan_mask, col_means, X_copy)
    
    return result

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    IQR = q3 - q1
    lower = q1 - k * IQR
    upper = q3 + k * IQR
    return lower, upper

# Step 3 - clip_columns (not yet solved)
# TODO: implement

# Step 4 - make_ratio_feature (not yet solved)
# TODO: implement

# Step 5 - append_column (not yet solved)
# TODO: implement

# Step 6 - one_hot_encode (not yet solved)
# TODO: implement

# Step 7 - fit_standardizer (not yet solved)
# TODO: implement

# Step 8 - apply_standardizer (not yet solved)
# TODO: implement

# Step 9 - add_bias_column (not yet solved)
# TODO: implement

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

