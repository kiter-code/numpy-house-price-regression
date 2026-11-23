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

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    return np.column_stack([X, col])

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)
    uniques = np.unique(labels)
    mask = (labels[:, None] == uniques[None, :])
    return mask.astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X, axis=0) # среднее по столбцам
    std = np.std(X, axis=0) # стандартное отклонение по столбцам
    std = np.where(std == 0, 1.0, std) 
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    N = X.shape[0] # Узнаем количество строк (N)
    N1 = np.ones((N, 1)) # Создаем столбец из единиц формы (N, 1)
    return np.hstack((N1, X)) # Приклеиваем его слева к матрице X (горизонтально)

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
   # 1. Создаем изолированный генератор случайных чисел с нашим seed
    rng = np.random.RandomState(seed)
    
    # 2. Генерируем перемешанный массив чисел от 0 до n_samples-1
    # Метод permutation(n) создает массив от 0 до n-1 и перемешивает его
    shuffled_indices = rng.permutation(n_samples)
    
    return shuffled_indices

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # 3. Делаем срезы
    train_idx = indices[0:n_train]
    val_idx = indices[n_train : n_train + n_val]
    
    # 4. Тест — это всё, что осталось до конца массива
    test_idx = indices[n_train + n_val:]
    
    return train_idx, val_idx, test_idx

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    X_sub = X[indices]
    y_sub = y[indices]
    return X_sub, y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A = X.T @ X
    b = X.T @ y
    return np.linalg.solve(A, b)

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return np.dot(X, theta)

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return float(np.mean(np.abs(y_true - y_pred))) # MAE

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Сумма квадратов остатков (residuals)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Общая сумма квадратов
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Защита от деления на ноль
    if ss_tot == 0:
        return 0.0
        
    return float(1.0 - (ss_res / ss_tot))

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r = y_true - y_pred
    
    return {
        'mean': float(np.mean(r)),
        'std': float(np.std(r)),  # ddof=0 по умолчанию
        'median_abs': float(np.median(np.abs(r)))
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X_imp = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X_imp, iqr_k)
    return clip_columns(X_imp, lower, upper)

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]
    # 2. Формируем признак отношения
    ratio = make_ratio_feature(numerator, denominator)
    
    # 3. Добавляем столбец к исходной матрице
    X_extended = append_column(X_num, ratio)
    
    # 4. Добавляем категориальные признаки, если они есть
    if cat_labels is not None:
        cat_block = one_hot_encode(cat_labels)
        X_extended = np.hstack([X_extended, cat_block])
        
    return X_extended

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # 1. Получаем количество строк
    n_samples = X.shape[0]
    
    # 2. Генерируем перемешанные индексы
    indices = make_shuffled_indices(n_samples, seed)
    
    # 3. Разделяем индексы согласно пропорциям
    train_idx, val_idx, test_idx = partition_indices(indices, train_ratio, val_ratio)
    
    # 4. Формируем выборки с помощью subset_xy
    X_train, y_train = subset_xy(X, y, train_idx)
    X_val, y_val = subset_xy(X, y, val_idx)
    X_test, y_test = subset_xy(X, y, test_idx)
    
    # 5. Возвращаем словарь
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # 1. Вычисляем среднее и стандартное отклонение ТОЛЬКО на train
    mean, std = fit_standardizer(splits['X_train'])
    
    # 2. Применяем преобразование и добавляем столбец смещения
    X_train_std = add_bias_column(apply_standardizer(splits['X_train'], mean, std))
    X_val_std = add_bias_column(apply_standardizer(splits['X_val'], mean, std))
    X_test_std = add_bias_column(apply_standardizer(splits['X_test'], mean, std))
    
    # 3. Собираем новый словарь
    std_splits = {
        'X_train': X_train_std,
        'y_train': splits['y_train'],  # y не меняется
        'X_val': X_val_std,
        'y_val': splits['y_val'],
        'X_test': X_test_std,
        'y_test': splits['y_test']
    }
    
    return std_splits, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r_squared(y_true, y_pred),
        'residual_summary': residual_summary(y_true, y_pred)
    }

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

