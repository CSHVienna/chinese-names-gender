from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def get_performance(data, col_true, col_pred):
    mae = mean_absolute_error(data[col_true], data[col_pred])
    mse = mean_squared_error(data[col_true], data[col_pred])
    rmse = np.sqrt(mse)
    r2 = r2_score(data[col_true], data[col_pred])
    return mae, mse, rmse, r2

