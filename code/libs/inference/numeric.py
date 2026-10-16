from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

import numpy as np
import pandas as pd


def pval_to_stars(p):
    if p <= 0.001:
        return "***"
    elif p <= 0.01:
        return "**"
    elif p <= 0.05:
        return "*"
    else:
        return "ns"
    
def get_performance(data, col_true, col_pred):
    mae = mean_absolute_error(data[col_true], data[col_pred])
    mse = mean_squared_error(data[col_true], data[col_pred])
    rmse = np.sqrt(mse)
    r2 = r2_score(data[col_true], data[col_pred])
    res = pearsonr(data[col_true], data[col_pred])
    return mae, mse, rmse, r2, res


def get_py_summary(df_py, gender_cols, ambiguity_labels):

    def summarize_group(g):

        ch_pred = g['ch_pred'].to_numpy()

        out = {
            'ch_n': len(ch_pred),
            'ch_pred_all': ch_pred.tolist(),

            'ch_pred_mean': np.mean(ch_pred),
            'ch_pred_std': np.std(ch_pred, ddof=1),
            'ch_pred_min': np.min(ch_pred),
            'ch_pred_max': np.max(ch_pred),
            'ch_pred_delta': ch_pred.max() - ch_pred.min(),

            'ch_pred_q10': np.percentile(ch_pred, 10),
            'ch_pred_q25': np.percentile(ch_pred, 25),
            'ch_pred_q50': np.percentile(ch_pred, 50),
            'ch_pred_q75': np.percentile(ch_pred, 75),
            'ch_pred_q90': np.percentile(ch_pred, 90),
        }

        out['ambiguity_10-90'] = out['ch_pred_q90'] - out['ch_pred_q10']
        out['ambiguity_25-75'] = out['ch_pred_q75'] - out['ch_pred_q25']
        
        for col in gender_cols:

            agg = g[col].iloc[0]

            out[col] = agg

            out[f'{col}_rmse'] = np.sqrt(
                np.mean((ch_pred - agg) ** 2)
            )

        return pd.Series(out)

    df_py_summary = (
        df_py
        .groupby('py_givenname')
        .apply(summarize_group)
        .reset_index()
    )

    df_py_summary.loc[:, 'ambiguity_class'] = pd.cut(df_py_summary['ambiguity_10-90'], bins=[-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, float('inf')], labels=['A','B','C','D','E','F','G'], ordered=True)
    df_py_summary.loc[:, 'ambiguity_label'] = df_py_summary['ambiguity_class'].map({'A': ambiguity_labels[0], 
                                                                                    'B': ambiguity_labels[1], 
                                                                                    'C': ambiguity_labels[2], 
                                                                                    'D': ambiguity_labels[3], 
                                                                                    'E': ambiguity_labels[4], 
                                                                                    'F': ambiguity_labels[5], 
                                                                                    'G': ambiguity_labels[6]})

    return df_py_summary