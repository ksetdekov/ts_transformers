import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def score_model(dict_valid):
    results = pd.DataFrame(columns=['mape', 'smape', 'mae', 'r2_value'])
    for key in dict_valid:
        y_true = dict_valid[key]['true']
        y_pred = dict_valid[key]['prediction']
        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        smape_v = smape(y_true, y_pred)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        r2_value = r2_score(y_true=y_true, y_pred=y_pred)
        
        this_key_df = pd.DataFrame({'mape':mape, 'smape':smape_v, 'mae':mae, 'r2_value':r2_value}, index=[key])
        results = results.append(this_key_df)

    return results
        
