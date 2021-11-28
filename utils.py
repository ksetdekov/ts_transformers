import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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

        this_key_df = pd.DataFrame({'mape': mape, 'smape': smape_v, 'mae': mae, 'r2_value': r2_value}, index=[key])
        results = results.append(this_key_df)

    return results


def get_one_pump_ext_val(inp_df):
    onehot_series_0 = TimeSeries.from_dataframe(inp_df, 'DT', fill_missing_dates=True, freq='5T')

    scaler = Scaler()
    ts = scaler.fit_transform(onehot_series_0)  # scale the whole time series not caring about train/val split...
    filler = MissingValuesFiller()
    ts = filler.transform(ts, method='linear')
    target = ts['DSHORTT1138P2300058']
    # Create training and validation sets:

    covariates = ts[['T1138P6000096', 'T1138P6000315', 'DMIDT1138P4000064',
                     'DSHORTT1138P4000064', 'DLONGT1138P4000064', 'DMIDT1138P2600012',
                     'DSHORTT1138P2600012', 'DLONGT1138P2600012', 'DMIDT1205P2300000',
                     'DSHORTT1205P2300000', 'DLONGT1205P2300000', 'T1205P2300000',
                     'T1138P4000064', 'T1138P2600012', 'T1138P600050', 'T1013P500399']]

    print(len(covariates), len(target))
    return covariates, target


def get_all_pump_ext_val(df_all_pumps):
    val_cov_all, val_target_all = [], []
    for pump_df in df_all_pumps:
        iter_val, iter_val_target = get_one_pump_ext_val(pump_df)
        val_cov_all.append(iter_val)
        val_target_all.append(iter_val_target)
    return val_cov_all, val_target_all


def read_valid(link, encoder):
    data = pd.read_csv(link, index_col=False)
    data.drop(['Unnamed: 0',
               'MIDUPT1138P2300058',
               'SHORTUPT1138P2300058',
               'LONGUPT1138P2300058',
               'DMIDT1138P2300058',
               'DLONGT1138P2300058',
               'UNIXDT', 'UUID'
               ], axis=1, inplace=True)
    data = data.fillna(0)

    transformed = encoder.transform(data.WELL_ID)
    ohe_df = pd.DataFrame(transformed)
    ohe_df.columns = encoder.classes_
    df_oh = pd.concat([data, ohe_df], axis=1).drop(['WELL_ID'], axis=1)

    pump_ids = set(data.WELL_ID.unique())

    df_list = list()

    for pump in pump_ids:
        df_list.append(df_oh[df_oh[pump] == 1])

    val_target, val_cov = get_all_pump_ext_val(df_list)

    return val_target, val_cov


def validate_model(model, validation_target, validation_cov, model_name, val_pumps):
    # create path
    Path(str(Path.cwd()) + "/" + model_name + "/").mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(columns=['mape', 'smape', 'mae', 'r2_value'])
    prediction_and_target_with_pumps = pd.DataFrame(columns=['pump_num',
                                                             'pump_name',
                                                             'true',
                                                             'predicted',
                                                             'resids',
                                                             'model'])
    true_values = []
    predict_values = []

    for part in range(len(validation_target)):
        print(len(validation_target[part]), len(validation_cov[part]))
        pump_name = list(val_pumps)[part]
        print(f'pump {part} named {pump_name}')
        backtest_model_all_pumps_iter = model.historical_forecasts(
            series=validation_target[part],
            past_covariates=validation_cov[part],
            start=0.4,
            retrain=False,
            verbose=True)

        backtest_v3 = validation_target[part]
        val_v3pumptarget_inters = backtest_v3.slice_intersect(backtest_model_all_pumps_iter)

        y_true = val_v3pumptarget_inters.values()
        y_pred = backtest_model_all_pumps_iter.values()

        true_values.append(y_true)
        predict_values.append(y_pred)

        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        smape_v = smape(y_true, y_pred)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        r2_value = r2_score(y_true=y_true, y_pred=y_pred)

        this_key_df = pd.DataFrame({'mape': mape, 'smape': smape_v, 'mae': mae, 'r2_value': r2_value}, index=[part])
        results = results.append(this_key_df)

        this_key_pred_df = pd.DataFrame({'pump_num': part,
                                         'pump_name': pump_name,
                                         'true': y_true.flatten(),
                                         'predicted': y_pred.flatten(),
                                         'resids': (y_true - y_pred).flatten(),
                                         'model': model_name})
        prediction_and_target_with_pumps = prediction_and_target_with_pumps.append(this_key_pred_df)

        val_v3pumptarget_inters[-100:].plot(label='actual')
        backtest_model_all_pumps_iter[-100:].plot(label='pred')

        plt.savefig(model_name + "/pump" + str(part) + "last_100.png", dpi=150)
        plt.show()

        val_v3pumptarget_inters[:1000].plot(label='actual_first1k')
        backtest_model_all_pumps_iter[:1000].plot(label='pred_first1k')

        plt.savefig(model_name + "/pump" + str(part) + "first_1000.png", dpi=150)
        plt.show()

    true_values = np.concatenate(true_values)
    predict_values = np.concatenate(predict_values)
    prediction_and_target_with_pumps.to_csv(model_name + "residual_data.csv")
    return results, true_values, predict_values, prediction_and_target_with_pumps
