import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tables
import DataManagement
from sklearn import linear_model, ensemble, neural_network
from sklearn.preprocessing import StandardScaler
from DataManagement import get_features, get_target, target_Irr_train, target_Irr_test

#Capacity = 20808.66
features = get_features() # training features
out = get_target() # test features
tar_Irr_train = target_Irr_train() # trainings data (kt (=B_0), clear, El)
tar_Irr_test = target_Irr_test() # test data (actual, kt (=B_0), clear, El)

features = features.dropna(axis=0, how="any")
train_Pdc = pd.concat([features.t, features.Pdc_5min, features.Pdc_10min, features.Pdc_15min, features.Pdc_20min,
                 features.Pdc_25min, features.Pdc_30min, features.ENI, features.El], axis=1)
Pdc_train = features.Pdc_33 # nur für max(Pdc)
features = features.drop(['Pdc_5min', 'Pdc_10min', 'Pdc_15min', 'Pdc_20min', 'Pdc_25min', 'Pdc_30min',
                      'ENI', 'El', 'Pdc_33'], axis=1)

out = out.dropna(axis=0, how="any")
tar = pd.concat([out.t, out.Pdc_5min, out.Pdc_10min, out.Pdc_15min, out.Pdc_20min,
                 out.Pdc_25min, out.Pdc_30min, out.ENI, out.El], axis=1)
Pdc_test = out.Pdc_33
test_features = out.drop(['Pdc_5min', 'Pdc_10min', 'Pdc_15min', 'Pdc_20min', 'Pdc_25min', 'Pdc_30min',
                      'ENI', 'El', 'Pdc_33'], axis=1)


def run_forecast(target,horizon):

    train = features.drop('t', axis=1)
    test = test_features.drop('t', axis=1)

    cols = [
        "{}_{}".format(target, horizon),  # actual
        "{}_kt_{}".format(target, horizon),  # clear-sky index
        "{}_clear_{}".format(target, horizon),  # clear-sky model
        "El_{}".format(horizon)]  # solar elevation

    train = train.join(tar_Irr_train, how="inner")
    test = test.join(tar_Irr_test, how="inner")
    feature_cols = features.filter(regex=target).columns.tolist()

    train = train[cols + feature_cols]
    test = test[cols + feature_cols]

    train_X = train[feature_cols].values
    test_X = test[feature_cols].values

    train_y = train['{}_kt_{}'.format(target, horizon)].values
    elev_train = train["El_{}".format(horizon)].values
    elev_test  = test["El_{}".format(horizon)].values

    train_clear = train["{}_clear_{}".format(target,horizon)].values
    test_clear = test["{}_clear_{}".format(target,horizon)].values

    # Ordinary Least-Squares (OLS)
    # Ridge Regression (OLS + L2-regularizer)
    # Lasso (OLS, L1-regularizer)

    models = [["ols", linear_model.LinearRegression()],
             ["ridge", linear_model.RidgeCV(cv=10)],
             ["lasso", linear_model.LassoCV(cv=10, max_iter=10000)]]

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    for name, model in models:
        model.fit(train_X, train_y)
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)

        # convert from kt back to irradiance
        train_pred *= train_clear
        test_pred *= test_clear

        # removes nighttime values (solar elevation < 5)
        train_pred[elev_train < 5] = np.nan
        test_pred[elev_test < 5] = np.nan

        train.insert(train.shape[1], "{}_{}".format(target,name), train_pred)
        test.insert(test.shape[1], "{}_{}".format(target,name), test_pred)

    # smart persistence forecast
    # uses the shortest backward average as the "current" kt value
    tmp = np.squeeze(train["B_{}_kt_0".format(target)].values) * train_clear
    train.insert(train.shape[1], "{}_sp".format(target), tmp)
    tmp = np.squeeze(test["B_{}_kt_0".format(target)].values) * test_clear
    test.insert(test.shape[1], "{}_sp".format(target), tmp)

    # save forecasts
    # only keep essential forecast columns
    cols = train.columns[train.columns.str.startswith("{}".format(target))]
    train = train[cols]
    test = test[cols]

    train.insert(train.shape[1], "dataset", "Train")
    test.insert(test.shape[1], "dataset", "Test")
    df = pd.concat([train, test], axis=0)
    df.insert(df.shape[1], "target", target)
    df.insert(df.shape[1], "horizon", horizon)
    df.to_hdf(os.path.join("forecasts",
                "forecasts_{}_{}.h5".format
                (horizon,target),
                           ), "df", mode="w",
              )

target = ["GHI", "BNI"]
horizon = ["5min", "10min", "15min", "20min", "25min", "30min"]


for t in target:
    for h in horizon:
        print("{} Pdc forecast for {}".format(h,t))
        run_forecast(t,h)
