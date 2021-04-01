import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tables
import DataManagement
from sklearn import linear_model, ensemble, neural_network
from sklearn.preprocessing import StandardScaler
from DataManagement import get_features, get_target

#Capacity = 20808.66
features = get_features() # only trainings data
out = get_target() # only test data

features = features.dropna()
out = out.dropna()

# split in trainings features and target and test features and target
features = features.dropna(axis=0, how="any")
train_Pdc = pd.concat([features.t, features.Pdc_0min, features.Pdc_5min, features.Pdc_10min, features.Pdc_15min, features.Pdc_20min,
                 features.Pdc_25min, features.Pdc_30min, features.ENI, features.El], axis=1)
Pdc_train = features.Pdc_33 # Pdc_33 oder Pdcmean
features = features.drop(['Pdc_0min', 'Pdc_5min', 'Pdc_10min', 'Pdc_15min', 'Pdc_20min', 'Pdc_25min', 'Pdc_30min',
                      'ENI', 'El', 'Pdc_33'], axis=1)

out = out.dropna(axis=0, how="any")
tar = pd.concat([out.t, out.Pdc_0min, out.Pdc_5min, out.Pdc_10min, out.Pdc_15min, out.Pdc_20min,
                 out.Pdc_25min, out.Pdc_30min, out.ENI, out.El], axis=1)
Pdc_test = out.Pdc_33
test_features = out.drop(['Pdc_0min', 'Pdc_5min', 'Pdc_10min', 'Pdc_15min', 'Pdc_20min', 'Pdc_25min', 'Pdc_30min',
                      'ENI', 'El', 'Pdc_33'], axis=1)
#tar = tar.drop(tar.columns[tar.columns.str.startswith(("B_", "L_", "V_"))], axis=1)


def run_forecast(target,horizon):

    train = features.drop('t', axis=1)
    test = test_features.drop('t', axis=1)
    train_y = train_Pdc.drop('t', axis=1)
    test_y = tar.drop('t', axis=1)

    feature_cols = features.filter(regex=target).columns.tolist()

    train = train[feature_cols] # cols +
    test = test[feature_cols] # cols +

    # Include Pdc in Trainingsset
    Pdc_35_train = Pdc_train.shift(periods=7)
    Pdc_35_train = Pdc_35_train[14:]
    Pdc_35_test = Pdc_test.shift(periods=7)
    Pdc_35_test = Pdc_35_test[14:]
    train = train[0:(len(train)-14)]
    test = test[0:(len(test)-14)]
    train_y = train_y[0:(len(train_y)-14)]
    test_y = test_y[0:(len(test_y)-14)]
    train.insert(len(train.columns), column="Pdc_35", value=Pdc_35_train.values)
    test.insert(len(test.columns), column="Pdc_35", value=Pdc_35_test.values)

    train_X = train[feature_cols  + ["Pdc_35"]].values #
    test_X = test[feature_cols + ["Pdc_35"]].values #

    train_Y = train_y['Pdc_{}'.format(horizon)].values
    test_Y = test_y['Pdc_{}'.format(horizon)].values

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
        model.fit(train_X, train_Y)
        train_pred = model.predict(train_X)
        test_pred = model.predict(test_X)

        train_pred = train_pred * train_y.ENI #* Capacity
        test_pred = test_pred * test_y.ENI #* Capacity

        train.insert(train.shape[1], "Pdc_{}_{}".format(target,name), train_pred)
        test.insert(test.shape[1], "Pdc_{}_{}".format(target,name), test_pred)

    train_Y = train_Y * train_y.ENI # * Capacity
    test_Y = test_Y * test_y.ENI # * Capacity

    # smart persistence forecast
    # uses the shortest Power Output as the current Power value
    tmp = np.squeeze(train_y["Pdc_0min"].values) * train_y.ENI #* Capacity
    train.insert(train.shape[1], "Pdc_{}_sp".format(target), tmp)
    tmp = np.squeeze(test_y["Pdc_0min"].values) * test_y.ENI #* Capacity
    test.insert(test.shape[1], "Pdc_{}_sp".format(target), tmp)

    # save actual values to compare with predicted values in Postprozess
    train.insert(train.shape[1], "Pdc_{}_actual".format(target), train_Y)
    test.insert(test.shape[1], "Pdc_{}_actual".format(target), test_Y)

    # save forecasts
    # only keep essential forecast columns
    cols = train.columns[train.columns.str.startswith("Pdc_{}".format(target))]
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
