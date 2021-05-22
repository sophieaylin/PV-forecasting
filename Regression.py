import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tables
import DataManagement
from sklearn import linear_model, ensemble, neural_network
from sklearn.preprocessing import StandardScaler
from DataManagement import DataManager #get_features, get_target_Pdc

# BEFORE RUN: check if "window_tar" in DataManagement.py fits to "horizon" of Regression.py

feat = DataManager()
dropnight = "true"
features = feat.get_features(window_ft=6, window_tar=6, dropnight=dropnight)
tar = feat.get_target_Pdc(window_tar=6, dropnight=dropnight)

def run_forecast(target,horizon):

    train_x = features[features["dataset"] == "Train"]
    test_x = features[features["dataset"] == "Test"]
    train_y = tar[tar["dataset"] == "Train"]
    test_y = tar[tar["dataset"] == "Test"]

    train_y = train_y.drop('dataset', axis=1)
    test_y = test_y.drop('dataset', axis=1)

    train = train_x.merge(train_y, on="t")
    test = test_x.merge(test_y, on="t")

    # delete night time values
    train = train.drop(train.index[train["El_x"] < 15])
    test = test.drop(test.index[test["El_x"] < 15])

    train = train.dropna()
    test = test.dropna()

    back = window_tar + 1
    Pdc_35_train = train.Pdc_33.shift(periods=back)
    Pdc_35_train = Pdc_35_train[2*back:]
    Pdc_35_test = test.Pdc_33.shift(periods=back)
    Pdc_35_test = Pdc_35_test[2*back:]
    train = train[0:(len(train)-2*back)]
    test = test[0:(len(test)-2*back)]
    train.insert(len(train.columns), column="Pdc_35", value=Pdc_35_train.values)
    test.insert(len(test.columns), column="Pdc_35", value=Pdc_35_test.values)

    """feature_cols = features.filter(regex=target).columns.tolist()"""

    feature_cols_G = features.filter(regex="GHI").columns.tolist()
    feature_cols_B = features.filter(regex="BNI").columns.tolist()
    feature_cols = feature_cols_G + feature_cols_B

    train_X = train[feature_cols + ["Pdc_33"]].values #
    test_X = test[feature_cols + ["Pdc_33"]].values #

    train_Y = train['Pdc_{}'.format(horizon)].values
    test_Y = test['Pdc_{}'.format(horizon)].values

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

        """plt.plot(test_Y)
        plt.plot(test_pred)"""

        # Denormalization of prediction
        train_pred = train_pred * train.ENI #* Capacity
        test_pred = test_pred * test.ENI #* Capacity

        train.insert(train.shape[1], "Pdc_{}_{}".format(target,name), train_pred)
        test.insert(test.shape[1], "Pdc_{}_{}".format(target,name), test_pred)

    # Denormalization of Power
    train_Y = train_Y * train.ENI # * Capacity
    test_Y = test_Y * test.ENI # * Capacity

    # smart persistence forecast
    # uses the shortest Power Output as the current Power value
    tmp = np.squeeze(train["Pdc_sp"].values) * train.ENI #* Capacity
    train.insert(train.shape[1], "Pdc_{}_sp".format(target), tmp)
    tmp = np.squeeze(test["Pdc_sp"].values) * test.ENI #* Capacity
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

# delete all files in folder to avoid having files of a bigger horizon in the folder when testing a smaller
# horizon -> for postprocess
for filename in os.listdir("forecasts"):
    os.remove(os.path.join("forecasts", filename))

target = ["GHI", "BNI"]
horizon = ["5min", "10min", "15min", "20min", "25min", "30min"]
window_tar = len(horizon)

""", "35min", "40min", "45min", "50min", "55min", "60min",
"65min", "70min", "75min", "80min", "85min", "90min", "95min", "100min", "105min", "110min", "115min",
"120min", "125min", "130min", "135min", "140min", "145min", "150min", "155min", "160min", "165min",
"170min", "175min", "180min"]"""

for t in target:
    for h in horizon:
        print("{} Pdc forecast for {}".format(h,t))
        run_forecast(t,h)
