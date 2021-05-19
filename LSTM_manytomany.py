# MHA: Consider including an environment.yml file in the repository, to make sure we are both using the same versions
# of python and its modules:
# https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
# https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html#conda-requirements
import numpy as np
import torch
import os
import math
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import LSTMModel
from random import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable
from DataManagement import get_features, get_target_LSTM, get_features_LSTM
from treatNaNs import IndicatorNaN, split_sequences

# MHA: see comment on DataManagement.py, you are setting yourself up for errors ---
# BEFORE RUN:
# check window_tar of DataManagement.py, window_tar has to be 36 for Ouput 36
# check if seasonal or chronological Dataset is choosen
# check Model parameters below (layer, Inputs, model_load, hidden, treatnans, batch size, epochs, optimizer, ...)
# ---

# Trainings- /Test Set

features = get_features_LSTM()
target = get_target_LSTM() # includes Trainings and Test data of target

# MHA: The need for this 'key' still escapes me, why not merge by time or original row ID?
features.insert(features.shape[1], column="key", value = np.array(range(0,len(features))))
target.insert(target.shape[1], column="key", value = np.array(range(0,len(target))))
tar = target.drop('t', axis=1)

train_x = features[features["dataset"] == "Train"]
test_x = features[features["dataset"] == "Test"]
train_y = tar[tar["dataset"] == "Train"]
test_y = tar[tar["dataset"] == "Test"]

train_y = train_y.drop('dataset', axis=1)
test_y = test_y.drop('dataset', axis=1)

train = train_x.merge(train_y, on="key")
test = test_x.merge(test_y, on="key")

train.index = pd.to_datetime(train["t"])
test.index = pd.to_datetime(test["t"])
# train = train.between_time("04:00:00", "22:00:00") # mehr Nachtstunden einbeziehen, "05:35:00", "20:05:00"
# test = test.between_time("04:00:00", "22:00:00")

# MHA: You should try to make this module completely blind to the variable names in the data set. ----
# move these lines to DataManagement.py, and let get_features_LSTM and get_target_LSTM give you the already
# "clean" train_X, train_Y, test_X, and test_Y
train_X = train[["GHI"] + ["BNI"] + ["Ta"] + ["TL"]] # + ["kd"] + ["kt"] + ["BNI_kt"] + ["TL"] + ["vw"] + ["AMa"]
test_X = test[["GHI"] + ["BNI"] + ["Ta"] + ["TL"]]
train_Y, train_ENI, Pdc_sp_train, train_denorm = train[["Pdc_5min"]], train[["ENI"]], train[["Pdc_sp"]], train[["Pdc_33"]].max()
test_Y, test_ENI, Pdc_sp_test, test_denorm = test[["Pdc_5min"]], test[["ENI"]], test[["Pdc_sp"]], test[["Pdc_33"]].max()
# ----

# nan values
train_X = train_X.fillna(value=0) # train_X = train_X.fillna(value=0), train_X = train_X.fillna(value=-100000)
test_X = test_X.fillna(value=0)
train_Y = train_Y.fillna(value=0)

# Indicator on missing values
"""train_X, test_X, train_Y = IndicatorNaN(train_X, test_X, train_Y)"""

# numpy.ndarray
train_X = train_X.values
test_X = test_X.values
train_Y = train_Y.values
test_Y = test_Y.values

# Scaler !oder MinMaxScaler: auch y gescaled!
"""scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)"""

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

# MHA: see comments above -----
# define forecast window and target window for Input shape
seq_dim = 60
window_tar = 36

train_ENI = train_ENI.fillna(method="ffill")
test_ENI = test_ENI.fillna(method="ffill")
# -----
traindata_stacked = np.hstack((train_X, train_Y, train_ENI))
testdata_stacked = np.hstack((test_X, test_Y, test_ENI))
X, Y, train_ENI = split_sequences(traindata_stacked, seq_dim, window_tar)
test_X, test_Y, test_ENI = split_sequences(testdata_stacked, seq_dim, window_tar)

# to torch
X_train = torch.from_numpy(X).float()
X_test = torch.from_numpy(test_X).float()
y_train = torch.from_numpy(Y).float()
y_test = torch.from_numpy(test_Y).float()
ENI_train = torch.from_numpy(train_ENI).float()
ENI_test = torch.from_numpy(test_ENI).float()

def initializeNewModel(input_dim, hidden_dim, layer_dim, output_dim):

    # Initializing LSTM
    # input_dim = number of features
    # hidden_dim = number of hidden layer
    # layer_dim = number of stacked LSTM's
    # output_dim = output horizon

    model = LSTMModel.LSTM(input_dim, hidden_dim, layer_dim, output_dim)

    return model

def trainModel(model, batch_size, seq_dim, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # Adam(model.parameters(), lr=1e-3)
    # Adagrad(model.parameters(), lr=1e-3)
    # Adadelta(model.parameters(), lr=1e-3)

    iter = 0

    train_loss = []
    test_rmse = []
    test_rmse_sp = []
    test_mae = []
    test_mbe = []

    metric = pd.DataFrame()

    for epoch in range(epochs):
        # one epoch = one time through dataset
        for step in range(0, int(len(X_train)/batch_size)):

            train_load = X_train[step * batch_size:batch_size * (step+1)].view(-1, seq_dim, X_train.shape[2])
            y = y_train[step * batch_size:(step + 1) * batch_size]

            optimizer.zero_grad()  # clears old gradients (w, r, t)

            y_pred = model(train_load)

            # Denormalize
            """train_ENI = ENI_train[step * batch_size:(step + 1) * batch_size]
            train_pred = y_pred * train_ENI # y_pred.detach() * train_denorm
            observ = y * train_ENI # y.detach() * train_denorm"""

            # compute loss: criterion RMSE
            RMSE = model.loss(y_pred, y) # train_pred, observ y_pred, y

            train_loss.append(RMSE.data)

            if iter % 100 == 99:
                test_len = 455

                rmse = []
                mae = []
                mbe = []

                # for step in range(0, int(len(y_test)/batch_size - 1)):
                for step in range(test_len):

                    # for Model Input/Ouput
                    test_load = X_test[step * batch_size:(step + 1) * batch_size, :].view(-1, seq_dim, X_test.shape[2])
                    y = y_test[step * batch_size:(step + 1) * batch_size]

                    # for Denormalization
                    # test_ENI = ENI_test[step * batch_size:(step + 1) * batch_size]

                    y_pred = model(test_load)

                    # Denormalize
                    test_pred = y_pred.detach() * test_denorm.values # test_ENI
                    observ = y * test_denorm.values # test_ENI

                    # plot prediction
                    # P[horizon] -> 0 == +5min, 1 == +10min ...

                    """if step == int(len(X_test)/batch_size): # & epoch == epochs - 1
                        P = y.transpose(0,1)
                        P_pred = y_pred.transpose(0, 1)
                        fig = plt.figure()
                        fig = plt.plot(P[0])
                        fig = plt.plot(P_pred[0].detach())
                        plt.savefig(PATH_fig)"""

                    # compute Metrics
                    error = observ.data.numpy() - test_pred.data.numpy() # .squeeze()

                    test_batch_rmse = np.sqrt(np.nanmean(error ** 2, axis=0))
                    test_batch_mae = np.nanmean(np.abs(error), axis=0)
                    test_batch_mbe = np.nanmean(error, axis=0)

                    rmse.append(test_batch_rmse)
                    mae.append(test_batch_mae)
                    mbe.append(test_batch_mbe)

                print('Epoch: {}, Iteration: {}, Train_RMSE: {}, Test_RMSE: {}, MAE: {}, MBE: {}'
                      .format(epoch, iter, RMSE.data, np.nanmean(rmse),
                              np.nanmean(mae), np.nanmean(mbe)))

            RMSE.backward()  # computes derivative of loss
            optimizer.step()  # next step based on gradient
            iter += 1

            #torch.save(model, PATH_save)

    # save results of one epoch and Model
    metric.insert(metric.shape[1], "MAE", value=mae)
    metric.insert(metric.shape[1], "MBE", value=mbe)
    metric.insert(metric.shape[1], "RMSE", value=rmse)
    metric.to_csv(PATH_save_met)
    torch.save(model, PATH_save)

    print('loss: ', RMSE.item())

    return metric

# START
# define which Model to load or name Model to be initialized (layer = ...)

batch_size = 30
layer = 2
hidden = 100
epochs = 70

# CHECK if train/test Set seasonal or chronological
PATH_load = 'LSTM_Models/LSTM_m2m_Layer_{}_Input_{}_hidden_{}_0_seas_shift_night_denorm_minmax'.format(layer, X_train.shape[2], hidden)
PATH_save = 'LSTM_Models/LSTM_m2m_Layer_{}_Input_{}_hidden_{}_0_seas_shift_night_denorm_minmax'.format(layer,X_train.shape[2], hidden)
PATH_save_met = "D:/TU_Stuttgart/Studienarbeit/LSTM_results/metricLSTM_m2m_layer_{}_Input_{}_hidden_{}_0_seas_shift_night_denorm_minmax.csv"\
    .format(layer, X_train.shape[2], hidden)
PATH_fig = 'D:/TU_Stuttgart/Studienarbeit/LSTM_results/figure'
# "/zhome/academic/HLRS/hlrs/hpcsayli/run/LSTM_results/metricLSTM_layer_{}_Input_{}_hidden_{}.csv"

# Models
# Datasets with GHI_kt, BNI_kt, Ta, TL, vw, AMa
# /LSTM_m2m_Layer_{}_Input_{}_hidden_{}_0_cr
# /LSTM_m2m_Layer_{}_Input_{}_hidden_{}_highneg_cr
# /LSTM_m2m_Layer_{}_Input_{}_hidden_{}_0_seas
# /LSTM_m2m_Layer_{}_Input_{}_hidden_{}_highneg_seas
# Datasets
# LSTM_m2m_Layer_{}_Input_{}_hidden_{}_0_seas_gtiBNIRHkd_sc

# True = load existing Model
# False = initialize new Model !insert number to not overwrite existing Model!
load_model = False

if load_model == False:
    # initialise Model and train IT
    model = initializeNewModel(input_dim=X_train.shape[2], hidden_dim=hidden, layer_dim=layer, output_dim=36)
    print(model)
    test_loss = trainModel(model, batch_size, seq_dim, epochs)
    print("finished training")
    print("trained Model: {}".format(model))
else:
    # load Model and train it
    model = torch.load(PATH_load)
    print("Model loaded: {}".format(model))
    test_loss = trainModel(model, batch_size, seq_dim, epochs)
    print("finished training")
    print("trained Model: {}".format(model))



# multiple Imputation
"""train_X_V1 = train_X.fillna(value=random())
test_X_V1 = train_X.fillna(value=random())
train_X_V2 = train_X.fillna(value=random())
test_X_V2 = train_X.fillna(value=random())
train_X_V3 = train_X.fillna(value=random())
test_X_V3 = train_X.fillna(value=random())

train_X = pd.concat([train_X_V1, train_X_V2, train_X_V3], axis=1)
test_X = pd.concat([test_X_V1, test_X_V2, test_X_V3], axis=1)

rng = np.arange(start=train["kt"].min(), stop=train["kt"].max())
train["kt"].hist(bin=rng.values)"""
