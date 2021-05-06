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
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from DataManagement import get_features, get_target_LSTM
from treatNaNs import IndicatorNaN, split_sequences

# Trainings- /Test Set

features = get_features()
target = get_target_LSTM() # includes Trainings and Test data of target
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
train = train.between_time("05:35:00", "20:05:00")
test = test.between_time("05:35:00", "20:05:00")

# Include Pdc in Trainingsset       # Normalisert? -nein-
periods_shift = 37
Pdc_35_train = train.Pdc_33.shift(periods=periods_shift)
Pdc_35_train = Pdc_35_train[2*periods_shift:]
Pdc_35_test = test.Pdc_33.shift(periods=periods_shift)
Pdc_35_test = Pdc_35_test[periods_shift*2:] # wieso 2 mal back?
train = train[0:(len(train)-periods_shift*2)]
test = test[0:(len(test)-periods_shift*2)]
train.insert(len(train.columns), column="Pdc_35", value=Pdc_35_train.values)
test.insert(len(test.columns), column="Pdc_35", value=Pdc_35_test.values)

feature_cols_G = features.filter(regex="GHI").columns.tolist()
feature_cols_B = features.filter(regex="BNI").columns.tolist()
feature_cols = feature_cols_G + feature_cols_B
tar_cols = target.filter(regex="Pdc_5min").columns.tolist()

train_X = train[feature_cols + ["Pdc_35"] + ["Ta"] + ["TL"] + ["vw"] + ["AMa"]] #
test_X = test[feature_cols + ["Pdc_35"] + ["Ta"] + ["TL"] + ["vw"] + ["AMa"]] #
train_Y, train_ENI, Pdc_sp_train = train[tar_cols], train[["ENI"]], train[["Pdc_sp"]]
test_Y, test_ENI, Pdc_sp_test = test[tar_cols], test[["ENI"]], test[["Pdc_sp"]]

# nan values
train_X = train_X.fillna(value=0) # train_X = train_X.fillna(value=0)
test_X = test_X.fillna(value=0)
train_Y = train_Y.fillna(value=0)

# Indicator on missing values

"""train_X, test_X, train_Y = IndicatorNaN(train_X, test_X, train_Y)"""

# multiple Imputation
"""train_X_V1 = train_X.fillna(value=random())
test_X_V1 = train_X.fillna(value=random())
train_X_V2 = train_X.fillna(value=random())
test_X_V2 = train_X.fillna(value=random())
train_X_V3 = train_X.fillna(value=random())
test_X_V3 = train_X.fillna(value=random())

train_X = pd.concat([train_X_V1, train_X_V2, train_X_V3], axis=1)
test_X = pd.concat([test_X_V1, test_X_V2, test_X_V3], axis=1)"""

# numpy.ndarray
train_X = train_X.values
test_X = test_X.values
train_Y = train_Y.values
test_Y = test_Y.values

# Scaler !oder MinMaxScaler: auch y gescaled!
scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

# define forecast window and target window for Input shape
seq_dim = 60
window_tar = 36

traindata_stacked = np.hstack((train_X, train_Y, train_ENI, Pdc_sp_train))
testdata_stacked = np.hstack((test_X, test_Y, test_ENI, Pdc_sp_test))
X, y, ENI, Pdc_sp = split_sequences(traindata_stacked, seq_dim, window_tar)
test_X, test_Y, test_ENI, Pdc_sp_test = split_sequences(testdata_stacked, seq_dim, window_tar)

# to torch
X_train = torch.from_numpy(X).float()
X_test = torch.from_numpy(test_X).float()
y_train = torch.from_numpy(y).float()
y_test = torch.from_numpy(test_Y).float()
ENI_train = torch.from_numpy(ENI).float() # train.ENI.values
ENI_test = torch.from_numpy(test_ENI).float() # test.ENI.values
# Pdc_sp_tr = torch.from_numpy(Pdc_sp).float() #Pdc_sp_train.values
# Pdc_sp_te = torch.from_numpy(Pdc_sp_test).float() #Pdc_sp_test.values

def initializeNewModel(input_dim, hidden_dim, layer_dim, output_dim):

    # Initializing LSTM
    # input_dim = number of features
    # hidden_dim = number of hidden layer
    # layer_dim = number of stacked LSTM's
    # output_dim = output horizon

    model = LSTMModel.LSTM(input_dim, hidden_dim, layer_dim, output_dim)

    return model

def trainModel(model, batch_size, seq_dim, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
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

    results = pd.DataFrame()
    metric = pd.DataFrame()

    for epoch in range(epochs):
        # one epoch = one time through dataset
        for step in range(0, int(len(X_train)/batch_size)):

            train_load = X_train[step * batch_size:batch_size * (step+1)].view(-1, seq_dim, X_train.shape[2])
            y = y_train[step * batch_size:(step + 1) * batch_size]

            optimizer.zero_grad()  # clears old gradients (w, r, t)

            y_pred = model(train_load)

            # plot prediction
            # P[horizon] -> 0 == +5min, 1 == +10min ...
            """if step == int(len(X_train)/batch_size):
                P = y.transpose(0,1)
                P_pred = y_pred.transpose(0, 1)
                plt.figure()
                plt.plot(P[0])
                plt.plot(P_pred[0].detach())
                fig.savefig"""

            # Denormalize
            train_ENI = ENI_train[step * batch_size:(step + 1) * batch_size]
            train_pred = y_pred * train_ENI
            observ = y * train_ENI

            # compute loss: criterion RMSE
            RMSE = model.loss(train_pred, observ)

            train_loss.append(RMSE.data)

            if iter % 100 == 99:
                test_len = 100

                # a) begin: per test
                rmse = []
                # rmse_sp = []
                mae = []
                mbe = []
                # a) end

                # for step in range(0, int(len(y_test)/batch_size - 1)):
                for step in range(test_len):
                    """test_batch_rmse = list()
                    test_batch_rmse_sp = list()
                    test_batch_mae = list()
                    test_batch_mbe = list()"""

                    # for Model Input/Ouput
                    test_load = X_test[step * batch_size:(step + 1) * batch_size, :].view(-1, seq_dim, X_test.shape[2])
                    y = y_test[step * batch_size:(step + 1) * batch_size]

                    # for Smart Persistence Model and Denormalization
                    # Pdcsp_test = Pdc_sp_te[step * batch_size:(step + 1) * batch_size]
                    test_ENI = ENI_test[step * batch_size:(step + 1) * batch_size]

                    y_pred = model(test_load)

                    # Denormalize
                    test_pred = y_pred * test_ENI
                    observ = y * test_ENI

                    # compute Metrics
                    error = observ.data.numpy() - test_pred.data.numpy() # .squeeze()
                    # error_sp = observ.data.numpy() - Pdcsp_test.data.numpy()

                    test_batch_rmse = np.sqrt(np.nanmean(error ** 2, axis=0))
                    # test_batch_rmse_sp = np.sqrt(np.nanmean(error_sp ** 2, axis=0))
                    test_batch_mae = np.nanmean(np.abs(error), axis=0)
                    test_batch_mbe = np.nanmean(error, axis=0)

                    # b) begin: pt
                    rmse.append(test_batch_rmse)
                    # rmse_sp.append(test_batch_rmse_sp)
                    mae.append(test_batch_mae)
                    mbe.append(test_batch_mbe)
                    # b) end

                    """test_rmse.append(test_batch_rmse)
                    test_rmse_sp.append(test_batch_rmse_sp)
                    test_mae.append(test_batch_mae)
                    test_mbe.append(test_batch_mbe)"""

                #print('Epoch: {}, Iteration: {}, Train_RMSE: {}, RMSE_sp: {}, Test_RMSE: {}, MAE: {}, MBE: {}'
                 #         .format(epoch, iter, RMSE.data, np.nanmean(test_rmse_sp), np.nanmean(test_rmse), np.nanmean(test_mae),
                  #                np.nanmean(test_mbe)))

                # c) begin: print
                print('Epoch: {}, Iteration: {}, Train_RMSE: {}, Test_RMSE: {}, MAE: {}, MBE: {}'
                      .format(epoch, iter, RMSE.data, np.nanmean(rmse),
                              np.nanmean(mae), np.nanmean(mbe)))
                # c) end

            RMSE.backward()  # computes derivative of loss
            optimizer.step()  # next step based on gradient
            iter += 1

            #torch.save(model, PATH_save)

    # save results of one epoch and Model
    metric.insert(metric.shape[1], "MAE", value=mae)
    metric.insert(metric.shape[1], "MBE", value=mbe)
    metric.insert(metric.shape[1], "RMSE", value=rmse)
    # metric.insert(metric.shape[1], "RMSE_sp", value=rmse_sp)
    metric.to_csv(PATH_save_met)
    torch.save(model, PATH_save)

    print('loss: ', RMSE.item())

    return metric, results

# START
# define which Model to load or name Model to be initialized (layer = ...)

batch_size = 12 # ein Tag mit Nachtstunden 176
layer = 10
hidden = 100
epochs = 10

# _fillna_bignegative'.format(layer, X_train.shape[1])
PATH_load = 'LSTM_Models/LSTM_m2m_Layer_{}_Input_{}_hidden_{}_0_cr_bt12'.format(layer, X_train.shape[2], hidden)
PATH_save = 'LSTM_Models/LSTM_m2m_Layer_{}_Input_{}_hidden_{}_0_cr_bt12'.format(layer,X_train.shape[2], hidden)
PATH_save_met = "D:/TU_Stuttgart/Studienarbeit/LSTM_results/metricLSTM_m2m_layer_{}_Input_{}_hidden_{}_0_cr_bt12.csv"\
    .format(layer, X_train.shape[2], hidden)

# Models
# /LSTM_m2m_Layer_{}_Input_{}_hidden_{}_Indicator
# /LSTM_m2m_Layer_{}_Input_{}_hidden_{}_0
# //LSTM_m2m_Layer_{}_Input_{}_hidden_{}_highneg

# True = load existing Model
# False = initialize new Model !insert number to not overwrite existing Model!
load_model = True

if load_model == False:
    # initialise Model and train IT
    model = initializeNewModel(input_dim=X_train.shape[2], hidden_dim=hidden, layer_dim=layer, output_dim=36)
    print(model)
    test_loss, results = trainModel(model, batch_size, seq_dim, epochs)
    print("finished training")
    print("trained Model: {}".format(model))
else:
    # load Model and train it
    model = torch.load(PATH_load)
    print("Model loaded: {}".format(model))
    test_loss, results = trainModel(model, batch_size, seq_dim, epochs)
    print("finished training")
    print("trained Model: {}".format(model))

# Underfitting -> MBE zu groß(LSTM)/klein(LR)

# abfolge nan/0 nacht tag komisch

# Pdc vor treat nans