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
from DataManagement import get_features, get_target_Pdc
from treatNaNs import IndicatorNaN

# Trainings- /Test Set

features = get_features()
target = get_target_Pdc() # includes Trainings and Test data of target
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

# Include Pdc in Trainingsset       # Normalisert?
Pdc_35_train = train.Pdc_33.shift(periods=7)
Pdc_35_train = Pdc_35_train[14:]
Pdc_35_test = test.Pdc_33.shift(periods=7)
Pdc_35_test = Pdc_35_test[14:]
train = train[0:(len(train)-14)]
test = test[0:(len(test)-14)]
train.insert(len(train.columns), column="Pdc_35", value=Pdc_35_train.values)
test.insert(len(test.columns), column="Pdc_35", value=Pdc_35_test.values)

feature_cols_G = features.filter(regex="GHI").columns.tolist()
feature_cols_B = features.filter(regex="BNI").columns.tolist()
feature_cols = feature_cols_G + feature_cols_B
tar_cols = target.filter(regex="min").columns.tolist()

train_X = train[feature_cols + ["Pdc_35"] + ["Ta"] + ["TL"] + ["vw"] + ["AMa"]]
test_X = test[feature_cols + ["Pdc_35"] + ["Ta"] + ["TL"] + ["vw"] + ["AMa"]]
train_Y = train[tar_cols]
test_Y = test[tar_cols]

# nan values
"""train_X = train_X.fillna(value=0.00) # train_X = train_X.fillna(value=0)
test_X = test_X.fillna(value=0.00)
train_Y = train_Y.fillna(value=0.00)
test_Y = test_Y.fillna(value=0.00) # oder muss test_Y unbearbeitet bleiben?"""

# Indicator on missing values

train_X, test_X, train_Y, test_Y = IndicatorNaN(train_X, test_X, train_Y, test_Y)

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

# Scaler
scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

# to torch
X_train = torch.from_numpy(train_X).float()
X_test = torch.from_numpy(test_X).float()
y_train = torch.from_numpy(train_Y).float()
y_test = torch.from_numpy(test_Y).float()
ENI_train = torch.from_numpy(train.ENI.values).float()
ENI_test = torch.from_numpy(test.ENI.values).float()

def initializeNewModel(input_dim, hidden_dim, layer_dim, output_dim):

    # Initializing LSTM
    # input_dim = number of features
    # hidden_dim = number of hidden layer
    # layer_dim = number of stacked LSTM's
    # output_dim = output horizon

    model = LSTMModel.LSTM(input_dim, hidden_dim, layer_dim, output_dim)

    return model

def trainModel(model, batch_size, seq_dim, epochs):

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # Adam(model.parameters(), lr=1e-3)
    # Adagrad(model.parameters(), lr=1e-3)
    # Adadelta(model.parameters(), lr=1e-3

    iter = 0

    train_loss = []
    test_rmse = []
    test_rmse_sp= []
    test_mae = []
    test_mbe = []

    results = pd.DataFrame()
    metric = pd.DataFrame()

    for epoch in range(epochs):
        for step in range(0, int(len(X_train)/batch_size -1)):

            train_load = Variable(X_train[step * batch_size:(step+1) * batch_size, :]).view(-1, seq_dim, X_train.shape[1])
            y = Variable(y_train[step * batch_size:(step+1) * batch_size])

            optimizer.zero_grad()  # clears old gradients (w, r, t)

            y_pred = model(train_load)

            # Denormalize
            train_ENI = Variable(ENI_train[step * batch_size:(step + 1) * batch_size])
            train_pred = torch.zeros(size=(y_pred.shape))
            observ = torch.zeros(size=(y_pred.shape))

            for i in range(0, batch_size):
                train_pred[i] = y_pred[i, :] * train_ENI[i]
                observ[i] = y[i, :] * ENI_train[i]

            # compute loss: criterion RMSE
            RMSE = model.loss(train_pred, observ)

            train_loss.append(RMSE.data)

            if iter % 10 == 9:
                 for step in range(0, int(len(y_test)/batch_size - 1)):     # X_test

                    test_batch_rmse = list()
                    test_batch_rmse_sp = list()
                    test_batch_mae = list()
                    test_batch_mbe = list()

                    test_load = Variable(X_test[step * batch_size:(step + 1) * batch_size, :]).view(-1, seq_dim, X_test.shape[1])
                    y = Variable(y_test[step * batch_size:(step + 1) * batch_size])

                    y_pred = model(test_load)

                    # Denormalize
                    test_ENI = Variable(ENI_test[step * batch_size:(step + 1) * batch_size])
                    test_pred = torch.zeros(size=(y_pred.shape))
                    observ = torch.zeros(size=(y_pred.shape))
                    P_observ_sp = np.zeros(len(y_pred))
                    P_pred_sp = np.zeros(len(y_pred))

                    for i in range(0, batch_size):
                        test_pred[i] = y_pred[i, :] * test_ENI[i]       # (P_pred)test_pred[:, 0] is Pdc_+5min
                        observ[i] = y[i, :] * ENI_test[i]
                        P_observ_sp[i] = observ[i][0]
                        P_pred_sp[i] = test_pred[i][0]

                    # compute Metrics
                    error = observ.data.numpy() - test_pred.data.numpy().squeeze()
                    error_sp = P_observ_sp - P_pred_sp

                    test_batch_rmse = np.sqrt(np.nanmean(error ** 2, axis=0))
                    test_batch_rmse_sp = np.sqrt(np.nanmean(error_sp ** 2, axis=0))
                    test_batch_mae = np.nanmean(np.abs(error), axis=0)
                    test_batch_mbe = np.nanmean(error, axis=0)
                    test_rmse.append(test_batch_rmse)
                    test_rmse_sp.append(test_batch_rmse_sp)
                    test_mae.append(test_batch_mae)
                    test_mbe.append(test_batch_mbe)

                    print('Epoch: {}, Iteration: {}, Train_RMSE: {}, Test_RMSE: {}, MAE: {}, MBE: {}'
                          .format(epoch, iter, RMSE.data, np.mean(test_rmse), np.mean(test_mae),
                                  np.mean(test_mbe)))

            RMSE.backward()  # computes derivative of loss
            optimizer.step()  # next step based on gradient
            iter += 1

    # save results of one epoch and Model
    metric.insert(metric.shape[1], "MAE", value=test_mae)
    metric.insert(metric.shape[1], "MBE", value=test_mbe)
    metric.insert(metric.shape[1], "RMSE", value=test_rmse)
    metric.insert(metric.shape[1], "RMSE_sp", value=test_rmse_sp)
    metric.to_csv("D:/TU_Stuttgart/Studienarbeit/LSTM_results/metricLSTM_layer_{}_Input_{}.csv".format(layer, X_train.shape[1]))
    torch.save(model, PATH_save)

    print('loss: ', RMSE.item())

    return metric, results

# START
# define which Model to load or name Model to be initialized (layer = ...)
layer = 2

batch_size = 176 # ein Tag mit Nachtstunden
seq_dim = 1
epochs = 5

# _fillna_bignegative'.format(layer, X_train.shape[1])
PATH_load = 'LSTM_Models/LSTM_Layer_{}_Input_{}'.format(layer, X_train.shape[1])
PATH_save = 'LSTM_Models/LSTM_Layer_{}_Input_{}'.format(layer,X_train.shape[1])

# True = load existing Model
# False = initialize new Model !insert number to not overwrite existing Model!
load_model = True

if load_model == False:
    # initialise Model and train IT
    model = initializeNewModel(input_dim=X_train.shape[1], hidden_dim=100, layer_dim=layer, output_dim=37)
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


