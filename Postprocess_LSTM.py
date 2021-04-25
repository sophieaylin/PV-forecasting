import numpy as np
import pandas as pd

# Postprozess
# which file "resultLSTM_Epoch_10 should be loaded, either res or met

def computeMetrics(path):

    file = pd.read_csv(path, sep=",")
    mae = file["MAE"]
    mbe = file["MBE"]
    rmse = file["RMSE"]
    rmse_sp = file["RMSE_sp"]

    MAE = []
    MBE = []
    RMSE = []
    skill = []

    for line in range(len(file)):
        row_mae = mae[line].strip("[]").split()
        row_mbe = mbe[line].strip("[]").split()
        row_rmse = rmse[line].strip("[]").split()
        for i, (item_mae, item_mbe, item_rmse) in enumerate(zip(row_mae, row_mbe, row_rmse)):
            row_mae[i] = float(item_mae)
            row_mbe[i] = float(item_mbe)
            row_rmse[i] = float(item_rmse)
        MAE.append(row_mae)
        MBE.append(row_mbe)
        RMSE.append(row_rmse)

    for i in range(len(rmse_sp)):
        row = 1 - (RMSE[i] / rmse_sp[i])
        skill.append(row)

    mae_mean = np.mean(MAE, axis=0)
    mbe_mean = np.mean(MBE, axis=0)
    rmse_mean = np.mean(RMSE, axis=0)
    skill_mean = np.mean(skill, axis=0)


    for t in range(len(mae_mean)-1):
        horizon = (t+1) * 5
        print("forecast {}min: MAE: {}, MBE: {}, RMSE: {}, skill: {}".format(horizon, mae_mean[t], mbe_mean[t], rmse_mean[t], skill_mean[t] * 100))

    return None

path = "D:/TU_Stuttgart/Studienarbeit/LSTM_results/metricLSTM_layer_2_Input_137.csv"
computeMetrics(path)
