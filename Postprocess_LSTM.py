import pandas as pd
import numpy as np
import statistics
from DataManagement import get_target_LSTM

# Postprozess
# NOTE: if horizon is changed

def computeMetrics(path):

    file = pd.read_csv(path, sep=",")
    mae = file["MAE"]
    mbe = file["MBE"]
    rmse = file["RMSE"]
    # rmse_sp = file["RMSE_sp"]

    MAE = []
    MBE = []
    RMSE = []
    # RMSE_sp = []
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
        val = 1 - (RMSE[i]/rmse_sp[i])
        skill.append(val)

    mae_mean = np.nanmean(MAE, axis=0)
    mbe_mean = np.nanmean(MBE, axis=0)
    rmse_mean = np.nanmean(RMSE, axis=0)
    skill_mean = np.nanmean(skill, axis=0)


    for t in range(len(mae_mean)-1):
        horizon = (t+1) * 5
        print("forecast {}min: MAE: {} std: {}, MBE: {} std: {}, RMSE: {} std: {}, skill: {} std: {}".format
              (horizon, mae_mean[t], statistics.stdev(MAE[t]), mbe_mean[t], statistics.stdev(MBE[t]),
               rmse_mean[t], statistics.stdev(RMSE[t]), skill_mean[t] * 100, statistics.stdev(MAE[t])))

    return None

path = "D:/TU_Stuttgart/Studienarbeit/LSTM_results/metricLSTM_m2m_layer_2_Input_41_hidden_100_0_cr.csv"
# metricLSTM_layer_2_Input_1591_hidden_100
# metricLSTM_m2m_layer_2_Input_60_hidden_100_0
# metricLSTM_m2m_layer_2_Input_41_hidden_100_0_cr
# metricLSTM_m2m_layer_10_Input_23_hidden_100_0_cr_bt12
target = get_target_LSTM()
target.insert(target.shape[1], column="key", value = np.array(range(0,len(target))))
train_y = target[target["dataset"] == "Train"]
test_y = target[target["dataset"] == "Test"]

# Pdc_sp_train = train_y["Pdc_sp"]
# Pdc_sp_test = test_y["Pdc_sp"]

tar_cols = test_y.filter(regex="min").columns.tolist()

rmse_sp = []
for step in range(0, len(tar_cols)):
    min = tar_cols[step]
    error_sp = np.sqrt(np.nanmean((test_y["Pdc_sp"] - test_y["{}".format(min)]).values ** 2))
    rmse_sp.append(error_sp)

computeMetrics(path)


