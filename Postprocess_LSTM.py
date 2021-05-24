import pandas as pd
import numpy as np
import statistics
from DataManagement import DataManager # get_target_LSTM
# from Postprocess import get_rmse_sp

# Postprocess

def computeMetrics(path):

    file = pd.read_csv(path, sep=",")
    mae = file["MAE"]
    mbe = file["MBE"]
    rmse = file["RMSE"]
    rmse_sp = file["RMSE_sp"]

    file_res = pd.read_csv(path_res, sep=",")
    p = file_res["P_act"]
    sp = file_res["P_sp"]

    MAE = []
    MBE = []
    RMSE = []
    RMSE_sp = []
    skill = []
    P_act = []
    P_sp = []

    for line in range(len(file)):
        row_mae = mae[line].strip("[]").split()
        row_mbe = mbe[line].strip("[]").split()
        row_rmse = rmse[line].strip("[]").split()
        row_rmse_sp = rmse_sp[line].strip("[]").split()
        for i, (item_mae, item_mbe, item_rmse, item_rmse_sp) in enumerate(zip(row_mae, row_mbe, row_rmse, row_rmse_sp)):
            row_mae[i] = float(item_mae)
            row_mbe[i] = float(item_mbe)
            row_rmse[i] = float(item_rmse)
            row_rmse_sp[i] = float(item_rmse_sp)
        MAE.append(row_mae)
        MBE.append(row_mbe)
        RMSE.append(row_rmse)
        RMSE_sp.append(row_rmse_sp)

    mae_mean = np.nanmean(MAE, axis=0)
    mbe_mean = np.nanmean(MBE, axis=0)
    rmse_mean = np.nanmean(RMSE, axis=0)
    rmse_sp_mean = np.nanmean(RMSE_sp, axis=0)

    for i in range(len(rmse_mean)):
        val = 1 - (rmse_mean[i]/rmse_sp_mean[i])
        skill.append(val)

    for t in range(len(mae_mean)-1):
        horizon = (t+1) * 5
        print("forecast {}min: MAE: {:.2f} std: {:.1f}, MBE: {:.2f} std: {:.1f}, RMSE: {:.2f} std: {:.1f}, skill: {:.2f} std: {:.1f}".format
              (horizon, mae_mean[t], np.nanstd(np.transpose(MAE)[t]), mbe_mean[t], np.nanstd(np.transpose(MBE)[t]),
               rmse_mean[t], np.nanstd(np.transpose(RMSE)[t]), skill[t] * 100, np.nanstd(np.transpose(skill)[t])))

    return None

# metricLSTM_m2m_layer_2_Input_5_hidden_100_highneg_cr, 100 epochen
# metricLSTM_m2m_layer_2_Input_5_hidden_100_0_cr
# metricLSTM_m2m_layer_2_Input_5_hidden_100_highneg_seas
# metricLSTM_m2m_layer_2_Input_5_hidden_100_0_seas, 100 epochen
# metricLSTM_m2m_layer_2_Input_3_hidden_100_0_seas_gtiBNIRHkd_sc
# metricLSTM_m2m_layer_2_Input_5_hidden_100_0_seas_minmax
# metricLSTM_m2m_layer_2_Input_8_hidden_100_0_seas_shift_Ind
# metricLSTM_m2m_layer_2_Input_8_hidden_100_0_seas_Ind
# LSTM_m2m_Layer_2_Input_5_hidden_150_0_shift_denorm_minmax_El_Az
file = "LSTM_m2m_Layer_2_Input_5_hidden_75_0_shift_denorm_minmax_El_Az_bt15.csv"
path = "D:/TU_Stuttgart/Studienarbeit/LSTM_results/{}".format(file)
path_res = "D:/TU_Stuttgart/Studienarbeit/LSTM_results/result{}".format(file)

computeMetrics(path)

"""window_LSTM = 36
tar = DataManager()
train_y, test_y = tar.get_target_LSTM(window_LSTM)
# rmse_sp = get_rmse_sp

test_y = test_y.drop(test_y.index[test_y["El"] < 15])
tar_cols = test_y.filter(regex="min").columns.tolist()

rmse_sp = []
for step in range(0, len(tar_cols)):
    min = tar_cols[step]
    Pdc_act = test_y["{}".format(min)] * tar.CAPACITY
    error_sp = np.sqrt(np.nanmean((Pdc_act - test_y["Pdc_sp"]).values ** 2))
    rmse_sp.append(error_sp)"""


