import pandas as pd
import numpy as np
import statistics
from DataManagement import get_target_LSTM

# Postprocess

def computeMetrics(path):

    file = pd.read_csv(path, sep=",")
    mae = file["MAE"]
    mbe = file["MBE"]
    rmse = file["RMSE"]

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

    for i in range(len(RMSE)):
        row = []
        for h in range(len(rmse_sp)):
            val = 1 - (RMSE[i][h]/rmse_sp[h])
            row.append(val)
        skill.append(row)

    mae_mean = np.nanmean(MAE, axis=0)
    mbe_mean = np.nanmean(MBE, axis=0)
    rmse_mean = np.nanmean(RMSE, axis=0)
    skill_mean = np.nanmean(skill, axis=0)


    for t in range(len(mae_mean)-1):
        horizon = (t+1) * 5
        print("forecast {}min: MAE: {:.2f} std: {:.1f}, MBE: {:.2f} std: {:.1f}, RMSE: {:.2f} std: {:.1f}, skill: {:.2f} std: {:.1f}".format
              (horizon, mae_mean[t], np.nanstd(np.transpose(MAE)[t]), mbe_mean[t], np.nanstd(np.transpose(MBE)[t]),
               rmse_mean[t], np.nanstd(np.transpose(RMSE)[t]), skill_mean[t] * 100, np.nanstd(np.transpose(skill)[t])))

    return None

path = "D:/TU_Stuttgart/Studienarbeit/LSTM_results/metricLSTM_m2m_layer_2_Input_8_hidden_100_0_seas_shift_Ind_denorm.csv"
# metricLSTM_m2m_layer_2_Input_5_hidden_100_highneg_cr, 100 epochen
# metricLSTM_m2m_layer_2_Input_5_hidden_100_0_cr
# metricLSTM_m2m_layer_2_Input_5_hidden_100_highneg_seas
# metricLSTM_m2m_layer_2_Input_5_hidden_100_0_seas, 100 epochen
# metricLSTM_m2m_layer_2_Input_3_hidden_100_0_seas_gtiBNIRHkd_sc
# metricLSTM_m2m_layer_2_Input_5_hidden_100_0_seas_minmax
# metricLSTM_m2m_layer_2_Input_8_hidden_100_0_seas_shift_Ind
# metricLSTM_m2m_layer_2_Input_8_hidden_100_0_seas_Ind

target = get_target_LSTM()
target.insert(target.shape[1], column="key", value = np.array(range(0,len(target))))
train_y = target[target["dataset"] == "Train"]
test_y = target[target["dataset"] == "Test"]

test_y = test_y.drop(test_y.index[test_y["El"] < 15])

tar_cols = test_y.filter(regex="min").columns.tolist()

rmse_sp = []
for step in range(0, len(tar_cols)):
    min = tar_cols[step]
    Pdc_act = test_y["{}".format(min)] * test_y["Pdc_33"].max() # test_y["ENI"]
    error_sp = np.sqrt(np.nanmean((Pdc_act - test_y["Pdc_sp"]).values ** 2))
    rmse_sp.append(error_sp)

computeMetrics(path)


