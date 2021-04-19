import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as plt


def get_data(deep_copy = True):
    return data.copy(deep_copy)

def get_features (deep_copy = True):
    # data have to be stored in a pandas DataFrame
    # for Input -> X (B = backward Average, L = lagged Average, V = Variability)
    # build feature Normalization

    # Trainings features

    #GHI_kt = GHI.div(ENI.mul(np.sin(np.deg2rad(El))))
    #gti_kt = gti30t187a_train.div(ENI_train)
    BNI_kt = BNI_train.div(ENI_train)

    B_BNI_kt_1 = BNI_kt
    B_BNI_kt = BNI_kt
    B_GHI_kt_1 = GHI_KT_train #  gti_kt
    B_GHI_kt = GHI_KT_train  #  gti_kt

    featuresB_GHI = pd.DataFrame()
    featuresB_BNI = pd.DataFrame()
    featuresL_GHI = pd.DataFrame()
    featuresL_BNI = pd.DataFrame()
    featuresV_GHI = pd.DataFrame()
    featuresV_BNI = pd.DataFrame()
    featuresL_GHI.insert(0, column='L_GHI_kt_0', value=B_GHI_kt)
    featuresL_BNI.insert(0, column='L_BNI_kt_0', value=B_BNI_kt)

    for col in range(0, window_ft): # ,shift; zip(range(0, window), range(-1, -(window+1), -1))
        featuresB_GHI.insert(col, column='B_GHI_kt_%i' % col, value=B_GHI_kt)
        featuresB_BNI.insert(col, column='B_BNI_kt_%i' % col, value=B_BNI_kt)
        BGHI_shift = B_GHI_kt_1.shift(periods=col+1)
        BBNI_shift = B_BNI_kt_1.shift(periods=col+1)
        clmn = col + 1
        featuresL_GHI.insert(clmn, column='L_GHI_kt_%i' % clmn, value=BGHI_shift)
        featuresL_BNI.insert(clmn, column='L_BNI_kt_%i' % clmn, value=BBNI_shift)
        B_GHI_kt = featuresL_GHI.mean(axis=1)
        B_BNI_kt = featuresL_BNI.mean(axis=1)

    featuresL_GHI = featuresL_GHI.drop('L_GHI_kt_6', axis=1)
    featuresL_BNI = featuresL_BNI.drop('L_BNI_kt_6', axis=1)
    GHI_kt_mean = featuresB_GHI["B_GHI_kt_5"].to_frame()
    BNI_kt_mean = featuresB_BNI["B_BNI_kt_5"].to_frame()

    for col in range(0, window_ft):
        delta_kt_GHI = featuresL_GHI.iloc[:, 0:col+1].sub(GHI_kt_mean.values.reshape(len(GHI_kt_mean), col+1)).pow(2)
        delta_kt_BNI = featuresL_BNI.iloc[:, 0:col+1].sub(BNI_kt_mean.values.reshape(len(BNI_kt_mean), col+1)).pow(2)
        GHI_kt_mean.insert(col+1, column="times_%i" % col, value=GHI_kt_mean.B_GHI_kt_5.values)
        BNI_kt_mean.insert(col+1, column="times_%i" % col, value=BNI_kt_mean.B_BNI_kt_5.values)
        V_GHI_kt = pd.Series(np.sqrt(np.divide(delta_kt_GHI.sum(axis=1), col+1)))
        V_BNI_kt = pd.Series(np.sqrt(np.divide(delta_kt_BNI.sum(axis=1), col+1)))
        featuresV_GHI.insert(col, column='V_GHI_kt_%i' % col, value=V_GHI_kt)
        featuresV_BNI.insert(col, column='V_BNI_kt_%i' % col, value=V_BNI_kt)

    features_train = pd.concat([time_train, featuresB_GHI, featuresB_BNI, featuresV_GHI, featuresV_BNI,
                          featuresL_GHI, featuresL_BNI, Ta_train, TL_train, vw_train, AMa_train], axis=1)

    features_train.insert(features_train.shape[1], "dataset", "Train")
    features_train = features_train[1:len(features_train)]

    # Test features

    #gti_kt = gti30t187a_test.div(ENI_test)
    BNI_kt = BNI_test.div(ENI_test)

    B_BNI_kt_1 = BNI_kt
    B_BNI_kt = BNI_kt
    B_GHI_kt_1 = GHI_KT_test  # gti_kt
    B_GHI_kt = GHI_KT_test  # gti_kt

    featuresB_GHI = pd.DataFrame()
    featuresB_BNI = pd.DataFrame()
    featuresL_GHI = pd.DataFrame()
    featuresL_BNI = pd.DataFrame()
    featuresV_GHI = pd.DataFrame()
    featuresV_BNI = pd.DataFrame()
    featuresL_GHI.insert(0, column='L_GHI_kt_0', value=B_GHI_kt)
    featuresL_BNI.insert(0, column='L_BNI_kt_0', value=B_BNI_kt)

    for col, shift in zip(range(0, window_ft), range(-1, -(window_ft + 1), -1)):
        featuresB_GHI.insert(col, column='B_GHI_kt_%i' % col, value=B_GHI_kt)
        featuresB_BNI.insert(col, column='B_BNI_kt_%i' % col, value=B_BNI_kt)
        BGHI_shift = B_GHI_kt_1.shift(periods=col + 1)
        BBNI_shift = B_BNI_kt_1.shift(periods=col + 1)
        clmn = col + 1
        featuresL_GHI.insert(clmn, column='L_GHI_kt_%i' % clmn, value=BGHI_shift)
        featuresL_BNI.insert(clmn, column='L_BNI_kt_%i' % clmn, value=BBNI_shift)
        B_GHI_kt = featuresL_GHI.mean(axis=1)
        B_BNI_kt = featuresL_BNI.mean(axis=1)

    featuresL_GHI = featuresL_GHI.drop('L_GHI_kt_6', axis=1)
    featuresL_BNI = featuresL_BNI.drop('L_BNI_kt_6', axis=1)
    GHI_kt_mean = featuresB_GHI["B_GHI_kt_5"].to_frame()
    BNI_kt_mean = featuresB_BNI["B_BNI_kt_5"].to_frame()

    for col in range(0, window_ft):
        delta_kt_GHI = featuresL_GHI.iloc[:, 0:col + 1].sub(GHI_kt_mean.values.reshape(len(GHI_kt_mean), col + 1)).pow(
            2)
        delta_kt_BNI = featuresL_BNI.iloc[:, 0:col + 1].sub(BNI_kt_mean.values.reshape(len(BNI_kt_mean), col + 1)).pow(
            2)
        GHI_kt_mean.insert(col + 1, column="times_%i" % col, value=GHI_kt_mean.B_GHI_kt_5.values)
        BNI_kt_mean.insert(col + 1, column="times_%i" % col, value=BNI_kt_mean.B_BNI_kt_5.values)
        V_GHI_kt = pd.Series(np.sqrt(np.divide(delta_kt_GHI.sum(axis=1), col + 1)))
        V_BNI_kt = pd.Series(np.sqrt(np.divide(delta_kt_BNI.sum(axis=1), col + 1)))
        featuresV_GHI.insert(col, column='V_GHI_kt_%i' % col, value=V_GHI_kt)
        featuresV_BNI.insert(col, column='V_BNI_kt_%i' % col, value=V_BNI_kt)

    features_test = pd.concat([time_test, featuresB_GHI, featuresB_BNI, featuresV_GHI, featuresV_BNI,
                          featuresL_GHI, featuresL_BNI, Ta_test, TL_test, vw_test, AMa_test], axis=1)

    features_test.insert(features_test.shape[1], "dataset", "Test")
    features_test = features_test[1:len(features_test)]

    features = pd.concat([features_train, features_test], axis=0)

    return features.copy(deep_copy)

def get_target_Pdc (deep_copy = True):
    # for Output -> Y (Power)
    # Train target

    Pdc_shift = pd.DataFrame()
    Pdc_norm = Pdc_train.div(ENI_train)

    for col in range(0, window_tar + 1):
        Pdc_shift.insert(col, column='Pdc_{}min'.format(delta * (col)), value=Pdc_norm)
        Pdc_norm = Pdc_norm.shift(periods=-1)

    target_train = pd.concat([time_train, Pdc_shift, ENI_train, El_train, Pdc_train], axis=1)
    target_train.insert(target_train.shape[1], "dataset", "Train")
    target_train.shift(periods=-1)
    target_train = target_train[0:len(target_train) - 1]

    # Test target

    Pdc_shift = pd.DataFrame()
    Pdc_norm = Pdc_test.div(ENI_test)

    for col in range(0, window_tar + 1):
        Pdc_shift.insert(col, column='Pdc_{}min'.format(delta * (col)), value=Pdc_norm)
        Pdc_norm = Pdc_norm.shift(periods=-1)

    target_test = pd.concat([time_test, Pdc_shift, ENI_test, El_test, Pdc_test], axis=1)
    target_test.insert(target_test.shape[1], "dataset", "Test")
    target_test.shift(periods=-1)
    target_test = target_test[0:len(target_test) - 1]

    target = pd.concat([target_train, target_test], axis=0)

    return target.copy(deep_copy)

def get_target_Irr(deep_copy = True):
    # for Output -> Y (Irradiance, kt)
    # Train target

    global BNI_train, GHI_train, El_train, CSGHI_train, CSBNI_train, GHI_KT_train, ENI_train

    target_Irr_train = pd.DataFrame()
    BNI_kt_train = BNI_train.div(ENI_train)

    for blk in range(0, window_tar):
        block = pd.DataFrame()
        block.insert(0, column="GHI_{}min".format(delta * (blk + 1)), value=GHI_train)
        block.insert(1, column="BNI_{}min".format(delta * (blk + 1)), value=BNI_train)
        block.insert(2, column="GHI_clear_{}min".format(delta * (blk + 1)), value=CSGHI_train)
        block.insert(3, column="BNI_clear_{}min".format(delta * (blk + 1)), value=CSBNI_train)
        block.insert(4, column="GHI_kt_{}min".format(delta * (blk + 1)), value=GHI_KT_train)
        block.insert(5, column="BNI_kt_{}min".format(delta * (blk + 1)), value=BNI_kt_train)
        block.insert(6, column="El_{}min".format(delta * (blk + 1)), value=El_train)
        block.insert(7, column="ENI_{}min".format(delta * (blk + 1)), value=ENI_train)
        target_Irr_train = pd.concat([target_Irr_train, block], axis=1)
        GHI_train = GHI_train.shift(periods=-1)
        BNI_train = BNI_train.shift(periods=-1)
        CSGHI_train = CSGHI_train.shift(periods=-1)
        CSBNI_train = CSBNI_train.shift(periods=-1)
        GHI_KT_train = GHI_KT_train.shift(periods=-1)
        BNI_kt_train = BNI_kt_train.shift(periods=-1)
        El_train = El_train.shift(periods=-1)
        ENI_train = ENI_train.shift(periods=-1)

    target_Irr_train.insert(target_Irr_train.shape[1], "dataset", "Train")
    target_Irr_train.shift(periods=-1)
    target_Irr_train = target_Irr_train[0:len(target_Irr_train)-1]

    # Test target

    global BNI_test, GHI_test, El_test, CSGHI_test, CSBNI_test, GHI_KT_test, ENI_test

    target_Irr_test = pd.DataFrame()
    BNI_kt_test = BNI_test.div(ENI_test)

    for blk in range(0, window_tar):
        block = pd.DataFrame()
        block.insert(0, column="GHI_{}min".format(delta * (blk + 1)), value=GHI_test)
        block.insert(1, column="BNI_{}min".format(delta * (blk + 1)), value=BNI_test)
        block.insert(2, column="GHI_clear_{}min".format(delta * (blk + 1)), value=CSGHI_test)
        block.insert(3, column="BNI_clear_{}min".format(delta * (blk + 1)), value=CSBNI_test)
        block.insert(4, column="GHI_kt_{}min".format(delta * (blk + 1)), value=GHI_KT_test)
        block.insert(5, column="BNI_kt_{}min".format(delta * (blk + 1)), value=BNI_kt_test)
        block.insert(6, column="El_{}min".format(delta * (blk + 1)), value=El_test)
        block.insert(7, column="ENI_{}min".format(delta * (blk + 1)), value=ENI_test)
        target_Irr_test = pd.concat([target_Irr_test, block], axis=1)
        GHI_test = GHI_test.shift(periods=-1)
        BNI_test = BNI_test.shift(periods=-1)
        CSGHI_test = CSGHI_test.shift(periods=-1)
        CSBNI_test = CSBNI_test.shift(periods=-1)
        GHI_KT_test = GHI_KT_test.shift(periods=-1)
        BNI_kt_test = BNI_kt_test.shift(periods=-1)
        El_test = El_test.shift(periods=-1)
        ENI_test = ENI_test.shift(periods=-1)

    target_Irr_test.insert(target_Irr_test.shape[1], "dataset", "Test")
    target_Irr_test.shift(periods=-1)
    target_Irr_test = target_Irr_test[0:len(target_Irr_test) - 1]

    target = pd.concat([target_Irr_train, target_Irr_test], axis=0)

    return target.copy(deep_copy)


filename = 'Daten/PVAMM_201911-202011_PT5M_merged.csv'
data = pd.read_csv(filename)
data_min = data

CAPACITY = 20808.66
window_ft = 6 # time window for feature generation (without nan; measurements from 06:40 - 15:40 == window of 108)
window_tar = 6 # time window for forecast horizon !adjust horizon respectively in Regression.py!
delta = 5  # step size [min]

# removing 0 Irradiation, night times (El<5 degrees) and nan
"""for Irr in ['GHI', 'DHI', 'gti30t187a', 'ENI']:
    data_min = data_min.drop(data_min.index[data_min[Irr] == 0])

data_min = data_min.drop(data_min.index[data_min["El"] < 5])

data_min = data_min.dropna(subset=['GHI', 'BNI', 'DHI', 'gti30t187a', 'ENI', 'El'])"""

# Trainingsset: "big" year, "small" year / chronologically

first = ["Sep", "Dec", "Mar", "Jun"]
second = ["Oct", "Jan", "Apr", "Jul"]
third = ["Nov", "Feb", "May", "Aug"] # 2019 und 2020
autumn = pd.DataFrame()
winter = pd.DataFrame()
spring = pd.DataFrame()
summer = pd.DataFrame()

for t in [first, second, third]:
     aut = data_min[data_min.t.str.contains(t[0])]
     win = data_min[data_min.t.str.contains(t[1])]
     spr = data_min[data_min.t.str.contains(t[2])]
     sum = data_min[data_min.t.str.contains(t[3])]
     autumn = pd.concat([autumn, aut], axis=0)
     winter = pd.concat([winter, win], axis=0)
     spring = pd.concat([spring, spr], axis=0)
     summer = pd.concat([summer, sum], axis=0)

"""train = pd.concat([autumn[0:int(len(autumn) * 0.8)], winter[0:int(len(winter) * 0.8)],
                   spring[0:int(len(spring) * 0.8)], summer[0:int(len(summer) * 0.8)]], axis=0)

test = pd.concat([autumn[int(len(autumn) * 0.8):len(autumn)], winter[int(len(winter) * 0.8):len(winter)],
                  spring[int(len(spring) * 0.8):len(spring)], summer[int(len(summer) * 0.8):len(summer)]], axis=0)"""

train, test = data_min[0:round(len(data_min)*0.8)], data_min[round(len(data_min)*0.8):len(data_min)]

# time dependent Variables
time_train, time_test = train.t, test.t
GHI_train, GHI_test = train.GHI, test.GHI
gti30t187a_train, gti30t187a_test = train.gti30t187a, test.gti30t187a
CSGHI_train, CSGHI_test = train.CSGHI, test.CSGHI
BNI_train, BNI_test = train.BNI, test.BNI
CSBNI_train, CSBNI_test = train.CSBNI, test.CSBNI
ENI_train, ENI_test = train.ENI, test.ENI
El_train, El_test = train.El, test.El
Pdc_train, Pdc_test = train.Pdc_33, test.Pdc_33
Pdcmean_train, Pdcmean_test = train.iloc[:, 109:].mean(axis=1), test.iloc[:, 109:].mean(axis=1)
Pdcmean_train = pd.Series(Pdcmean_train, name="Pdcmean")
Pdcmean_test = pd.Series(Pdcmean_test, name="Pdcmean")
GHI_KT_train, GHI_KT_test = train.kt, test.kt

# time independent Variables (almost constant)
TL_train, TL_test = train.TL, test.TL
Ta_train, Ta_test = train.Ta, test.Ta
vw_train, vw_test = train.vw, test.vw
AMa_train, AMa_test = train.AMa, test.AMa
