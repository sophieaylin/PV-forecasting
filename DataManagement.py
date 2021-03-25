import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

def get_data(deep_copy = True):
    return data.copy(deep_copy)

def get_features(deep_copy = True):
    # data have to be stored in a pandas DataFrame
    # build feature Normalization

    #GHI_kt = GHI.div(ENI.mul(np.sin(np.deg2rad(El))))
    BNI_kt = BNI_train.div(ENI_train)

    B_BNI_kt_1 = BNI_kt
    B_BNI_kt = BNI_kt
    B_GHI_kt_1 = GHI_KT_train
    B_GHI_kt = GHI_KT_train

    featuresB_GHI = pd.DataFrame()
    featuresB_BNI = pd.DataFrame()
    featuresL_GHI = pd.DataFrame()
    featuresL_BNI = pd.DataFrame()
    featuresV_GHI = pd.DataFrame()
    featuresV_BNI = pd.DataFrame()
    featuresL_GHI.insert(0, column='L_GHI_kt_0', value=B_GHI_kt)
    featuresL_BNI.insert(0, column='L_BNI_kt_0', value=B_BNI_kt)
    Pdc_shift = pd.DataFrame()

    for col,shift in zip(range(0, window), range(-1, -(window+1), -1)):
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

    for col in range(0, window):
        delta_kt_GHI = featuresL_GHI.iloc[:, 0:col+1].sub(GHI_kt_mean.values.reshape(len(GHI_kt_mean), col+1)).pow(2)
        delta_kt_BNI = featuresL_BNI.iloc[:, 0:col+1].sub(BNI_kt_mean.values.reshape(len(BNI_kt_mean), col+1)).pow(2)
        GHI_kt_mean.insert(col+1, column="times_%i" % col, value=GHI_kt_mean.B_GHI_kt_5.values)
        BNI_kt_mean.insert(col+1, column="times_%i" % col, value=BNI_kt_mean.B_BNI_kt_5.values)
        V_GHI_kt = pd.Series(np.sqrt(np.divide(delta_kt_GHI.sum(axis=1), col+1)))
        V_BNI_kt = pd.Series(np.sqrt(np.divide(delta_kt_BNI.sum(axis=1), col+1)))
        featuresV_GHI.insert(col, column='V_GHI_kt_%i' % col, value=V_GHI_kt)
        featuresV_BNI.insert(col, column='V_BNI_kt_%i' % col, value=V_BNI_kt)

    #Pdc_norm = Pdc.div(np.max(Pdc))
    #Pdc_norm = Pdc.div(ENI.mul(Capacity))
    Pdc_norm = Pdc_train.div(ENI_train)

    for col in range(0, window):
        Pdc_shift.insert(col, column='Pdc_{}min'.format(delta*(col+1)), value=Pdc_norm)
        Pdc_norm = Pdc_norm.shift(periods=1)        # check whether periods 1 or -1

    Pdc_shift = pd.concat([Pdc_shift, ENI_train, El_train], axis=1)

    features = pd.concat([time_train, featuresB_GHI, featuresB_BNI, featuresV_GHI, featuresV_BNI,
                          featuresL_GHI, featuresL_BNI, Pdc_shift, Pdc_train], axis=1)

    return features.copy(deep_copy)

def get_target(deep_copy = True):

    BNI_kt = BNI_test.div(ENI_test)

    B_BNI_kt_1 = BNI_kt
    B_BNI_kt = BNI_kt
    B_GHI_kt_1 = GHI_KT_test
    B_GHI_kt = GHI_KT_test

    featuresB_GHI = pd.DataFrame()
    featuresB_BNI = pd.DataFrame()
    featuresL_GHI = pd.DataFrame()
    featuresL_BNI = pd.DataFrame()
    featuresV_GHI = pd.DataFrame()
    featuresV_BNI = pd.DataFrame()
    featuresL_GHI.insert(0, column='L_GHI_kt_0', value=B_GHI_kt)
    featuresL_BNI.insert(0, column='L_BNI_kt_0', value=B_BNI_kt)
    Pdc_shift = pd.DataFrame()

    for col, shift in zip(range(0, window), range(-1, -(window + 1), -1)):
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

    for col in range(0, window):
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

    # Pdc_norm = Pdc.div(np.max(Pdc))
    # Pdc_norm = Pdc.div(ENI.mul(Capacity))
    Pdc_norm = Pdc_test.div(ENI_test)

    for col in range(0, window):
        Pdc_shift.insert(col, column='Pdc_{}min'.format(delta * (col + 1)), value=Pdc_norm)
        Pdc_norm = Pdc_norm.shift(periods=1)  # check whether periods 1 or -1

    Pdc_shift = pd.concat([Pdc_shift, ENI_test, El_test], axis=1)

    target = pd.concat([time_test, featuresB_GHI, featuresB_BNI, featuresV_GHI, featuresV_BNI,
                          featuresL_GHI, featuresL_BNI, Pdc_shift, Pdc_test], axis=1)

    return target.copy(deep_copy)

global Capacity
filename = 'Daten/PVAMM_201911-202011_PT5M_merged.csv'
data = pd.read_csv(filename)
data_min = data

Capacity = 20808.66
window = 6  # window size * delta[min] = forecasting period
delta = 5  # step size [min]

for Irr in ['GHI', 'DHI', 'gti30t187a', 'ENI']:         # 'Pdc_33'
    data_min = data_min.drop(data_min.index[data_min[Irr] == 0])

data_min = data_min.dropna(subset=['GHI', 'BNI', 'DHI', 'gti30t187a', 'ENI', 'El'])

first = ["Sep", "Dec", "Mar", "Jun"]
second = ["Oct", "Jan", "Apr", "Jul"]
third = ["Nov", "Feb", "May", "Aug"]

for t in [first, second, third]:
     herbst = data_min[data_min.t.str.contains(t[0])]
     winter = data_min[data_min.t.str.contains(t[1])]
     spring = data_min[data_min.t.str.contains(t[2])]
     summer = data_min[data_min.t.str.contains(t[3])]


"""train = pd.concat([herbst[0:int(len(herbst) * 0.8)], winter[0:int(len(winter) * 0.8)],
                   spring[0:int(len(spring) * 0.8)], summer[0:int(len(summer) * 0.8)]], axis=0)

test = pd.concat([herbst[int(len(herbst) * 0.8):len(herbst)], winter[int(len(winter) * 0.8):len(winter)],
                  spring[int(len(spring) * 0.8):len(spring)], summer[int(len(summer) * 0.8):len(summer)]], axis=0)"""

train, test = data_min[0:round(len(data_min)*0.8)], data_min[round(len(data_min)*0.8):len(data_min)]

time_train, time_test = train.t, test.t
GHI_train, GHI_test = train.GHI, test.GHI
BNI_train, BNI_test = train.BNI, test.BNI
ENI_train, ENI_test = train.ENI, test.ENI
El_train, El_test = train.El, test.El
Pdc_train, Pdc_test = train.Pdc_33, test.Pdc_33
GHI_KT_train, GHI_KT_test = train.kt, test.kt

""" first = ["Sep", "Dec", "Mar", "Jun"]
    second = ["Oct", "Jan", "Apr", "Jul"]
    third = ["Nov","Feb", "May", "Aug"]

    for t in [first, second, third]:
        herbst = features[features.t.str.contains(t[0])]
        winter = features[features.t.str.contains(t[1])]
        spring = features[features.t.str.contains(t[2])]
        summer = features[features.t.str.contains(t[3])]
        tarherbst = tar[tar.t.str.contains(t[0])]
        tarwinter = tar[tar.t.str.contains(t[1])]
        tarspring = tar[tar.t.str.contains(t[2])]
        tarsummer = tar[tar.t.str.contains(t[3])]

    train = pd.concat([herbst[0:int(len(herbst)*0.8)], winter[0:int(len(winter)*0.8)],
                   spring[0:int(len(spring)*0.8)], summer[0:int(len(summer)*0.8)]], axis=0)

    test = pd.concat([herbst[int(len(herbst)*0.8):len(herbst)], winter[int(len(winter)*0.8):len(winter)],
                   spring[int(len(spring)*0.8):len(spring)], summer[int(len(summer)*0.8):len(summer)]], axis=0)

    train_y = pd.concat([tarherbst[0:int(len(tarherbst)*0.8)], tarwinter[0:int(len(tarwinter)*0.8)],
                   tarspring[0:int(len(tarspring)*0.8)], tarsummer[0:int(len(tarsummer)*0.8)]], axis=0)

    test_y = pd.concat([tarherbst[int(len(tarherbst)*0.8):len(tarherbst)], tarwinter[int(len(tarwinter)*0.8):len(tarwinter)],
                   tarspring[int(len(tarspring)*0.8):len(tarspring)], tarsummer[int(len(tarsummer)*0.8):len(tarsummer)]], axis=0)"""