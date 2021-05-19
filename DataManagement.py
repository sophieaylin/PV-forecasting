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
    gti_kt_train = gti30t187a_train.div(ENI_train)
    BNI_kt_train = BNI_train.div(ENI_train)

    B_BNI_kt_1 = BNI_kt_train
    B_BNI_kt = BNI_kt_train
    B_GHI_kt_1 = GHI_KT_train #  gti_kt_train  GHI_KT_train
    B_GHI_kt = GHI_KT_train  #  gti_kt_train  GHI_KT_train

    featuresB_GHI = pd.DataFrame()
    featuresB_BNI = pd.DataFrame()
    featuresL_GHI = pd.DataFrame()
    featuresL_BNI = pd.DataFrame()
    featuresV_GHI = pd.DataFrame()
    featuresV_BNI = pd.DataFrame()
    featuresL_GHI.insert(0, column='L_GHI_kt_0', value=B_GHI_kt)
    featuresL_BNI.insert(0, column='L_BNI_kt_0', value=B_BNI_kt)

    for col in range(0, window_ft):
        featuresB_GHI.insert(col, column='B_GHI_kt_%i' % col, value=B_GHI_kt)
        featuresB_BNI.insert(col, column='B_BNI_kt_%i' % col, value=B_BNI_kt)
        BGHI_shift = B_GHI_kt_1.shift(periods=col+1)
        BBNI_shift = B_BNI_kt_1.shift(periods=col+1)
        clmn = col + 1
        featuresL_GHI.insert(clmn, column='L_GHI_kt_%i' % clmn, value=BGHI_shift)
        featuresL_BNI.insert(clmn, column='L_BNI_kt_%i' % clmn, value=BBNI_shift)
        B_GHI_kt = featuresL_GHI.mean(axis=1)
        B_BNI_kt = featuresL_BNI.mean(axis=1)

    featuresL_GHI = featuresL_GHI.drop('L_GHI_kt_{}'.format(window_ft), axis=1)
    featuresL_BNI = featuresL_BNI.drop('L_BNI_kt_{}'.format(window_ft), axis=1)
    GHI_kt_mean = featuresB_GHI["B_GHI_kt_{}".format(window_ft-1)].to_frame()
    BNI_kt_mean = featuresB_BNI["B_BNI_kt_{}".format(window_ft-1)].to_frame()

    for col in range(0, window_ft):
        delta_kt_GHI = featuresL_GHI.iloc[:, 0:col+1].sub(GHI_kt_mean.values.reshape(len(GHI_kt_mean), col+1)).pow(2)
        delta_kt_BNI = featuresL_BNI.iloc[:, 0:col+1].sub(BNI_kt_mean.values.reshape(len(BNI_kt_mean), col+1)).pow(2)
        GHI_kt_mean.insert(col+1, column="times_%i" % col, value=GHI_kt_mean["B_GHI_kt_{}".format(window_ft-1)].values)
        BNI_kt_mean.insert(col+1, column="times_%i" % col, value=BNI_kt_mean["B_BNI_kt_{}".format(window_ft-1)].values)
        V_GHI_kt = pd.Series(np.sqrt(np.divide(delta_kt_GHI.sum(axis=1), col+1)))
        V_BNI_kt = pd.Series(np.sqrt(np.divide(delta_kt_BNI.sum(axis=1), col+1)))
        featuresV_GHI.insert(col, column='V_GHI_kt_%i' % col, value=V_GHI_kt)
        featuresV_BNI.insert(col, column='V_BNI_kt_%i' % col, value=V_BNI_kt)

    features_train = pd.concat([time_train, featuresB_GHI, featuresB_BNI, featuresV_GHI, featuresV_BNI,
                          featuresL_GHI, featuresL_BNI, Ta_train, TL_train, vw_train, AMa_train], axis=1)

    features_train.insert(features_train.shape[1], "dataset", "Train")
    ft_train = features_train[0:len(features_train) - window_tar]

    # Test features

    gti_kt_test = gti30t187a_test.div(ENI_test)
    BNI_kt_test = BNI_test.div(ENI_test)

    B_BNI_kt_1 = BNI_kt_test
    B_BNI_kt = BNI_kt_test
    B_GHI_kt_1 = GHI_KT_test  # gti_kt GHI_KT_test
    B_GHI_kt = GHI_KT_test  # gti_kt GHI_KT_test

    featuresB_GHI = pd.DataFrame()
    featuresB_BNI = pd.DataFrame()
    featuresL_GHI = pd.DataFrame()
    featuresL_BNI = pd.DataFrame()
    featuresV_GHI = pd.DataFrame()
    featuresV_BNI = pd.DataFrame()
    featuresL_GHI.insert(0, column='L_GHI_kt_0', value=B_GHI_kt)
    featuresL_BNI.insert(0, column='L_BNI_kt_0', value=B_BNI_kt)

    for col in range(0, window_ft):
        featuresB_GHI.insert(col, column='B_GHI_kt_%i' % col, value=B_GHI_kt)
        featuresB_BNI.insert(col, column='B_BNI_kt_%i' % col, value=B_BNI_kt)
        BGHI_shift = B_GHI_kt_1.shift(periods=col + 1)
        BBNI_shift = B_BNI_kt_1.shift(periods=col + 1)
        clmn = col + 1
        featuresL_GHI.insert(clmn, column='L_GHI_kt_%i' % clmn, value=BGHI_shift)
        featuresL_BNI.insert(clmn, column='L_BNI_kt_%i' % clmn, value=BBNI_shift)
        B_GHI_kt = featuresL_GHI.mean(axis=1)
        B_BNI_kt = featuresL_BNI.mean(axis=1)

    featuresL_GHI = featuresL_GHI.drop('L_GHI_kt_{}'.format(window_ft), axis=1)
    featuresL_BNI = featuresL_BNI.drop('L_BNI_kt_{}'.format(window_ft), axis=1)
    GHI_kt_mean = featuresB_GHI["B_GHI_kt_{}".format(window_ft-1)].to_frame()
    BNI_kt_mean = featuresB_BNI["B_BNI_kt_{}".format(window_ft-1)].to_frame()

    for col in range(0, window_ft):
        delta_kt_GHI = featuresL_GHI.iloc[:, 0:col + 1].sub(GHI_kt_mean.values.reshape(len(GHI_kt_mean), col + 1)).pow(
            2)
        delta_kt_BNI = featuresL_BNI.iloc[:, 0:col + 1].sub(BNI_kt_mean.values.reshape(len(BNI_kt_mean), col + 1)).pow(
            2)
        GHI_kt_mean.insert(col + 1, column="times_%i" % col, value=GHI_kt_mean["B_GHI_kt_{}".format(window_ft-1)].values)
        BNI_kt_mean.insert(col + 1, column="times_%i" % col, value=BNI_kt_mean["B_BNI_kt_{}".format(window_ft-1)].values)
        V_GHI_kt = pd.Series(np.sqrt(np.divide(delta_kt_GHI.sum(axis=1), col + 1)))
        V_BNI_kt = pd.Series(np.sqrt(np.divide(delta_kt_BNI.sum(axis=1), col + 1)))
        featuresV_GHI.insert(col, column='V_GHI_kt_%i' % col, value=V_GHI_kt)
        featuresV_BNI.insert(col, column='V_BNI_kt_%i' % col, value=V_BNI_kt)

    features_test = pd.concat([time_test, featuresB_GHI, featuresB_BNI, featuresV_GHI, featuresV_BNI,
                          featuresL_GHI, featuresL_BNI, Ta_test, TL_test, vw_test, AMa_test], axis=1)

    features_test.insert(features_test.shape[1], "dataset", "Test")
    ft_test = features_test[0:len(features_test) - window_tar]

    features = pd.concat([ft_train, ft_test], axis=0)

    return features.copy(deep_copy)

def get_target_Pdc (deep_copy = True):
    # for Output -> Y (Power)
    # Train target
    # For Linear Regression for time t: x(t), y(t+1) in one row, SEE
    # https://ichi.pro/de/so-formen-sie-daten-neu-und-fuhren-mit-lstm-eine-regression-fur-zeitreihen-durch-21155626274048

    Pdc_shift = pd.DataFrame()
    Pdc_norm = Pdc_train.div(ENI_train)
    Pdc_sp = Pdc_norm.shift(periods=1)
    Pdc_shift.insert(0, column='Pdc_sp', value=Pdc_sp)

    for col in range(1, window_tar + 1):
        Pdc_norm = Pdc_norm.shift(periods=-1)
        Pdc_shift.insert(col, column='Pdc_{}min'.format(delta * (col)), value=Pdc_norm)

    target_train = pd.concat([time_train, Pdc_shift, ENI_train, El_train, Pdc_train], axis=1)
    target_train.insert(target_train.shape[1], "dataset", "Train")
    t_train = target_train[0:len(target_train) - window_tar]

    # Test target

    Pdc_shift = pd.DataFrame()
    Pdc_norm = Pdc_test.div(ENI_test)
    Pdc_sp = Pdc_norm.shift(periods=1)
    Pdc_shift.insert(0, column='Pdc_sp', value=Pdc_sp)

    for col in range(1, window_tar + 1):
        Pdc_norm = Pdc_norm.shift(periods=-1)
        Pdc_shift.insert(col, column='Pdc_{}min'.format(delta * (col)), value=Pdc_norm)

    target_test = pd.concat([time_test, Pdc_shift, ENI_test, El_test, Pdc_test], axis=1)
    target_test.insert(target_test.shape[1], "dataset", "Test")
    t_test = target_test[0:len(target_test) - window_tar]

    target = pd.concat([t_train, t_test], axis=0)

    return target.copy(deep_copy)

def get_features_LSTM (deep_copy = True):

    # Train

    BNI_kt_train = BNI_train.div(ENI_train)
    BNI_kt_train = pd.DataFrame(BNI_kt_train, columns=["BNI_kt"])

    gti_kt_train = gti30t187a_train.div(ENI_train)
    gti_kt_train = pd.DataFrame(gti_kt_train, columns=["gti_kt"])

    features_train = pd.DataFrame()
    features_train = pd.concat([time_train, GHI_train, BNI_train, GHI_KT_train, BNI_kt_train, gti_kt_train, Ta_train, TL_train, vw_train,
                                AMa_train, RH_train, kd_train], axis=1)
    features_train.insert(features_train.shape[1], "dataset", "Train")
    ft_train = features_train[0:len(features_train) - window_LSTM]

    # Test

    BNI_kt_test = BNI_test.div(ENI_test)
    BNI_kt_test = pd.DataFrame(BNI_kt_test, columns=["BNI_kt"])

    gti_kt_test = gti30t187a_test.div(ENI_test)
    gti_kt_test = pd.DataFrame(gti_kt_test, columns=["gti_kt"])

    features_test = pd.DataFrame()
    features_test = pd.concat([time_test, GHI_test, BNI_test, GHI_KT_test, BNI_kt_test, gti_kt_test, Ta_test, TL_test, vw_test,
                               AMa_test, RH_test, kd_test], axis=1)
    features_test.insert(features_test.shape[1], "dataset", "Test")
    ft_test = features_test[0:len(features_test)-window_LSTM]

    features = pd.concat([ft_train, ft_test], axis=0)

    return features.copy(deep_copy)

def get_target_LSTM(deep_copy = True):
    # for Output -> Y (Power)
    # Train target
    # for LSTM Input for time t: x(t), y(t) in one row
    # take the shortest backwards step as Smart Persistence Model (sp)

    Pdc_shift = pd.DataFrame()
    Pdc_norm = Pdc_train.div(Pdc_train.max()) # ENI_train
    Pdc_sp = Pdc_train.shift(periods=1)
    Pdc_shift.insert(0, column='Pdc_sp', value=Pdc_sp)

    for col in range(1, window_LSTM + 1):
        Pdc_norm = Pdc_norm.shift(periods=-1)
        Pdc_shift.insert(col, column='Pdc_{}min'.format(delta * (col)), value=Pdc_norm)


    target_train = pd.concat([time_train, Pdc_shift, ENI_train, El_train, Pdc_train], axis=1)
    target_train.insert(target_train.shape[1], "dataset", "Train")
    t_train = target_train[0:len(target_train) - window_LSTM]

    # Test target

    Pdc_shift = pd.DataFrame()
    Pdc_norm = Pdc_test.div(Pdc_test.max()) # ENI_test
    Pdc_sp = Pdc_test.shift(periods=1)
    Pdc_shift.insert(0, column='Pdc_sp', value=Pdc_sp)

    for col in range(1, window_LSTM + 1):
        Pdc_norm = Pdc_norm.shift(periods=-1)
        Pdc_shift.insert(col, column='Pdc_{}min'.format(delta * (col)), value=Pdc_norm)

    target_test = pd.concat([time_test, Pdc_shift, ENI_test, El_test, Pdc_test], axis=1)
    target_test.insert(target_test.shape[1], "dataset", "Test")
    t_test = target_test[0:len(target_test) - window_LSTM]

    target = pd.concat([t_train, t_test], axis=0)

    return target.copy(deep_copy)

def get_target_Irr(deep_copy = True):
    # for Output -> Y (Irradiance, kt)
    # Train target

    global BNI_train, GHI_train, El_train, CSGHI_train, CSBNI_train, GHI_KT_train, ENI_train

    target_Irr_train = pd.DataFrame()
    BNI_kt_train = BNI_train.div(ENI_train)

    for blk in range(0, window_tar):
        GHI_train = GHI_train.shift(periods=-1)
        BNI_train = BNI_train.shift(periods=-1)
        CSGHI_train = CSGHI_train.shift(periods=-1)
        CSBNI_train = CSBNI_train.shift(periods=-1)
        GHI_KT_train = GHI_KT_train.shift(periods=-1)
        BNI_kt_train = BNI_kt_train.shift(periods=-1)
        El_train = El_train.shift(periods=-1)
        ENI_train = ENI_train.shift(periods=-1)
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

    target_Irr_train = pd.concat([time_train, target_Irr_train], axis=1) # del later, just for validation

    target_Irr_train.insert(target_Irr_train.shape[1], "dataset", "Train")
    t_Irr_train = target_Irr_train[0:len(target_Irr_train) - window_tar]

    # key = pd.DataFrame(np.array(range(0, len(t_Irr_train))), columns=["key"])  # del later, just for validation
    # tar_Irr_train = pd.concat([key, t_Irr_train], axis=1)  # del later, just for validation

    # Test target

    global BNI_test, GHI_test, El_test, CSGHI_test, CSBNI_test, GHI_KT_test, ENI_test

    target_Irr_test = pd.DataFrame()
    BNI_kt_test = BNI_test.div(ENI_test)

    for blk in range(0, window_tar):
        GHI_test = GHI_test.shift(periods=-1)
        BNI_test = BNI_test.shift(periods=-1)
        CSGHI_test = CSGHI_test.shift(periods=-1)
        CSBNI_test = CSBNI_test.shift(periods=-1)
        GHI_KT_test = GHI_KT_test.shift(periods=-1)
        BNI_kt_test = BNI_kt_test.shift(periods=-1)
        El_test = El_test.shift(periods=-1)
        ENI_test = ENI_test.shift(periods=-1)
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

    target_Irr_test = pd.concat([time_test, target_Irr_test], axis=1) # del later, just for validation

    target_Irr_test.insert(target_Irr_test.shape[1], "dataset", "Test")
    t_Irr_test = target_Irr_test[0:len(target_Irr_test) - window_tar]

    # key = pd.DataFrame(np.array(range(0, len(t_Irr_test))), columns=["key"]) # del later, just for validation
    # tar_Irr_test = pd.concat([key, t_Irr_test], axis=1) # del later, just for validation

    target = pd.concat([t_Irr_train, t_Irr_test], axis=0)

    return target.copy(deep_copy)


filename = 'Daten/PVAMM_201911-202011_PT5M_merged.csv'
data = pd.read_csv(filename)
data_min = data

CAPACITY = 20808.66
window_ft = 23 # time window for feature generation;
window_tar = 12 # time window for forecast horizon !!!adjust horizon respectively in Regression.py!!!
window_LSTM = 36
delta = 5  # step size [min]

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

train = pd.concat([autumn[0:int(len(autumn) * 0.8)], winter[0:int(len(winter) * 0.8)],
                   spring[0:int(len(spring) * 0.8)], summer[0:int(len(summer) * 0.8)]], axis=0)

test = pd.concat([autumn[int(len(autumn) * 0.8):len(autumn)], winter[int(len(winter) * 0.8):len(winter)],
                  spring[int(len(spring) * 0.8):len(spring)], summer[int(len(summer) * 0.8):len(summer)]], axis=0)

"""train, test = data_min[0:round(len(data_min)*0.8)], data_min[round(len(data_min)*0.8):len(data_min)]"""

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
RH_train, RH_test = train.RH, test.RH   # relative humidity
tpw_train, twp_test = train.tpw, test.tpw   # total precipitable water [mm]
kd_train, kd_test = train.kd, test. kd  # diffuse fraction kd=DHI/GHI

# time independent Variables (almost constant)
TL_train, TL_test = train.TL, test.TL   # modeled link turbity
Ta_train, Ta_test = train.Ta, test.Ta   # vw speed
vw_train, vw_test = train.vw, test.vw
AMa_train, AMa_test = train.AMa, test.AMa


""" qsub -N other_name -l select=2:node_type=hsw:mpiprocs=24 -l walltime=00:20:00 my_batchjob_script.pbs
putty.exe -ssh hpcsayli@vulcan.hww.hlrs.de

scp LOCALPATH/FILE hpcsayli@vulcan.hww.hlrs.de:$HOME/...

qsub -I -l walltime=01:00:00

qsub -v -l select=10:ncpus=40 -l walltime=03:00:00 script.pbs

#!/usr/bin/env bash
#PBS -l select=1:node_type=skl192gb40c:mpiprocs=40
#PBS -m abe
#
# Print CPU information
NCPUS=$(sort -u $PBS_NODEFILE)
echo "Assigned node(s):"
echo "Node ID: $(pbsnodes $NCPUS)"
NCPUS=$(pbsnodes $NCPUS | grep pcpus | grep -Eo [0-9]+)
#
# using the INTEL MPI module
module load mpi/impi
mpirun -np 1 -ppn $NCPUS "python_script.py"
#
echo "Job ended $(date)"

"""