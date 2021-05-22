import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

class DataManager:

    def __init__(self):

        self.CAPACITY = 20808.66
        self.delta = 5 # step size [min]

        filename = 'Daten/PVAMM_201911-202011_PT5M_merged.csv'
        data = pd.read_csv(filename)

        gti_kt = data["gti30t187a"].div(data["ENI"])
        BNI_kt = data["BNI"].div(data["ENI"])

        data.insert(data.shape[1], "gti_kt", value=gti_kt)
        data.insert(data.shape[1], "BNI_kt", value=BNI_kt)

        self.data = data

        train = pd.DataFrame()
        test = pd.DataFrame()
        t = pd.to_datetime(self.data["t"]).array
        month_number = t.month

        for m in range(1, 13):
            in_month_m = month_number == m
            this_month = self.data.iloc[in_month_m, :]
            # splitting by day number ensures you have whole days on each data set
            day_number = t[in_month_m].day
            in_training_set = day_number < np.percentile(day_number, 80)
            train = pd.concat([train, this_month.iloc[in_training_set, :]], axis=0)
            test = pd.concat([test, this_month.iloc[~in_training_set, :]], axis=0)

        self.train = train
        self.test = test

    def get_data(self, deep_copy = True):
        return self.data.copy(deep_copy)

    def get_features (self, window_ft, window_tar, dropnight, deep_copy = True):
        # data have to be stored in a pandas DataFrame
        # for Input -> X (B = backward Average, L = lagged Average, V = Variability)
        # build feature Normalization

        # Trainings features

        B_BNI_kt_1 = self.train["BNI_kt"]
        B_BNI_kt = self.train["BNI_kt"]
        B_GHI_kt_1 = self.train["kt"] #  gti_kt_train  GHI_KT_train
        B_GHI_kt = self.train["kt"]  #  gti_kt_train  GHI_KT_train

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

        # Include Pdc to Trainingsset, which is not within the Horizon
        back = window_tar + 1
        Pdc_35_train = self.train["Pdc_33"].shift(periods=back)
        Pdc_35_train = Pdc_35_train[2*back:]

        features_train = pd.concat([self.train["t"], featuresB_GHI, featuresB_BNI, featuresV_GHI, featuresV_BNI,
                              featuresL_GHI, featuresL_BNI, self.train["El"], self.train["Pdc_33"]], axis=1)

        ft_train = features_train[0:len(features_train) - window_tar] # 2*back
        # ft_train.insert(features_train.shape[1], "Pdc_33", value=Pdc_35_train)
        ft_train.insert(ft_train.shape[1], "dataset", "Train")

        # Test features

        B_BNI_kt_1 = self.test["BNI_kt"]
        B_BNI_kt = self.test["BNI_kt"]
        B_GHI_kt_1 = self.test["kt"] #  gti_kt_train  GHI_KT_train
        B_GHI_kt = self.test["kt"]

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

        # Include Pdc to Testset, which is not within the Horizon
        Pdc_35_test = self.test["Pdc_33"].shift(periods=back)
        Pdc_35_test = Pdc_35_test[2*back:]

        features_test = pd.concat([self.test["t"], featuresB_GHI, featuresB_BNI, featuresV_GHI, featuresV_BNI,
                              featuresL_GHI, featuresL_BNI, self.test["El"], self.test["Pdc_33"]], axis=1)

        ft_test = features_test[0:len(features_test) - window_tar] # 2*back
        # ft_test.insert(features_test.shape[1], "Pdc_33", value=Pdc_35_test)
        ft_test.insert(ft_test.shape[1], "dataset", "Test")

        """if dropnight == "true":
            ft_train = ft_train.drop(ft_train.index[ft_train["El"] < 15])
            ft_test = ft_test.drop(ft_test.index[ft_test["El"] < 15])"""

        features = pd.concat([ft_train, ft_test], axis=0)

        return features.copy(deep_copy)

    def get_target_Pdc (self, window_tar, dropnight, deep_copy = True):
        # for Output -> Y (Power)
        # Train target
        # For Linear Regression for time t: x(t), y(t+1) in one row, SEE
        # https://ichi.pro/de/so-formen-sie-daten-neu-und-fuhren-mit-lstm-eine-regression-fur-zeitreihen-durch-21155626274048

        Pdc_shift = pd.DataFrame()
        Pdc_norm = self.train["Pdc_33"].div(self.train["ENI"])
        Pdc_sp = Pdc_norm.shift(periods=1)
        Pdc_shift.insert(0, column='Pdc_sp', value=Pdc_sp)

        for col in range(1, window_tar + 1):
            Pdc_norm = Pdc_norm.shift(periods=-1)
            Pdc_shift.insert(col, column='Pdc_{}min'.format(self.delta * (col)), value=Pdc_norm)

        target_train = pd.concat([self.train["t"], Pdc_shift, self.train["ENI"], self.train["El"]], axis=1)
        target_train.insert(target_train.shape[1], "dataset", "Train")
        t_train = target_train[0:len(target_train) - window_tar]

        # Test target

        Pdc_shift = pd.DataFrame()
        Pdc_norm = self.test["Pdc_33"].div(self.test["ENI"])
        Pdc_sp = Pdc_norm.shift(periods=1)
        Pdc_shift.insert(0, column='Pdc_sp', value=Pdc_sp)

        for col in range(1, window_tar + 1):
            Pdc_norm = Pdc_norm.shift(periods=-1)
            Pdc_shift.insert(col, column='Pdc_{}min'.format(self.delta * (col)), value=Pdc_norm)

        target_test = pd.concat([self.test["t"], Pdc_shift, self.test["ENI"], self.test["El"]], axis=1)
        target_test.insert(target_test.shape[1], "dataset", "Test")
        t_test = target_test[0:len(target_test) - window_tar]

        """if dropnight == "true":
            t_train = t_train.drop(t_train.index[t_train["El"] < 15])
            t_test = t_test.drop(t_test.index[t_test["El"] < 15])"""

        target = pd.concat([t_train, t_test], axis=0)

        return target.copy(deep_copy)

    def get_features_LSTM (self, window_LSTM, feature_str):

        # Train
        features_train = self.train[feature_str]
        X_train = features_train[0:len(features_train) - window_LSTM]

        # Test
        features_test = self.test[feature_str]
        X_test = features_test[0:len(features_test) - window_LSTM]

        return X_train, X_test # features.copy(deep_copy)

    def get_target_LSTM(self, window_LSTM, deep_copy = True):
        # for Output -> Y (Power)
        # Train target
        # for LSTM Input for time t: x(t), y(t) in one row
        # take the shortest backwards step as Smart Persistence Model (sp)

        Pdc_shift = pd.DataFrame()
        Pdc_norm = self.train["Pdc_33"].div(self.CAPACITY) # self.train["ENI"]
        Pdc_sp = self.train["Pdc_33"].shift(periods=1)
        Pdc_shift.insert(0, column='Pdc_sp', value=Pdc_sp)

        for col in range(1, window_LSTM + 1):
            Pdc_norm = Pdc_norm.shift(periods=-1)
            Pdc_shift.insert(col, column='Pdc_{}min'.format(self.delta * (col)), value=Pdc_norm)

        target_train = pd.concat([self.train["t"], Pdc_shift, self.train["ENI"], self.train["El"], self.train["Pdc_33"]], axis=1)
        Y_train = target_train[0:len(target_train) - window_LSTM]

        # Test target

        Pdc_shift = pd.DataFrame()
        Pdc_norm = self.test["Pdc_33"].div(self.CAPACITY) # self.test["ENI"]
        Pdc_sp = self.test["Pdc_33"].shift(periods=1)
        Pdc_shift.insert(0, column='Pdc_sp', value=Pdc_sp)

        for col in range(1, window_LSTM + 1):
            Pdc_norm = Pdc_norm.shift(periods=-1)
            Pdc_shift.insert(col, column='Pdc_{}min'.format(self.delta * (col)), value=Pdc_norm)

        target_test = pd.concat([self.test["t"], Pdc_shift, self.test["ENI"], self.test["El"], self.test["Pdc_33"]], axis=1)
        Y_test = target_test[0:len(target_test) - window_LSTM]

        return Y_train.copy(deep_copy), Y_test.copy(deep_copy)

    def get_target_Irr(self, window_tar, deep_copy = True):
        # for Output -> Y (Irradiance, kt)
        # Train target

        BNI_train = self.train["BNI"]
        GHI_train = self.train["GHI"]
        El_train = self.train["El"]
        CSGHI_train = self.train["CSGHI"]
        CSBNI_train = self.train["CSBNI"]
        GHI_KT_train = self.train["kt"]
        ENI_train = self.train["ENI"]

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
            block.insert(0, column="GHI_{}min".format(self.delta * (blk + 1)), value=GHI_train)
            block.insert(1, column="BNI_{}min".format(self.delta * (blk + 1)), value=BNI_train)
            block.insert(2, column="GHI_clear_{}min".format(self.delta * (blk + 1)), value=CSGHI_train)
            block.insert(3, column="BNI_clear_{}min".format(self.delta * (blk + 1)), value=CSBNI_train)
            block.insert(4, column="GHI_kt_{}min".format(self.delta * (blk + 1)), value=GHI_KT_train)
            block.insert(5, column="BNI_kt_{}min".format(self.delta * (blk + 1)), value=BNI_kt_train)
            block.insert(6, column="El_{}min".format(self.delta * (blk + 1)), value=El_train)
            block.insert(7, column="ENI_{}min".format(self.delta * (blk + 1)), value=ENI_train)
            target_Irr_train = pd.concat([target_Irr_train, block], axis=1)

        target_Irr_train = pd.concat([self.train["t"], target_Irr_train], axis=1) # del later, just for validation

        target_Irr_train.insert(target_Irr_train.shape[1], "dataset", "Train")
        t_Irr_train = target_Irr_train[0:len(target_Irr_train) - window_tar]

        # key = pd.DataFrame(np.array(range(0, len(t_Irr_train))), columns=["key"])  # del later, just for validation
        # tar_Irr_train = pd.concat([key, t_Irr_train], axis=1)  # del later, just for validation

        # Test target

        BNI_test = self.test["BNI"]
        GHI_test = self.test["GHI"]
        El_test = self.test["El"]
        CSGHI_test = self.test["CSGHI"]
        CSBNI_test = self.test["CSBNI"]
        GHI_KT_test = self.test["kt"]
        ENI_test = self.test["ENI"]

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
            block.insert(0, column="GHI_{}min".format(self.delta * (blk + 1)), value=GHI_test)
            block.insert(1, column="BNI_{}min".format(self.delta * (blk + 1)), value=BNI_test)
            block.insert(2, column="GHI_clear_{}min".format(self.delta * (blk + 1)), value=CSGHI_test)
            block.insert(3, column="BNI_clear_{}min".format(self.delta * (blk + 1)), value=CSBNI_test)
            block.insert(4, column="GHI_kt_{}min".format(self.delta * (blk + 1)), value=GHI_KT_test)
            block.insert(5, column="BNI_kt_{}min".format(self.delta * (blk + 1)), value=BNI_kt_test)
            block.insert(6, column="El_{}min".format(self.delta * (blk + 1)), value=El_test)
            block.insert(7, column="ENI_{}min".format(self.delta * (blk + 1)), value=ENI_test)
            target_Irr_test = pd.concat([target_Irr_test, block], axis=1)

        target_Irr_test = pd.concat([self.test["t"], target_Irr_test], axis=1) # del later, just for validation

        target_Irr_test.insert(target_Irr_test.shape[1], "dataset", "Test")
        t_Irr_test = target_Irr_test[0:len(target_Irr_test) - window_tar]

        # key = pd.DataFrame(np.array(range(0, len(t_Irr_test))), columns=["key"]) # del later, just for validation
        # tar_Irr_test = pd.concat([key, t_Irr_test], axis=1) # del later, just for validation

        target = pd.concat([t_Irr_train, t_Irr_test], axis=0)

        return target.copy(deep_copy)


