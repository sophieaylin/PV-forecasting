import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pa
import sklearn as sk
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVR
from sklearn.metrics import mean_squared_error, r2_score
from DataManagement import get_data

data = get_data()

time = data.t
gti30t187a = data.gti30t187a
GHI = data.GHI
Ta = data.Ta
BNI = data.BNI
Pdc = data.Pdc_1

# datatype list

Pdc = Pdc.fillna(0.01)  # replace_NaN(Pdc)
gti30t187a = gti30t187a.fillna(0.01)
GHI = GHI.fillna(0.01)

# Scaling trainings data: Normalization

Pdc_norm = Pdc_float/max(Pdc_float)
gti30t187a_norm = gti30t187a_float/max(gti30t187a_float)    
GHI_norm = GHI_float/max(GHI_float)
Ta_norm = Ta_float/max(Ta_float)

# Input

for m in range(0,len(Pdc)):
    Indata_nlist.append([gti30t187a_norm[m], Pdc_norm[m]])

Indata_array = np.array(Indata_nlist, dtype=float)

# Trainingsdata Split

gti30t187a_re = pa.DataFrame(gti30t187a_norm)
gti30t187a_re.values.reshape(-1, 1)
Pdc_re = np.ravel(Pdc_norm)
#Pdc_re.ravel.reshape(-1, 1)

Indata_array_train, Indata_array_test = train_test_split(Indata_array, test_size=0.4)
gti30t187a_train, gti30t187a_test = train_test_split(gti30t187a_re, test_size=0.4)
Pdc_train, Pdc_test = train_test_split(Pdc_re, test_size=0.4)

# Model SVR

SVR_model = SVR() #kernel='rbf')
SVR_model.fit(gti30t187a_train, Pdc_train)

Pdc_pred = SVR_model.predict(gti30t187a_test)

mse = mean_squared_error(Pdc_test, Pdc_pred)
rmse = np.sqrt(mse)

print('Root mean square: %f' % (rmse))