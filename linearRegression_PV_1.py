import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pa
import sklearn as sk
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from DataManagement import get_data


data = get_data()
data_min = data

for Irr in ['GHI', 'DHI', 'gti30t187a', 'ENI']:
    data_min = data_min.drop(data_min.index[data_min[Irr]==0])

data_min = data_min.dropna(subset=['GHI', 'BNI', 'DHI', 'gti30t187a', 'ENI', 'Pdc_33'])
#dropnaP = data_min.dropna(subset=['Pdc_1', 'Pdc_2', 'Pdc_3', 'Pdc_4', 'Pdc_5'])

time = data_min.t
gti30t187a = data_min.gti30t187a
GHI = data_min.GHI
Ta = data_min.Ta
BNI = data_min.BNI
Pdc = data_min.Pdc_33
Pdcmean = data_min.iloc[:, 109:].mean(axis=1)

# Scaling trainings data: Normalization

Pdc_norm = Pdc/max(Pdc)
gti30t187a_norm = gti30t187a/max(gti30t187a)
GHI_norm = GHI/max(GHI)
Ta_norm = Ta/max(Ta)

# Input

Indata_nlist = []

for m in range(0,len(Pdc)):
    Indata_nlist.append([gti30t187a_norm.iloc[m], GHI_norm.iloc[m]])

Indata_array = np.array(Indata_nlist, dtype=float)

# Trainingsdata Split

"""Indata_array_train, Indata_array_test = Indata_array[1:int(len(Indata_array)*0.8)], \
                                        Indata_array[int(len(Indata_array)*0.8):]"""
Indata_array_train, Indata_array_test = Indata_array[1:int(len(Indata_array)-288)], \
                                        Indata_array[int(len(Indata_array)-288):]
Pdc_train, Pdc_test = Pdc_norm[1:len(Pdc)-288], Pdc_norm[len(Pdc)-288:len(Pdc)]
"""Pdc_train, Pdc_test = Pdc_norm[1:int(len(Pdc_norm)*0.8)], Pdc_norm[int(len(Pdc_norm)*0.8):]"""

# Linear Regression

regr = LinearRegression().fit(Indata_array_train, Pdc_train)
Pdc_pred = regr.predict(Indata_array_test)

# Denormalize

Pdc_test = Pdc_test * max(Pdc)
Pdc_pred = Pdc_pred * max(Pdc)

# Metrics
error = (Pdc_test-Pdc_pred)
MBE = error.mean()
MAE = np.nanmean(np.abs(error))
MSE = sk.metrics.mean_squared_error(Pdc_test, Pdc_pred)
RMSE = np.sqrt(MSE)
r_sq = r2_score(Pdc_test, Pdc_pred)

x, y = np.mgrid[0:1:0.05, 0:1:0.05]
xy = np.vstack((x.flatten(), y.flatten()))

Power_pred = regr.intercept_ + np.dot(regr.coef_, xy)

print('Coefficients: ', regr.coef_)
print('MAE: {} // MBE: {} // MSE: {} // RMSE: {}'.format(MAE, MBE, MSE, RMSE))
print('Coefficient of determination: %.2f \n' % r2_score(Pdc_test,Pdc_pred))

plt.figure()
plt.plot(Pdc_test.values)
plt.plot(Pdc_pred, color='red')

plt.figure(figsize=(10,5))
plt.subplot(131)
plt.plot(Indata_array_test[:, 0], Pdc_test, 'ro')
plt.xlabel('gti30t187a_test')
plt.ylabel('Pdc_test')
plt.subplot(132)
plt.plot(Indata_array_test[:, 0], Pdc_pred, 'bo')
plt.xlabel('gti30t187a_test')
plt.ylabel('Pdc_pred')
plt.subplot(133)
plt.plot(Power_pred)
plt.xlabel('gti30t187a_test')
plt.ylabel('Power_pred')
plt.suptitle('Linear Regression of 1 year')

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(Indata_array_test[:, 0], Indata_array_test[:, 1], Pdc_test, label='Test Werte')
ax.scatter(Indata_array_test[:, 0], Indata_array_test[:, 1], Pdc_pred, label='Prognose')
ax.scatter(xy[0, :], xy[1, :], Power_pred, s=1, label='Power Funktion')
ax.set_xlabel('gti30t187a')
ax.set_ylabel('GHI')
ax.set_zlabel('Power')
plt.suptitle('Lineare Regression over one year')
ax.legend()
ax.grid(True)
plt.show()