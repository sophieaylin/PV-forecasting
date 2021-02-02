import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pa
import sklearn as sk
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from pandas.plotting import autocorrelation_plot, lag_plot
from DataManagement import get_data


data = get_data()

time = data.t
gti30t187a = data.gti30t187a
GHI = data.GHI
Ta = data.Ta
BNI = data.BNI
Pdc = data.Pdc_1
Pdcmean = data.iloc[:, 109:].mean(axis=1)

"""for n in range(1,113185):
  linie = d[n]
  D = linie.split(",")
  time.append(D[0])
  gti30t187a.append(D[9])
  GHI.append(D[8])
  Ta.append(D[5])
  BNI.append(D[23])
  Pdc.append(D[32])
  for m in range (32,72):
        summe += float(D[m])
  Pdcmean = summe/m
  Pdcall.append(Pdcmean)"""


# datatype list

Pdc = Pdc.fillna(0.01)  # replace_NaN(Pdc)
gti30t187a = gti30t187a.fillna(0.01)
GHI = GHI.fillna(0.01)
Ta = Ta.fillna(0.01)

"""Pdc_float = np.array(Pdc, dtype=float)
gti30t187a_float = np.array(gti30t187a, dtype=float)
GHI_float = np.array(GHI, dtype=float)
Ta_float = np.array(Ta, dtype=float)"""

# Scaling trainings data: Normalization

Pdc_norm = Pdc/max(Pdc)
gti30t187a_norm = gti30t187a/max(gti30t187a)
GHI_norm = GHI/max(GHI)
Ta_norm = Ta/max(Ta)

# data analysis

"""plt.figure(figsize=(10,5))
plt.subplot(131)
plt.plot(gti30t187a_norm, Pdc_norm, 'ro')
plt.xlabel('gti30t187a')
plt.ylabel('Pdc')
plt.subplot(132)
plt.plot(GHI_norm, Pdc_norm, 'go')
plt.xlabel('GHI')
plt.ylabel('Pdc')
plt.subplot(133)
plt.boxplot(gti30t187a_norm, notch=1)
plt.suptitle('Normalized Data')"""

#plt.figure()
#lag_plot(gti30t187a_norm)

# Input
Indata_nlist = []

for m in range(0,len(Pdc)):
    Indata_nlist.append([gti30t187a_norm[m], GHI_norm[m]])

Indata_array = np.array(Indata_nlist, dtype=float)

# Trainingsdata Split

"""Indata_array_train, Indata_array_test, Pdc_train, Pdc_test = train_test_split(Indata_array, Pdc_norm)
sk.model_selection.train_test_split(Pdc_norm)"""

Indata_array_train, Indata_array_test = Indata_array[1:int(len(Indata_array)*0.8)], \
                                        Indata_array[int(len(Indata_array)*0.8):]
Pdc_train, Pdc_test = Pdc_norm[1:int(len(Pdc_norm)*0.8)], Pdc_norm[int(len(Pdc_norm)*0.8):]

# Linear Regression

regr = LinearRegression().fit(Indata_array_train, Pdc_train)
Pdc_pred = regr.predict(Indata_array_test)
r_sq = r2_score(Pdc_test, Pdc_pred)

x, y = np.mgrid[0:1:0.05, 0:1:0.05]
xy = np.vstack((x.flatten(), y.flatten()))

Power_pred = regr.intercept_ + np.dot(regr.coef_, xy)

print('Coefficients: ', regr.coef_)
print('Mean squared error: %.2f' % mean_squared_error(Pdc_test,Pdc_pred))
print('Coefficient of determination: %.2f \n' % r2_score(Pdc_test,Pdc_pred))

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