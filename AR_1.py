import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pa
import sklearn as sk
import statsmodels as stat
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from pandas.plotting import autocorrelation_plot, lag_plot
from statsmodels.tsa.ar_model import AutoReg
from DataManagement import get_data

data = get_data()
data_min = data

for Irr in ['GHI', 'DHI', 'gti30t187a', 'ENI']:
    data_min = data_min.drop(data_min.index[data_min[Irr]==0])

data_min = data_min.dropna(subset=['GHI', 'BNI', 'DHI', 'gti30t187a', 'ENI', 'Pdc_33'])

time = data_min.t
gti30t187a = data_min.gti30t187a
GHI = data_min.GHI
Ta = data_min.Ta
BNI = data_min.BNI
Pdc = data_min.Pdc_33
Pdcmean = data_min.iloc[:, 109:].mean(axis=1)

Indata_nlist = []

# Scaling trainings data: Normalization

Pdc_norm = Pdc/max(Pdc)
gti30t187a_norm = gti30t187a/max(gti30t187a)    
GHI_norm = GHI/max(GHI)
Ta_norm = Ta/max(Ta)

# Input

"""for m in range(0,len(Pdc)):
    Indata_nlist.append([gti30t187a_norm[m], Pdc_norm[m]])

Indata_array = np.array(Indata_nlist, dtype=float)"""

# Trainingsdata Split

#Indata_array_train, Indata_array_test = train_test_split(Indata_array)
Pdc_train, Pdc_test = Pdc_norm[1:len(Pdc)-288], Pdc_norm[len(Pdc)-288:len(Pdc)]

# Autoregression fixed Model

model = AutoReg(Pdc_train, lags=1000)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)

# pred = model_fit.predict(start=len(Pdc_train), end=len(Pdc_train)+len(Pdc_test)-1, dynamic=False)
# for i in range(len(pred)):
#     print('predicted=%f, expected=%f' % (pred[i], Pdc_test[i]))
# rmse = np.sqrt(mean_squared_error(Pdc_test, pred))
# print('Test RMSE: %.3f' % rmse)
# plt.plot(Pdc_test)
# plt.plot(pred, color='red')
# plt.show()

# Autoregression rolling Model

window = 1000
hist = Pdc_train[len(Pdc_train)-window:] # .values versuchen
hist = hist.reset_index()
hist = [hist.at[i, 'Pdc_33'] for i in range(len(hist))] # makes a row [list] from Series with columns
pred = list()
for t in range(len(Pdc_test)):
      length = len(hist)
      lag = [hist[i] for i in range(length-window, length)]
      Pdc_pred = model_fit.params[0]
      for d in range(window):
            Pdc_pred += model_fit.params[d+1] * lag[window-d-1]
      obs = Pdc_test.values[t]
      pred.append(Pdc_pred)
      hist.append(obs)
      print('predicted=%f, expected=%f' % (Pdc_pred, obs))

# Denormalisation

Pdc_test = Pdc_test * max(Pdc)
pred = pa.Series(pred) * max(Pdc)

error = (Pdc_test.values-pred)
MBE = error.mean()
MAE = np.nanmean(np.abs(error))
MSE = sk.metrics.mean_squared_error(Pdc_test, pred)
RMSE = np.sqrt(MSE)
r_sq = r2_score(Pdc_test, pred)
print('MAE: {} // MBE: {} // MSE: {} // RMSE: {}'.format(MAE, MBE, MSE, RMSE))
print('Coefficient of determination: %.2f \n' % r2_score(Pdc_test,pred))

plt.figure()
plt.plot(Pdc_test.values)
plt.plot(pred, color='red')
plt.show()


