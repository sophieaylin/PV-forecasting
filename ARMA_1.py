import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pa
import sklearn as sk
import statsmodels as stat
from pandas import DataFrame
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from pandas.plotting import autocorrelation_plot, lag_plot
from statsmodels.tsa.arima.model import ARIMA
from DataManagement import get_data

data = get_data()

data_min = data

for Irr in ['GHI', 'DHI', 'gti30t187a', 'ENI']:
    data_min = data_min.drop(data_min.index[data_min[Irr]==0])

data_min = data_min.dropna(subset=['GHI', 'BNI', 'DHI', 'gti30t187a', 'ENI', 'Pdc_33'])
data_min = data_min.drop(data_min.index[data_min['El']<5])

time = data_min.t
gti30t187a = data_min.gti30t187a
GHI = data_min.GHI
Ta = data_min.Ta
BNI = data_min.BNI
Pdc = data_min.Pdc_33
El = data_min.El
Pdcmean = data_min.iloc[:, 109:].mean(axis=1)

Indata_nlist = []

# Scaling trainings data: Normalization

Pdc_norm = Pdc/max(Pdc)
gti30t187a_norm = gti30t187a/max(gti30t187a)    
GHI_norm = GHI/max(GHI)
Ta_norm = Ta/max(Ta)

# Input

"""for m in range(0,len(Pdc)):
    Indata_nlist.append([gti30t187a_norm[m], GHI_norm[m]])

Indata_array = np.array(Indata_nlist, dtype=float)"""

# Trainingsdata Split

#Indata_array_train, Indata_array_test = train_test_split(Indata_array)
#Pdc_train, Pdc_test = Pdc_norm[0:80000], Pdc_norm[80000:len(Pdc_norm)]
Pdc_train, Pdc_test = Pdc_norm[1:len(Pdc)-288], Pdc_norm[len(Pdc)-288:len(Pdc)]
#gti30t187a_train, gti30t187a_test = gti30t187a_norm[0:80000], gti30t187a_norm[80000:len(gti30t187a_norm)]
gti30t187a_train, gti30t187a_test = gti30t187a_norm[1:len(gti30t187a)-288],\
                                    gti30t187a_norm[len(gti30t187a)-288:len(gti30t187a)]

# ARMA Model walk-forward

model = ARIMA(Pdc_train.values, order=(30,1,6))
model_fit = model.fit()
print(model_fit.summary)

"""res = DataFrame(model_fit.resid)
res.plot()
plt.show()
res.plot(kind='kde')
plt.show()
print(res.describe())"""

hist = [x for x in Pdc_train]
pred = list()
for t in range(len(Pdc_test)):
      model = ARIMA(Pdc_train, gti30t187a_train, order=(40,1,6))
      model_fit = model.fit(maxiter=1000)
      output = model_fit.forecast()
      Pdc_pred = output[0]
      pred.append(Pdc_pred)
      obs = Pdc_test.values[t]
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

plt.plot(Pdc_test.values)
plt.plot(pred, color='red')
plt.show()





