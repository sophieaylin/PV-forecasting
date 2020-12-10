import numpy as np
import matplotlib.pyplot as plt
import pandas as pa
import sklearn as sk
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def replace_NaN(var):
    
# Input var has to be of the class list. Replace with minimum value
      
    for i in range(0,len(var)):
      if var[i] == 'NaN\n':
        var.remove('NaN\n')
        var.insert(i,0.00001)

    return var

dezember = open (r'Daten/AMM_PT5M_201912_merge.dat', 'r')
d = dezember.readlines()
march = open (r'Daten/AMM_PT5M_202003_merge.dat', 'r')
m = march.readlines()
june = open (r'Daten/AMM_PT5M_202006_merge.dat', 'r')
j = june.readlines()

time = []
gti30t187a = []
GHI = []
Ta = []
BNI = []
Pdc = []
Indata_nlist = []

timemar = []
gti30t187amar = []
GHImar = []
Tamar = []
BNImar = []
Pdcmar = []
Indata_nlistmar = []

timejun = []
gti30t187ajun = []
GHIjun = []
Tajun = []
BNIjun = []
Pdcjun = []
Indata_nlistjun = []

for n in range(12,2672):
  linie = d[n]
  D = linie.split("\t")
  time.append(D[0])
  gti30t187a.append(D[9])
  GHI.append(D[4])
  Ta.append(D[5])
  BNI.append(D[23])
  Pdc.append(D[32])

for n in range(12,2672):
  linie = m[n]
  M = linie.split("\t")
  timemar.append(M[0])
  gti30t187amar.append(M[9])
  GHImar.append(M[4])
  Tamar.append(M[5])
  BNImar.append(M[23])
  Pdcmar.append(M[32])

for n in range(12,2672):
  linie = j[n]
  J = linie.split("\t")
  timejun.append(J[0])
  gti30t187ajun.append(J[9])
  GHIjun.append(J[4])
  Tajun.append(J[5])
  BNIjun.append(J[23])
  Pdcjun.append(J[32])

# datatype list

Pdc = replace_NaN(Pdc)
gti30t187a = replace_NaN(gti30t187a)
GHI = replace_NaN(GHI)
Ta = replace_NaN(Ta)

Pdc_float = np.array(Pdc, dtype=float)
gti30t187a_float = np.array(gti30t187a, dtype=float)
GHI_float = np.array(GHI, dtype=float)
Ta_float = np.array(Ta, dtype=float)

# Scaling trainings data: Normalization

Pdc_norm = Pdc_float/max(Pdc_float)
gti30t187a_norm = gti30t187a_float/max(gti30t187a_float)    
GHI_norm = GHI_float/max(GHI_float)
Ta_norm = Ta_float/max(Ta_float)

# data analysis

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(gti30t187a_norm, Pdc_norm, 'ro')
plt.xlabel('gti30t187a')
plt.ylabel('Pdc')
plt.subplot(122)
plt.boxplot(gti30t187a_float)
plt.suptitle('Normalized Data')
#plt.show()

# Trainingsdata Split

gti30t187a_train, gti30t187a_test, Pdc_train, Pdc_test = train_test_split(gti30t187a_norm, Pdc_norm)
sk.model_selection.train_test_split(Pdc_norm)

# Linear Regression

regr = linear_model.LinearRegression()
regr.fit(gti30t187a_train, Pdc_train)
Pdc_pred = regr.predict(gti30t187a_test)

print('Coefficients: \n', regr.coef_)
print('Mean squared error: %.2f' % mean_squared_error(Pdc_test,Pdc_pred))
print('Coefficient of determination: %.2f' % r2_score(Pdc_test,Pdc_pred))

# plt.scatter(gti30t187a_test, Pdc_test, color='red')
# plt.plot(gti30t187a_test, Pdc_pred, color='blue')
# plt.show