import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pa
import sklearn as sk
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def replace_NaN(var): 
# Input var has to be of the class list. Replace with minimum value 
    for i in range(0,len(var)):
      if var[i] == 'NaN':
        var.remove('NaN')
        var.insert(i,0.01)
    return var

def replace_NaN_n(var): 
# Input var has to be of the class list. Replace with minimum value 
    for i in range(0,len(var)):
      if var[i] == 'NaN\n':
        var.remove('NaN\n')
        var.insert(i,0.01)
    return var

def replace_NaN_NN(var):
    # Input var has to be of the class list.    
    for i in range(0,len(var)):
      if (var[0] == 'NaN') and (i < len(var)):
        var.remove('NaN')
        while var[i] == 'NaN':
              continue
        var.insert(i,var[i+1])
      elif (var[i] == 'NaN') and (i <= len(var)):
        var.remove('NaN')
        var.insert(i,var[i-1])
    return var

def replace_NaN_n_NN(var):
    # Input var has to be of the class list.    
    for i in range(0,len(var)):
      if (var[0] == 'NaN\n') and (i < len(var)):
        var.remove('NaN\n')
        while var[i] == 'NaN\n':
              continue
        var.insert(i,var[i+1])
      elif (var[i] == 'NaN\n') and (i <= len(var)):
        var.remove('NaN\n')
        var.insert(i,var[i-1])
    return var

data = open (r'Daten/PVAMM_201911-202011_PT5M_merged.csv', 'r')
d = data.readlines()

time = []
gti30t187a = []
GHI = []
Ta = []
BNI = []
Pdc = []
Pdcmean = []
summe = 0.0
Pdcall = []
Indata_nlist = []

for n in range(1,113185):
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
  Pdcall.append(Pdcmean)


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
plt.subplot(131)
plt.plot(gti30t187a_norm, Pdc_norm, 'ro')
plt.xlabel('gti30t187a')
plt.ylabel('Pdc')
plt.subplot(132)
plt.plot(GHI_norm, Pdc_norm, 'go')
plt.xlabel('GHI')
plt.ylabel('Pdc')
plt.subplot(133)
plt.boxplot(gti30t187a_float, notch=1)
plt.suptitle('Normalized Data')

#plt.figure()
#lag_plot(gti30t187a_norm)

# Input

for m in range(0,len(Pdc)):
    Indata_nlist.append([gti30t187a_norm[m], GHI_norm[m]])

Indata_array = np.array(Indata_nlist, dtype=float)

# Trainingsdata Split

Indata_array_train, Indata_array_test, Pdc_train, Pdc_test = train_test_split(Indata_array, Pdc_norm)
sk.model_selection.train_test_split(Pdc_norm)