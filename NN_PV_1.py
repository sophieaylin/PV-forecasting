import numpy as np
import neurolab as nl
import torch 
import matplotlib.pyplot as plt 

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

def replace_NaN(var): 
# Input var has to be of the class list. Replace with minimum value 
    for i in range(0,len(var)):
      if var[i] == 'NaN':
        var.remove('NaN')
        var.insert(i,0.01)
    return var

# GHI - Global Horizontal Irradiance [W/m²], data[8]
# BNI - Beam Normal Irradiance (modeled) [W/m²], data[23]
# DHI - Diffuse Horizontal Irradiance (modeled) [W/m²], data[22]
# gti30t187a - Global Irradiance on a tilted plane (30° tilt, 187° azimuth) [W/m²], data[9]
# ENI - Extraterrestrial Normal Irradiance [W/m²], data[19]
# 
# CSGHI, CSBNI, CSDHI - (modeled) Clear-sky irradiances [W/m²], data[14], data[12], data[13]
# TL - (modeled) Linke Turbidity Factor, data[20]
# CS - boolean flag (modeled), does sky seem clear? (from GHI), data[18]
#
# Ta - Ambient temperature (°C), data[5]
# vw - wind speed [m/s], data[7]
# wdir - wind direction (degrees), data[6]
# Patm - ambient pressure [Pa], data[1]
# RH - relative humidity [0-1], data[4]
# tpw - Total Precipitable Water [mm], data[15]
# 
# AMa - Absolute Air-Mass (equivalent clean atmospheres), data[21]
# kd - Diffuse fraction kd = DHI/GHI, data[25]
# kt - Clearness Index kt = GHI/(ENI·cos(zenith)), data[24]
# 
# Az - Solar azimuth angle (degrees) - convention: Equator = 90°, positive around zenith., data[27]
# El - Apparent Solar elevation angle (degrees), considering atmospheric refraction, data[28]
# w - Solar hour angle (degrees), data[29]
# dec - Solar declination angle (degrees), data[30]
# 
# Pdc - Measured plant output (mean of 40 monitored inverters) [W], data[32]
# PR - Performance Ratio: output / nominal installed power (W/Wp), data[31]

# time - data[0]
# windir - , data[10]

# Import testing data

data = open (r'Daten/PVAMM_201911-202011_PT5M_merged.csv', 'r')
r = data.readlines()

time = []
Patm = []
GHI = []
Ta = []
BNI = []
Pdc = []
Indata_nlist = []

for n in range(1,len(r)):
  linie = r[n]
  d = linie.split(",")
  time.append(d[0])
  Patm.append(d[1])
  GHI.append(d[4])
  Ta.append(d[5])
  BNI.append(d[23])
  Pdc.append(d[32])

# datatype list

Pdc = replace_NaN(Pdc)
BNI = replace_NaN(BNI)
Ta = replace_NaN(Ta)
GHI = replace_NaN(GHI)

# Indata_list = [GHI, Ta]   # might delete later

# for row in Indata_list:
#     for k in (1,2,3,5):
#         row[k] = float(row[k])

# Inputdata = map(list, zip(*Indata_list))    # class map

# for m in range(0,len(Pdc)):
#   Indata_nlist.append([GHI[m], Ta[m]])      # class nested list

# datatype nd.array (numpy.ndarray)/ numpy.matrix

Pdc_float = np.array(Pdc, dtype=float)
BNI_float = np.array(BNI, dtype=float)
Ta_float = np.array(Ta, dtype=float)
GHI_float = np.array(GHI, dtype=float)

# Scaling trainings data: Normalization

BNI_norm = BNI_float/max(BNI_float)   
Ta_norm = Ta_float/max(Ta_float)
Pdc_norm = Pdc_float/max(Pdc_float)
GHI_norm = GHI_float/max(GHI_float)


Indata_mat = np.matrix([GHI_norm, Pdc_norm])
Indata = Indata_mat.transpose()         # class numpy.matirx

Indata_array = np.array(Indata_nlist, dtype=float)    # class Numpy.ndarray
#print(Indata_array)

# Neural Network

perceptrons = nl.net.newff([[0,1],[0,1]], [3,1])
nl.train.train_gd(Indata_array, Pdc)
