import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt 

def replace_NaN(var):

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

data = open (r'Daten/AMM_PT5M_201912_merge.dat', 'r')
r = data.readlines()

time = []
Patm = []
GHI = []
Ta = []
BNI = []
Pdc = []

for n in range(12,2672):
  linie = r[n]
  d = linie.split("\t")
  time.append(d[0])
  Patm.append(d[1])
  GHI.append(d[4])
  Ta.append(d[5])
  BNI.append(d[23])
  Pdc.append(d[32])

Pdc = replace_NaN(Pdc)
BNI = replace_NaN(BNI)
Ta = replace_NaN(Ta)
GHI = replace_NaN(GHI)

Pdc_float = np.array(Pdc, dtype=float)
BNI_float = np.array(BNI, dtype=float)
Ta_float = np.array(Ta, dtype=float)
GHI_float = np.array(GHI, dtype=float)

# Scaling trainings data: Normalization

BNI_norm = BNI_float/max(BNI_float)    # np.max(BNI_float) = nan, aber max(BNI_float) = 726.3 ?
Ta_norm = Ta_float/np.max(Ta_float)
Pdc_norm = Pdc_float/np.max(Pdc_float)
GHI_norm = GHI_float/np.max(GHI_float)

Indata_mat = np.matrix([GHI_norm, Ta_norm])
Indata = Indata_mat.transpose()
#print(Indata)
#print(max(GHI_norm))

class neural_network(object):
      def __init__(self):
            self.inputSize = 2
            self.outputSize = 1
            self.hiddenSize = 3
            self.W1 = np.random.rand(self.inputSize, self.hiddenSize)   # (2x3)
            self.W2 = np.random.rand(self.hiddenSize, self.outputSize)  # (3x1)

      def sigmoid(self, s):  #activation function
            return 1/(1+np.exp(-s))
      
      def forward(self, Indata):
            self.z = np.dot(Indata, self.W1)
            self.z2 = self.sigmoid(self.z)
            self.z3 = np.dot(self.z2, self.W2)
            out = self.sigmoid(self.z3)
            return out

nn = neural_network()
out = nn.forward(Indata)

print("Predicted Output: \n" + str(out))
print("Actual Output: \n" + str(Pdc))

perceptrons = nl.net.newff([[0,1],[0,1]], [3,1])
nl.train.train_gd(perceptrons)

# def forward(self,)
# Input = [BNI, Ta]
# plt.figure()
# plt.scatter(BNI, Ta)
# plt.xlabel('BNI')
# plt.ylabel('Ta')
# plt.savefig("graphBNI_Ta.png")
# plt.scatter(BNI, Pdc)
# plt.xlabel('BNI')
# plt.ylabel('Pdc')
# plt.savefig("graphBNI_Pdc.png")

