import numpy as np
import matplotlib.pyplot as plt

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
DHI = []
ENI = []
CSGHI = []
CSBNI = []
CSDHI = []
TL = []
CS = []
vw = []
wdir = []
Patm = []
RH = []
tpw = []
AMa = []
kd = []
kt = []
Az = []
El = []
w = []
dec = []
Indata_nlist = []

timemar = []
gti30t187amar = []
GHImar = []
Tamar = []
BNImar = []
Pdcmar = []
DHImar = []
ENImar = []
CSGHImar = []
CSBNImar = []
CSDHImar = []
TLmar = []
CSmar = []
vwmar = []
wdirmar = []
Patmmar = []
RHmar = []
tpwmar = []
AMamar = []
kdmar = []
ktmar = []
Azmar = []
Elmar = []
wmar = []
decmar = []
Indata_nlistmar = []

timejun = []
gti30t187ajun = []
GHIjun = []
Tajun = []
BNIjun = []
Pdcjun = []
DHIjun = []
ENIjun = []
CSGHIjun = []
CSBNIjun = []
CSDHIjun = []
TLjun = []
CSjun = []
vwjun = []
wdirjun = []
Patmjun = []
RHjun = []
tpwjun = []
AMajun = []
kdjun = []
ktjun = []
Azjun = []
Eljun = []
wjun = []
decjun = []
Indata_nlistjun = []

Indata_all_nlist = []

for n in range(12,2672):
  linie_d = d[n]
  D = linie_d.split("\t")
  time.append(D[0])
  gti30t187a.append(D[9])
  GHI.append(D[8])
  Ta.append(D[5])
  BNI.append(D[23])
  Pdc.append(D[32])
  DHI.append(D[22])
  ENI.append(D[19])
  CSGHI.append(D[14])
  CSBNI.append(D[12])
  CSDHI.append(D[13])
  TL.append(D[20])
  CS.append(D[18])
  vw.append(D[7])
  wdir.append(D[6])
  Patm.append(D[1])
  RH.append(D[4])
  tpw.append(D[15])
  AMa.append(D[21])
  kd.append(D[25])
  kt.append(D[24])
  Az.append(D[27])
  El.append(D[28])
  w.append(D[29])
  dec.append(D[30])

  linie_m = m[n]
  M = linie_m.split("\t")
  timemar.append(M[0])
  gti30t187amar.append(M[9])
  GHImar.append(M[8])
  Tamar.append(M[5])
  BNImar.append(M[23])
  Pdcmar.append(M[32])
  DHImar.append(M[22])
  ENImar.append(M[19])
  CSGHImar.append(M[14])
  CSBNImar.append(M[12])
  CSDHImar.append(M[13])
  TLmar.append(M[20])
  CSmar.append(M[18])
  vwmar.append(M[7])
  wdirmar.append(M[6])
  Patmmar.append(M[1])
  RHmar.append(M[4])
  tpwmar.append(M[15])
  AMamar.append(M[21])
  kdmar.append(M[25])
  ktmar.append(M[24])
  Azmar.append(M[27])
  Elmar.append(M[28])
  wmar.append(M[29])
  decmar.append(M[30])

  linie_n = j[n]
  J = linie_n.split("\t")
  timejun.append(J[0])
  gti30t187ajun.append(J[9])
  GHIjun.append(J[8])
  Tajun.append(J[5])
  BNIjun.append(J[23])
  Pdcjun.append(J[32])
  DHIjun.append(J[22])
  ENIjun.append(J[19])
  CSGHIjun.append(J[14])
  CSBNIjun.append(J[12])
  CSDHIjun.append(J[13])
  TLjun.append(J[20])
  CSjun.append(J[18])
  vwjun.append(J[7])
  wdirjun.append(J[6])
  Patmjun.append(J[1])
  RHjun.append(J[4])
  tpwjun.append(J[15])
  AMajun.append(J[21])
  kdjun.append(J[25])
  ktjun.append(J[24])
  Azjun.append(J[27])
  Eljun.append(J[28])
  wjun.append(J[29])
  decjun.append(J[30])

# Data of March, June and Dezember

time_all = timemar + timejun + time
gti30t187a_all = gti30t187amar + gti30t187ajun + gti30t187a
GHI_all = GHImar + GHIjun + GHI
Ta_all = Tamar + Tajun+ Ta
BNI_all = BNImar + BNIjun + BNI
Pdc_all = Pdcmar + Pdcjun + Pdc
DHI_all = DHImar + DHIjun + DHI
ENI_all = ENImar + ENIjun + ENI
CSGHI_all = CSGHImar + CSGHIjun + CSGHI
CSBNI_all = CSBNImar + CSBNIjun + CSBNI
CSDHI_all = CSDHImar + CSDHIjun + CSDHI
TL_all = TLmar + TLjun + TL
CS_all = CSmar + CSjun + CS
vw_all = vwmar + vwjun + vw
wdir_all = wdirmar + wdirjun + wdir
Patm_all = Patmmar + Patmjun + Patm
RH_all = RHmar + RHjun + RH
tpw_all = tpwmar + tpwjun + tpw
AMa_all = AMamar + AMajun + AMa
kd_all = kdmar + kdjun + kd
kt_all = ktmar + ktjun + kt
Az_all = Azmar + Azjun + Az
El_all = Elmar + Eljun + El
w_all = wmar + wjun + w
dec_all = decmar + decjun + dec

print(time_all)

Pdc_all = replace_NaN_n(Pdc_all)
gti30t187a_all = replace_NaN(gti30t187a_all)
GHI_all = replace_NaN(GHI_all)
Ta_all = replace_NaN(Ta_all)
BNI_all = replace_NaN(BNI_all)
DHI_all = replace_NaN(DHI_all)
ENI_all = replace_NaN(ENI_all)
CSGHI_all = replace_NaN(CSGHI_all)
CSBNI_all = replace_NaN(CSBNI_all)
CSDHI_all = replace_NaN(CSDHI)
TL_all = replace_NaN(TL_all)
CS_all = replace_NaN(CS_all)
vw_all = replace_NaN(vw_all)
wdir_all = replace_NaN(wdir_all)
Patm_all = replace_NaN(Patm_all)
RH_all = replace_NaN(RH_all)
tpw_all = replace_NaN(tpw_all)
AMa_all = replace_NaN(AMa_all)
kd_all = replace_NaN(kd_all)
kt_all = replace_NaN(kt_all)
Az_all = replace_NaN(Az_all)
El_all = replace_NaN(El_all)
w_all = replace_NaN(w_all)
dec_all = replace_NaN(dec_all)

Pdc_all_float = np.array(Pdc_all, dtype=float)
gti30t187a_all_float = np.array(gti30t187a_all, dtype=float)
GHI_all_float = np.array(GHI_all, dtype=float)
Ta_all_float = np.array(Ta_all, dtype=float)
BNI_all_float = np.array(BNI_all, dtype=float)
DHI_all_float = np.array(DHI_all, dtype=float)
ENI_all_float = np.array(ENI_all, dtype=float)
CSGHI_all_float = np.array(CSGHI_all, dtype=float)
CSBNI_all_float = np.array(CSBNI_all, dtype=float)
CSDHI_all_float = np.array(CSDHI, dtype=float)
TL_all_float = np.array(TL_all, dtype=float)
CS_all_float = np.array(CS_all, dtype=float)
vw_all_float = np.array(vw_all, dtype=float)
wdir_all_float = np.array(wdir_all, dtype=float)
Patm_all_float = np.array(Patm_all, dtype=float)
RH_all_float = np.array(RH_all, dtype=float)
tpw_all_float = np.array(tpw_all, dtype=float)
AMa_all_float = np.array(AMa_all, dtype=float)
kd_all_float = np.array(kd_all, dtype=float)
kt_all_float = np.array(kt_all, dtype=float)
Az_all_float = np.array(Az_all, dtype=float)
El_all_float = np.array(El_all, dtype=float)
w_all_float = np.array(w_all, dtype=float)
dec_all_float = np.array(dec_all, dtype=float)

Pdc_all_norm = Pdc_all_float/max(Pdc_all_float)
gti30t187a_all_norm = gti30t187a_all_float/max(gti30t187a_all_float)
GHI_all_norm = GHI_all_float/max(GHI_all_float)
Ta_all_norm = Ta_all_float/max(Ta_all_float)
BNI_all_norm = BNI_all_float/max(BNI_all_float)
DHI_all_norm = DHI_all_float/max(DHI_all_float)
ENI_all_norm  = ENI_all_float/max(ENI_all_float)
CSGHI_all_norm  = CSGHI_all_float/max(CSGHI_all_float)
CSBNI_all_norm  = CSBNI_all_float/max(CSGHI_all_float)
CSDHI_all_norm  = CSDHI_all_float/max(CSDHI_all_float)
TL_all_norm  = TL_all_float/max(TL_all_float)
CS_all_norm  = CS_all_float/max(CS_all_float)
vw_all_norm  = vw_all_float/max(vw_all_float)
wdir_all_norm  = wdir_all_float/max(wdir_all_float)
Patm_all_norm  = Patm_all_float/max(Patm_all_float)
RH_all_norm  = RH_all_float/max(RH_all_float)
tpw_all_norm  = tpw_all_float/max(tpw_all_float)
AMa_all_norm  = AMa_all_float/max(AMa_all_float)
kd_all_norm  = kd_all_float/max(kd_all_float)
kt_all_norm  = kt_all_float/max(kt_all_float)
Az_all_norm  = Az_all_float/max(Az_all_float)
El_all_norm  = El_all_float/max(El_all_float)
w_all_norm  = w_all_float/max(w_all_float)
dec_all_norm  = dec_all_float/max(dec_all_float)

# datatype list

Pdc = replace_NaN_n_NN(Pdc)
gti30t187a = replace_NaN_NN(gti30t187a)
GHI = replace_NaN_NN(GHI)
Ta = replace_NaN_NN(Ta)

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

plt.figure(figsize=(15,5))
plt.subplot(151)
plt.plot(gti30t187a_all_norm, Pdc_all_norm, 'ro')
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.xlim(right=1)
plt.xlabel('gti30t187a')
plt.ylabel('Pdc')
plt.subplot(152)
plt.plot(GHI_all_norm, Pdc_all_norm, 'go')
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.xlim(right=1)
plt.xlabel('GHI')
plt.subplot(153)
plt.plot(BNI_all_norm, Pdc_all_norm, 'bo')
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.xlim(right=1)
plt.xlabel('BNI')
plt.subplot(154)
plt.plot(ENI_all_norm, Pdc_all_norm, 'yo')
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.xlim(right=1)
plt.xlabel('ENI')
plt.subplot(155)
plt.plot(DHI_all_norm, Pdc_all_norm, 'mo')
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.xlim(left=0)
# plt.xlim(right=1)
plt.xlabel('DNI')
#plt.show()
#plt.savefig('Irradiation')

# plt.subplot(153)
# plt.boxplot(gti30t187a_float, notch=1)
# plt.suptitle('Normalized Data')