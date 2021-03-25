import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt 
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
wdir = data_min.wdir
kt = data_min.kt
Pdc = data_min.Pdc_33
Pdcmean = data.iloc[:, 109:].mean(axis=1)

# Scaling trainings data: Normalization

BNI_norm = BNI/max(BNI)   
Ta_norm = Ta/max(Ta)
Pdc_norm = Pdc/max(Pdc)
GHI_norm = GHI/max(GHI)
gti30t187a_norm = gti30t187a/max(gti30t187a)
wdir_norm = wdir/max(wdir)

# Test- Trainingsset

BNI_train, BNI_test = BNI[0:round(len(BNI)*0.8)], BNI[round(len(BNI)*0.8):len(BNI)]
Ta_train, Ta_test = Ta[0:round(len(Ta)*0.8)], Ta[round(len(Ta)*0.8):len(Ta)]
Pdc_train, Pdc_test = Pdc[0:round(len(Pdc)*0.8)], Pdc[round(len(Pdc)*0.8):len(Pdc)]
GHI_train, GHI_test = GHI[0:round(len(GHI)*0.8)], GHI[round(len(GHI)*0.8):len(GHI)]
gti30t187a_train, gti30t187a_test = gti30t187a[0:round(len(gti30t187a)*0.8)], \
                                    gti30t187a[round(len(gti30t187a)*0.8):len(gti30t187a)]
wdir_train, wdir_test = wdir[0:round(len(wdir)*0.8)], wdir[round(len(wdir)*0.8):len(wdir)]
kt_train, kt_test = kt[0:round(len(kt)*0.8)], kt[round(len(kt)*0.8):len(kt)]

"""BNI_train, BNI_test = BNI_norm[0:round(len(BNI_norm)*0.8)], BNI_norm[round(len(BNI_norm)*0.8):len(BNI_norm)]
Ta_train, Ta_test = Ta_norm[0:round(len(Ta_norm)*0.8)], Ta_norm[round(len(Ta_norm)*0.8):len(Ta_norm)]
Pdc_train, Pdc_test = Pdc_norm[0:round(len(Pdc_norm)*0.8)], Pdc_norm[round(len(Pdc_norm)*0.8):len(Pdc_norm)]
GHI_train, GHI_test = GHI_norm[0:round(len(GHI_norm)*0.8)], GHI_norm[round(len(GHI_norm)*0.8):len(GHI_norm)]
gti30t187a_train, gti30t187a_test = gti30t187a_norm[0:round(len(gti30t187a_norm)*0.8)], \
                                    gti30t187a_norm[round(len(gti30t187a_norm)*0.8):len(gti30t187a_norm)]
wdir_train, wdir_test = wdir_norm[0:round(len(wdir_norm)*0.8)], wdir_norm[round(len(wdir_norm)*0.8):len(wdir_norm)]
kt_train, kt_test = kt[0:round(len(kt)*0.8)], kt[round(len(kt)*0.8):len(kt)]"""

# Neural Network

class TestNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(in_features=5, out_features=64)
        self.ln2 = nn.Linear(in_features=64, out_features=128)
        self.ln3 = nn.Linear(128, 32)
        self.ln4 = nn.Linear(32, 1)

        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        x = torch.sigmoid(self.ln1(x))
        x = torch.sigmoid(self.ln2(x))
        x = torch.sigmoid(self.ln3(x))
        x = (self.ln4(x))
        return x

    def loss(self, x, y):
        loss = torch.sqrt(self.criterion(x, y))
        return loss

model = TestNN()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

X = np.array([gti30t187a_train, Ta_train, BNI_train, wdir_train, kt_train], dtype=np.float)
y = np.array(Pdc_train, dtype=np.float)
y_test = np.array(Pdc_test, dtype=np.float)

# to torch
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
y_test = torch.from_numpy(y_test).float()

mini_batch_size = 64

for i in range(1000000):
    idx = np.random.randint(X.shape[0], size=mini_batch_size)
    x = X[:, idx]
    x = x.T
    y_pred = model(x)

    y_original = y_test[idx]
    y_original = y_original.unsqueeze(1)
    loss = model.loss(y_pred, y_original)
    if i % 10000 == 9999:
        print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('loss: ', loss.item())





