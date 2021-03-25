import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import random
import matplotlib.pyplot as plt
from DataManagement import get_data
from DataManagement import get_features

# trainingsset 1
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


# Trainings- Testset 2

data = get_features()
data = data.dropna(axis=0, how="any")
features = data.drop(['Pdc_5min', 'Pdc_10min', 'Pdc_15min', 'Pdc_20min', 'Pdc_25min', 'Pdc_30min',
                      'ENI', 'El', 'Pdc_33'], axis=1)
tar = pd.concat([data.t, data.Pdc_5min, data.Pdc_10min, data.Pdc_15min, data.Pdc_20min,
                 data.Pdc_25min, data.Pdc_30min, data.ENI, data.El], axis=1)
Pdc = data.Pdc_3

first = ["Sep", "Dec", "Mar", "Jun"]
second = ["Oct", "Jan", "Apr", "Jul"]
third = ["Nov","Feb", "May", "Aug"]

for t in [first, second, third]:
     herbst = features[features.t.str.contains(t[0])]
     winter = features[features.t.str.contains(t[1])]
     spring = features[features.t.str.contains(t[2])]
     summer = features[features.t.str.contains(t[3])]
     tarherbst = tar[tar.t.str.contains(t[0])]
     tarwinter = tar[tar.t.str.contains(t[1])]
     tarspring = tar[tar.t.str.contains(t[2])]
     tarsummer = tar[tar.t.str.contains(t[3])]

train = pd.concat([herbst[0:int(len(herbst)*0.8)], winter[0:int(len(winter)*0.8)],
               spring[0:int(len(spring)*0.8)], summer[0:int(len(summer)*0.8)]], axis=0)

test = pd.concat([herbst[int(len(herbst)*0.8):len(herbst)], winter[int(len(winter)*0.8):len(winter)],
               spring[int(len(spring)*0.8):len(spring)], summer[int(len(summer)*0.8):len(summer)]], axis=0)

train_y = pd.concat([tarherbst[0:int(len(tarherbst)*0.8)], tarwinter[0:int(len(tarwinter)*0.8)],
               tarspring[0:int(len(tarspring)*0.8)], tarsummer[0:int(len(tarsummer)*0.8)]], axis=0)

test_y = pd.concat([tarherbst[int(len(tarherbst)*0.8):len(tarherbst)], tarwinter[int(len(tarwinter)*0.8):len(tarwinter)],
               tarspring[int(len(tarspring)*0.8):len(tarspring)], tarsummer[int(len(tarsummer)*0.8):len(tarsummer)]], axis=0)

train = train.drop('t', axis=1)
test = test.drop('t', axis=1)
train_y = train_y.drop('t', axis=1)
test_y = test_y.drop('t', axis=1)

train_X = train.values
test_X = test.values
train_Y = train_y['Pdc_5min'].values
test_Y = test_y['Pdc_5min'].values

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

X = np.array([gti30t187a, Ta, BNI, wdir, kt], dtype=np.float)
y = np.array(Pdc, dtype=np.float)

# to torch
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# X_train, X_test = X[0,len(X)*0.8], X[len(X)*0.8,len(X)]

mini_batch_size = 64

for i in range(100):
    idx = np.random.randint(X.shape[0], size=mini_batch_size)
    x = X[:, idx]
    x = x.T
    y_pred = model(x)

    y_original = y[idx]
    loss = model.loss(y_pred, y)
    if i % 10 == 9:
        print(i, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('loss: ', loss.item())





