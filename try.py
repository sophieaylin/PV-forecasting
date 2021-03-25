import pandas as pd
import matplotlib.pyplot as plt

filename = 'Daten/Irradiance_features_intra-hour.csv'
filename1 = 'Daten/Target_intra-hour.csv'
data = pd.read_csv(filename)
data1 = pd.read_csv(filename1)

print('done')