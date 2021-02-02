import pandas as pd

filename = 'Daten/PVAMM_201911-202011_PT5M_merged.csv'
data = pd.read_csv(filename)

if


def get_data(deep_copy=True):
    return data.copy(deep_copy)

print('done')