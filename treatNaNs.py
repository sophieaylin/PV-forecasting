import numpy as np
import pandas as pd

def IndicatorNaN(train_X, test_X, train_Y, test_Y):

    for i, m in zip(range(1, train_X.shape[1]-1, 2), range(0, train_X.shape[1], 2)): # test und trainingsset?
        bool = train_X[train_X.columns[m]].isna()
        bool = np.multiply(bool, 1)
        train_X.insert(i, "Indicator_{}".format(m), value=bool)
        bool = test_X[test_X.columns[m]].isna()
        bool = np.multiply(bool, 1)
        test_X.insert(i, "Indicator_{}".format(m), value=bool)

    train_X = train_X.fillna(value=0.00)
    test_X = test_X.fillna(value=0.00)

    """for i, m in zip(range(1, train_Y.shape[1]-1, 2), range(0, train_Y.shape[1], 2)):
        bool = train_Y[train_Y.columns[m]].isna()
        bool = np.multiply(bool, 1)
        train_Y.insert(i, "Indicator_{}".format(m), value=bool)"""

    train_Y = train_Y.fillna(value=0.00)
    test_Y = test_Y.fillna(value=0.00)

    return train_X, test_X, train_Y, test_Y

