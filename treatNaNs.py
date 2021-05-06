import numpy as np
import pandas as pd


def IndicatorNaN(train_X, test_X, train_Y):
    for i, m in zip(range(1, train_X.shape[1] - 1, 2), range(0, train_X.shape[1], 2)):  # test und trainingsset?
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
    # test_Y = test_Y.fillna(value=0.00)

    return train_X, test_X, train_Y


def split_sequences(sequences, n_steps_in, n_steps_out):

    X, y , ENI, Pdc_sp = list(), list(), list(), list()

    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y, seq_ENI, seq_Pdc_sp = sequences[i:end_ix, 0:sequences.shape[1]-3],sequences[end_ix - 1:out_end_ix, -3],\
                                sequences[end_ix - 1:out_end_ix, -2], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
        ENI.append(seq_ENI)
        Pdc_sp.append(seq_Pdc_sp)

    print("done")

    return np.array(X), np.array(y), np.array(ENI), np.array(Pdc_sp)
