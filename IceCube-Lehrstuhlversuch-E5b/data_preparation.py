import pandas as pd
import numpy as np


def remove_constant_columns(df):
    data = df.loc[:, (df != df.ix[0]).any()]
    return data


def combine_data(sig, bkg):
    ind = sig.columns.isin(bkg.columns)
    column_ind = sig.columns[ind]
    data = sig[column_ind].append(bkg[column_ind], ignore_index=True)
    labels = np.append(np.ones(sig.shape[0]), np.zeros(bkg.shape[0]))
    return data, labels


def read_dataset(filter_data=True):
    SIG = pd.read_csv('data/signal.csv', sep=';')
    BKG = pd.read_csv('data/background.csv', sep=';')

    SIG = remove_constant_columns(SIG)
    BKG = remove_constant_columns(BKG)

    DATA, LABELS = combine_data(SIG, BKG)
    DATA = DATA.dropna(axis=1)
    LABELS = pd.Series(LABELS.astype('int'))

    if filter_data:
        DATA.drop(DATA.filter(regex='MC|Weight|EventHeader|ID').columns,
                  axis=1,
                  inplace=True)

    return DATA, LABELS
