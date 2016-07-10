#! /usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 10)
plt.style.use('ggplot')

from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

import pickle
import tqdm

from data_preparation import read_dataset
from neural_net import build_net, iterate_epochs, iterate_minibatches

import lasagne
import os

print('Reading Data.')
DATA, LABELS = read_dataset()
train_data, test_data, train_labels, test_labels = train_test_split(
    DATA, LABELS, test_size=0.3)

classifiers = {
    'RF': {
        'classifier': RandomForestClassifier(n_jobs=4),
    },
    'GB': {
        'classifier': GradientBoostingClassifier(),
    },
    'ADA': {
        'classifier': AdaBoostClassifier(),
    },
    'KNN': {
        'classifier': KNeighborsClassifier(n_jobs=4,
                                           n_neighbors=15),
    },
}

print('Training/Loading several ({}) classifiers now.'
      .format(len(classifiers)))
for name, d in classifiers.items():
    path = 'build/{}.pcl'.format(name)
    try:
        with open(path, 'rb') as f:
            cl = pickle.load(f)
            print('Found classifier object for {}'.format(name))
    except:
        with open(path, 'wb') as f:
            print('No object found. Writing classifier object for {}'
                  .format(name))
            cl = d['classifier']
            cl.fit(train_data, train_labels)
            pickle.dump(cl, f)
    probas = cl.predict_proba(test_data)[:, 1]
    curve = roc_curve(test_labels, probas)[:2]
    score = roc_auc_score(test_labels, probas)
    classifiers[name]['curve'] = curve
    classifiers[name]['score'] = score
    plt.plot(*curve, label='{}: {:2.2f}% ROC AUC'.format(name, score * 100))


print('Compiling Network.', end='')
train_fun, val_fun, nn = build_net(len(DATA.columns))
print(' Done.')
if os.path.isfile('build/NNet.pcl'):
    with open('build/NNet.pcl', 'rb') as f:
        params = pickle.load(f)
        lasagne.layers.set_all_param_values(nn, params)
else:
    print('No file found. Training Neural net now.')
    train_curve, val_auc_curve = iterate_epochs(train_fun, val_fun,
                                                train_data, train_labels,
                                                test_data, test_labels)
    params = lasagne.layers.get_all_param_values(nn)
    with open('build/NNet.pcl', 'wb') as f:
        pickle.dump(params, f)

Ypred_batches = []
Ytrue_batches = []

# this is the validation batch loop
for inputs, targets in iterate_minibatches(
        test_data.as_matrix(),
        test_labels.astype('bool').as_matrix(),
        1000,
        shuffle=False):
    err, y_pred = val_fun(inputs, targets)

    Ypred_batches.append(y_pred)
    Ytrue_batches.append(targets)

Ytrue_val = np.concatenate(Ytrue_batches)
Ypred_val = np.concatenate(Ypred_batches)
nnet_score = roc_auc_score(Ytrue_val, Ypred_val)
nnet_curve = roc_curve(Ytrue_val, Ypred_val)[:2]

plt.plot(*nnet_curve, label='NNet: {:2.2f}% ROC AUC'.format(nnet_score * 100))

plt.xlim(-0.05, 1.05)
plt.ylim(0, 1.05)
plt.legend(loc='best')
plt.savefig('build/plots/comparison.pdf')
plt.clf()
