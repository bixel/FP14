#! /usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 16
plt.style.use('ggplot')

from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

import pickle
import tqdm

from data_preparation import read_dataset
from neural_net import build_net, iterate_epochs, iterate_minibatches
from best_feature import mrmr_best_features

import lasagne
import os

print('Reading Data.')
DATA, LABELS = read_dataset()
train_data, test_data, train_labels, test_labels = train_test_split(
    DATA, LABELS, test_size=0.3, random_state=1)

classifiers = [
    {
        'name': 'RF',
        'classifier': RandomForestClassifier(n_jobs=4),
    },
    {
        'name': 'GB',
        'classifier': GradientBoostingClassifier(),
    },
    {
        'name': 'ADA',
        'classifier': AdaBoostClassifier(),
    },
    {
        'name': 'KNN',
        'classifier': KNeighborsClassifier(n_jobs=4,
                                           n_neighbors=15),
    },
    {
        'name': 'KNN20',
        'classifier': KNeighborsClassifier(n_jobs=4,
                                           n_neighbors=15),
        'cols': mrmr_best_features[:20],
    },
]

print('Training/Loading several ({}) classifiers now.'
      .format(len(classifiers)))

roc_fig = plt.figure(figsize=(5, 5))
pr_fig = plt.figure(figsize=(5, 5))
eff_fig = plt.figure(figsize=(5, 5))
for d in classifiers:
    name = d['name']
    if 'cols' in d:
        X_train = train_data[d['cols']]
        X_test = test_data[d['cols']]
    else:
        X_train = train_data
        X_test = test_data

    assert(X_train.shape[1] == X_test.shape[1])

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
            cl.fit(X_train, train_labels)
            pickle.dump(cl, f)
    probas = cl.predict_proba(X_test)[:, 1]
    curve = roc_curve(test_labels, probas)[:2]
    score = roc_auc_score(test_labels, probas)
    pr_curve = precision_recall_curve(test_labels, probas)
    d['curve'] = curve
    d['score'] = score
    label = '{}: {:2.2f}% ROC AUC'.format(name, score * 100)
    roc_fig.gca().plot(*curve, label=label)
    pr_fig.gca().plot(pr_curve[2], pr_curve[0][:-1], label=name)
    eff_fig.gca().plot(pr_curve[2], pr_curve[1][:-1], label=name)


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
nnet_pr_curve = precision_recall_curve(Ytrue_val, Ypred_val)

roc_fig.gca().plot(*nnet_curve, label='NNet: {:2.2f}% ROC AUC'.format(nnet_score * 100))
roc_fig.gca().plot([0, 1], '--', label='Random Selection')
roc_fig.gca().set_xlim(-0.05, 1.05)
roc_fig.gca().set_ylim(0, 1.05)
roc_fig.gca().set_xlabel('False positive rate')
roc_fig.gca().set_ylabel('True positive rate')
roc_fig.gca().legend(loc='best')
roc_fig.tight_layout()
roc_fig.savefig('build/plots/comparison.pdf')
roc_fig.clf()

pr_fig.gca().plot(nnet_pr_curve[2], nnet_pr_curve[0][:-1], label='NNet')
pr_fig.gca().set_ylim(0.5, 1.05)
pr_fig.gca().set_xlabel('Klassifizierungsschwelle')
pr_fig.gca().set_ylabel('Reinheit')
pr_fig.gca().legend(loc='best')
pr_fig.tight_layout()
pr_fig.savefig('build/plots/pr_comparison.pdf')
pr_fig.clf()

eff_fig.gca().plot(nnet_pr_curve[2], nnet_pr_curve[1][:-1], label='NNet')
eff_fig.gca().set_ylim(0, 1.05)
eff_fig.gca().set_xlabel('Klassifizierungsschwelle')
eff_fig.gca().set_ylabel('Effizienz')
eff_fig.gca().legend(loc='best')
eff_fig.tight_layout()
eff_fig.savefig('build/plots/eff_comparison.pdf')
eff_fig.clf()
