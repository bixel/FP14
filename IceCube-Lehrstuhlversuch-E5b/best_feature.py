#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from data_preparation import read_dataset


plt.style.use('ggplot')

DATA, LABELS = read_dataset()

mrmr_order, mrmr_feature_index, mrmr_feature_name, mrmr_score =\
        np.genfromtxt('./data/mrmr_output.txt',
                      unpack=True,
                      skip_header=6,
                      max_rows=50,)
mrmr_feature_index -= 1
mrmr_best_features = DATA.columns[mrmr_feature_index.astype(int)]

for i, f in enumerate(mrmr_best_features[:5]):
    _, bins = np.histogram(DATA[f], 30, range=(-600, 600))
    plt.figure(figsize=(4, 4))
    plt.hist(DATA[LABELS==1][f], bins=bins, alpha=0.5)
    plt.hist(DATA[LABELS==0][f], bins=bins, alpha=0.5)
    plt.title(f[:32])
    plt.tight_layout()
    plt.savefig('build/plots/best-{}.pdf'.format(i))
    plt.clf()
