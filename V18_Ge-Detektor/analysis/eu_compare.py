import numpy as np
import matplotlib.pyplot as plt
from unbinner import unbinned_array

eu_1 = np.genfromtxt('../data/Europium.txt', unpack=True)[:5000]
eu_2 = np.genfromtxt('../data/Europium_proof.txt', unpack=True)[:5000]

bins = 200
fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
fig.subplots_adjust(hspace=0)
ax1.hist(unbinned_array(eu_1), bins=bins, histtype='stepfilled',
        edgecolor='none')
ax2.hist(unbinned_array(eu_2), bins=bins, histtype='stepfilled',
        edgecolor='none')

ax.set_yticks([])
ax.set_xticks([])
ax1.set_yticks([])
ax1.set_xticks([])
ax2.set_yticks([])

ax2.set_xlabel('Kanal')
ax.set_ylabel('Ereignisse')

fig.savefig('../build/plots/comparison.pdf')
fig.clf()

