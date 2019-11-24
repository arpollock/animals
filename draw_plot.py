import numpy as np
import matplotlib.pyplot as plt
from irl3 import img_utils

filename = input("Enter filename: ")

rewards_maxent = np.load(filename, allow_pickle=True)

plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
img_utils.heatmap2d(np.reshape(rewards_maxent,
                               (int(filename.split['_'][-1]))**0.5,
                               order='F'), 'Reward Map - Maxent', block=False)
plt.show()
