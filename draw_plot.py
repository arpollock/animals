import numpy as np
import matplotlib.pyplot as plt
from irl3 import img_utils

filename = input("Enter filename: ")

rewards_maxent = np.load(filename, allow_pickle=True)
print(rewards_maxent.shape)
size = int((filename.split('_')[-1]).split('.')[0])
plt.figure(figsize=(size, size))
plt.subplot(1, 1, 1)
img_utils.heatmap2d(np.reshape(rewards_maxent,
                               (size, size),
                               order='F'), 'Reward Map - Maxent', block=False)
plt.show()
