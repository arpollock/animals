import numpy as np
import matplotlib.pyplot as plt
from irl3 import img_utils

rewards_maxent = np.load("50x50rewards.npy", allow_pickle=True)

plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
img_utils.heatmap2d(np.reshape(rewards_maxent,
                               (21, 21),
                               order='F'), 'Reward Map - Maxent', block=False)
plt.show()
