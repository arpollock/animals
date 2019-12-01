import numpy as np
import matplotlib.pyplot as plt


def draw_plot(filename):
    """Draws a reward map specified in filename
    """

    rewards_maxent = np.load(filename, allow_pickle=True)
    print(rewards_maxent.shape)
    size = (filename.split('_')[1]).split('x')
    size = (int(size[0]), int(size[1]))
    plt.imshow(np.reshape(rewards_maxent, size, order='F'), cmap='viridis')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    filename = input("Enter filename: ")
    draw_plot(filename)
