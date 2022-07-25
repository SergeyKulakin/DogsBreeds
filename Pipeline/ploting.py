import matplotlib.pyplot as plt
import numpy as np


def plot_history(train_history, val_history, title="loss"):
    '''INPUT
            -> train_history : история ибучения batch train
            -> val_history : история ибучения batch val
            -> title = "loss"
        OUTPUT
            -> plot results'''

    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label="train", zorder=1)

    points = np.array(val_history)
    steps = list(range(0, len(train_history) + 1, int(len(train_history) / len(val_history))))[1:]

    plt.scatter(steps, val_history, marker='+', s=180, c="orange", label="val", zorder=2)
    plt.xlabel("train steps")

    plt.legend(loc="best")
    plt.grid()

    plt.show()