import matplotlib.pyplot as plt
import os
import numpy as np


def plot_fig(fig_name, x, ys, y_names, bound_value, save_dir=None, show_fig=True):
    # init figure
    fig = plt.figure(figsize=(10, 10))
    plt.title(fig_name)
    plt.xlabel("epochs")

    # draw
    for i, y_name in enumerate(y_names):
        y = ys[i]
        # min max scatter
        y_max_index = np.argmax(y)
        y_min_index = np.argmin(y)
        plt.text(x[y_max_index], y[y_max_index], f"{y_max_index}, {y[y_max_index]:.2f}", size='small')
        plt.text(x[y_min_index], y[y_min_index], f"{y_min_index}, {y[y_min_index]:.2f}", size='small')
        plt.scatter([x[y_max_index], x[y_min_index]], [y[y_max_index], y[y_min_index]], s=25)
        # line
        plt.plot(x, y, label=y_name)
    margin = [bound_value] * len(x)
    plt.plot(x, margin)
    plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, fig_name + ".png"))
    if show_fig:
        plt.show()
