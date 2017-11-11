import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def visualize2(x1, x2):
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.scatter(x1, x2, marker=".")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Visualization")

    plt.show()