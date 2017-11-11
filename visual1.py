import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


def visualize1(data):
    xrp = []
    yrp = []
    xrn = []
    yrn = []
    xpp = []
    ypp = []
    xpn = []
    ypn = []

    for i in xrange(0, len(data)):
        if data[i][2] > 0:
            xrp.append(data[i][0])
            yrp.append(data[i][1])
        else:
            xrn.append(data[i][0])
            yrn.append(data[i][1])
        if data[i][3] > 0:
            xpp.append(data[i][0])
            ypp.append(data[i][1])
        else:
            xpn.append(data[i][0])
            ypn.append(data[i][1])

    plt.figure(1)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.scatter(xrn, yrn, marker='o', color='r', label="Real negative")
    plt.scatter(xrp, yrp, marker=9, color='g', label="Real positive")
    plt.xlabel("x2")
    plt.ylabel("x1")
    plt.title("Real classification")
    plt.legend()

    plt.figure(2)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.scatter(xpn, ypn, marker='o', color='r', label="Predicted negative")
    plt.scatter(xpp, ypp, marker=9, color='g', label="Predicted positive")
    plt.xlabel("x2")
    plt.ylabel("x1")
    plt.title("Predicted classification")
    plt.legend()

    plt.show()