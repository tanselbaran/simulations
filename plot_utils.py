import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LFPy
from matplotlib.collections import PolyCollection, LineCollection

def plot_morphology_3d(cell):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_zlabel('z (um)')
    for i in range(len(cell.allsecnames)):
        if "soma" in cell.allsecnames[i]:
            color = 'r'
        elif "dend" in cell.allsecnames[i]:
            color = 'b'
        elif "axon" in cell.allsecnames[i]:
            color = 'k'
        plt.plot(cell.x3d[i], cell.y3d[i], cell.z3d[i], color)
    plt.show()

def plot_morphology_2d_crossection(cell, axes, *args):
    electrode  = args[0]
    plt.figure()

    first_axis = getattr(cell, '{:s}3d'.format(axes[0]))
    second_axis = getattr(cell, '{:s}3d'.format(axes[1]))

    electrode_position1 = getattr(electrode, axes[0])
    electrode_position2 = getattr(electrode, axes[1])

    for i in range(len(cell.allsecnames)):
        if "soma" in cell.allsecnames[i]:
            color = 'r'
        elif "dend" in cell.allsecnames[i]:
            color = 'b'
        elif "axon" in cell.allsecnames[i]:
            color = 'k'
        plt.plot(first_axis[i], second_axis[i], color)

    plt.plot(electrode_position1, electrode_position2, 'yo')

    plt.xlabel('{:s} (um)'.format(axes[0]))
    plt.ylabel('{:s} (um)'.format(axes[1]))
    plt.show()
