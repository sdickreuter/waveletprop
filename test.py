import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import matplotlib as mpl

mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float32, int32, void, boolean
import cmath
from wavelets2 import *
import progress

@jit
def make_planewave(num,mode=modes['ray']):
    rs = np.zeros((num, 2))
    rs[:, 0] = np.repeat(0.0, num)

    #rs[:, 1] = np.random.rand(num)
    rs[:, 1] = np.linspace(0,1,num)
    rs[:, 1] *= 2
    rs[:, 1] -= 1

    ks = np.zeros((num, 2))
    ks[:, 0] = np.repeat(1.0, num)
    t0s = np.zeros((num))
    phases = np.zeros((num))
    return Wavelets(r=rs, k=ks, t0=t0s, wavelength=0.1, phases=phases, mode=mode)


num = 101
ys = np.linspace(-1, 1, num)
xs = np.repeat(1.0, num)
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)

num = 100

planewave = make_planewave(num)


plt.plot(planewave.r[:, 0], planewave.r[:, 1])
for i in range(planewave.n):
    plt.plot(planewave.r[i,0], planewave.r[i,1], "bo")
    plt.arrow(planewave.r[i,0], planewave.r[i,1], planewave.k[i,0], planewave.k[i,1])
plt.plot(screen.points[:, 0], screen.points[:, 1])
plt.show()


onscreen = screen.interact_with_all_wavelets(planewave)

print("onscreen: " + str(onscreen.n))
screen.add_field_from_wavelets(onscreen)

plt.plot(onscreen.t0)
plt.show()

plt.plot(screen.field)
plt.show()
