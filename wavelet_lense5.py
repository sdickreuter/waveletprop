import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float32, int32, void, boolean
import cmath
from wavelets2 import *




num = 320

lense1 = Lense(x=0.6,y=0,height=2.0,num=num)

num = 100
ys = np.linspace(-1,1,num)
xs = np.repeat(1.61,num)
screen = Surface(np.vstack((xs,ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)

plt.plot(lense1.front.points[:,0],lense1.front.points[:,1])
plt.plot(lense1.back.points[:,0],lense1.back.points[:,1])
plt.plot(screen.points[:,0],screen.points[:,1])
plt.show()

num = 5000

rs = np.zeros((num,2))
rs[:,0] = np.repeat(0.0,num)
rs[:,1] = np.linspace(-0.9, 0.9, num)
ks = np.zeros((num,2))
ks[:,0] = np.repeat(1.0,num)
t0s = np.zeros((num))
phases = np.zeros((num))
planewave = Wavelets(r=rs, k=ks, t0=t0s, wavelength=0.1, phases=phases, mode=modes['gaussian'])
print("planewave: "+ str(planewave.n))



onlense1 = lense1.front.interact_with_all_wavelets(planewave)
print("onlense1: "+ str(onlense1.n))
onlense2 = lense1.back.interact_with_all_wavelets(onlense1)
print("onlense2: "+ str(onlense2.n))
onlense2.mode = modes['gaussian']
onscreen = screen.interact_with_all_wavelets(onlense2)
print("onscreen: "+ str(onscreen.n))

I = onlense2.calc_field(screen.points, 1.0,lense1.back.n2)
plt.plot(I**2)
plt.show()

# positions, field = screen.intensity_on_surface(onscreen)
# intensity = np.zeros(screen.points.shape[0]-1)
# hits = np.zeros(screen.points.shape[0]-1)
# for i in range(len(field)):
#     for j in range(len(intensity)):
#         if (positions[i,1] > screen.points[j,1]) and (positions[i,1] < screen.points[j+1,1]):
#             intensity[j] += field[i]
#             hits[j] += 1
#
#
# plt.plot(intensity**2)
# plt.show()
#
# plt.plot(hits)
# plt.show()



for i in range(50):

    onlense1.append_wavelets(lense1.front.interact_with_all_wavelets(planewave))
    #print("onlense1: "+ str(onlense1.n))
    onlense2.append_wavelets(lense1.back.interact_with_all_wavelets(onlense1))
    #print("onlense2: "+ str(onlense2.n))
    # onlense2.mode = modes['gaussian']
    onscreen.append_wavelets(screen.interact_with_all_wavelets(onlense2))
    print(str(i)+" onscreen: "+ str(onscreen.n))

    # I = onlense2.calc_field(screen.points, 1.0,lense1.back.n2)
    # plt.plot(I**2)
    # plt.show()

    positions, field = screen.intensity_on_surface(onscreen)

    plt.plot(field**2)
    plt.show()

    # plt.plot(hits)
    # plt.show()