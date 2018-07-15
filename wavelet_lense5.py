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
def make_planewave(num):
    rs = np.zeros((num, 2))
    rs[:, 0] = np.repeat(0.0, num)

    #rs[:, 1] = np.random.normal(0,0.5,num)
    #rs[:, 1] = np.random.rand(num)
    rs[:, 1] = np.linspace(0,1, num)
    rs[:, 1] *= 2
    rs[:, 1] -= 1

    ks = np.zeros((num, 2))
    ks[:, 0] = np.repeat(1.0, num)
    t0s = np.zeros((num))
    phases = np.zeros((num))
    return Wavelets(r=rs, k=ks, t0=t0s, wavelength=0.1, phases=phases, mode=modes['ray'])


plotit = False

num = 100

lense1 = Lense(x=0.5, y=0, height=2.0, num=num)

num = 201
ys = np.linspace(-0.5, 0.5, num)
xs = np.repeat(2.65, num)
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
screen.flip_normals()

num = 201
xs = np.linspace(1.0, 3, num)
ys = np.repeat(0, num)
screen2 = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
screen2.flip_normals()



plt.plot(lense1.front.points[:, 0], lense1.front.points[:, 1])
plt.plot(lense1.back.points[:, 0], lense1.back.points[:, 1])
plt.plot(screen.points[:, 0], screen.points[:, 1])
plt.plot(screen2.points[:, 0], screen2.points[:, 1])
plt.show()

num = 300

planewave = make_planewave(num)
print("planewave: " + str(planewave.n))

onlense_front = lense1.front.interact_with_all_wavelets(planewave)
# onlense_front.mode = modes['gaussian']
print("onlense1: " + str(onlense_front.n))
onlense_back = lense1.back.interact_with_all_wavelets(onlense_front)
print("onlense2: " + str(onlense_back.n))
onlense_back.mode = modes['gaussian']
onscreen = screen.interact_with_all_wavelets(onlense_back)
onscreen2 = screen2.interact_with_all_wavelets(onlense_back)

print("onscreen: " + str(onscreen.n))
screen.add_field_from_wavelets(onscreen)
screen2.add_field_from_wavelets(onscreen2)

if plotit:
    plt.plot(onlense_front.t0)
    plt.show()

    plt.plot(onlense_back.t0)
    plt.show()

    plt.plot(onscreen.t0)
    plt.show()


    divider = 300
    #plt.plot(lense1.front.points[:, 0], planewave.r[:, 1])
    for i in range(planewave.n):
        plt.plot(planewave.r[i,0], planewave.r[i,1], "bo")
        plt.arrow(planewave.r[i,0], planewave.r[i,1], planewave.k[i,0]/divider, planewave.k[i,1]/divider)
    plt.plot(lense1.front.points[:, 0], lense1.front.points[:, 1])
    for i in range(onlense_front.n):
        plt.plot(onlense_front.r[i,0], onlense_front.r[i,1], "bo")
        plt.arrow(onlense_front.r[i,0], onlense_front.r[i,1], onlense_front.k[i,0]/divider, onlense_front.k[i,1]/divider)
    plt.plot(lense1.back.points[:, 0], lense1.back.points[:, 1])
    for i in range(onlense_back.n):
        plt.plot(onlense_back.r[i, 0], onlense_back.r[i, 1], "bo")
        plt.arrow(onlense_back.r[i, 0], onlense_back.r[i, 1], onlense_back.k[i, 0] / divider, onlense_back.k[i, 1] / divider)
    plt.plot(screen.points[:, 0], screen.points[:, 1])
    for i in range(onscreen.n):
        plt.plot(onscreen.r[i, 0], onscreen.r[i, 1], "bo")
        plt.arrow(onscreen.r[i, 0], onscreen.r[i, 1], onscreen.k[i, 0] / divider, onscreen.k[i, 1] / divider)
    #plt.plot(onlense_back.points[:, 0], onlense_back.points[:, 1])
    plt.show()



# I = onlense2.calc_field(screen.points, 1.0,lense1.back.n2)
# plt.plot(I**2)
# plt.show()

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

iterations = 1000
prog = progress.Progress(max=iterations)

for i in range(iterations):
    #planewave = make_planewave(num)
    # onlense1.mode = modes['gaussian']
    #onlense_front = lense1.front.interact_with_all_wavelets(planewave)
    # print("onlense1: "+ str(onlense1.n))
    #onlense_back = lense1.back.interact_with_all_wavelets(onlense_front)
    # print("onlense2: "+ str(onlense2.n))
    onlense_back.mode = modes['gaussian']
    onscreen = screen.interact_with_all_wavelets(onlense_back)
    onscreen2 = screen2.interact_with_all_wavelets(onlense_back)

    # I = onlense2.calc_field(screen.points, 1.0,lense1.back.n2)
    # plt.plot(I**2)
    # plt.show()

    screen.add_field_from_wavelets(onscreen)
    screen2.add_field_from_wavelets(onscreen2)

    print(str(i) + " count on screen1: " + str(screen.count)+ " count on screen2: " + str(screen2.count))
    prog.next()
    print(str(np.round(prog.percent,1))+'%  ' + str(prog.eta_td))


plt.plot(screen.midpoints[:,1],screen.field ** 2)
plt.savefig("wavelet_lense5_screeny.png", dpi=600)
plt.show()

plt.plot(screen2.midpoints[:,0],screen2.hits)
plt.savefig("wavelet_lense_screenx.png", dpi=600)
plt.show()