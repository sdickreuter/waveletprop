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
    ks = np.zeros((num, 2))
    ks[:, 0] = np.repeat(1.0, num)
    alphas = np.linspace(-np.pi / 3, np.pi / 3, num)
    for i in range(ks.shape[0]):
        ks[i, :] = rotate_vector(ks[i, :], alphas[i])
    t0s = np.zeros((num))
    phases = np.zeros((num))
    return Wavelets(r=rs, k=ks, t0=t0s, wavelength=0.1, phases=phases, mode=modes['ray'])


plotit = False

num = 301

lense1 = Lense(x=0.0, y=0, height=2.0, num=num)
lense2 = Lense(x=0.0, y=0, height=2.0, num=num)

lense1.shift(dx=lense1.f)
lense2.shift(dx=lense1.x+lense2.f+lense2.d)


num = 301
ys = np.linspace(-0.3, 0.3, num)
xs = np.repeat(lense2.x+lense2.f, num)
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
screen.flip_normals()


plt.plot(lense1.front.points[:, 0], lense1.front.points[:, 1])
plt.plot(lense1.back.points[:, 0], lense1.back.points[:, 1])
plt.plot(lense2.front.points[:, 0], lense2.front.points[:, 1])
plt.plot(lense2.back.points[:, 0], lense2.back.points[:, 1])
plt.plot(screen.points[:, 0], screen.points[:, 1])
plt.show()

num = 301

pointsource = make_planewave(num)
print("pointsource: " + str(pointsource.n))

onlense1_front = lense1.front.interact_with_all_wavelets(pointsource)
print("onlense1 front: " + str(onlense1_front.n))
onlense1_back = lense1.back.interact_with_all_wavelets(onlense1_front)
print("onlense1 back: " + str(onlense1_back.n))

onlense2_front = lense2.front.interact_with_all_wavelets(onlense1_back)
print("onlense2 front: " + str(onlense2_front.n))
onlense2_back = lense2.back.interact_with_all_wavelets(onlense2_front)
print("onlense2 back: " + str(onlense2_back.n))


onlense2_back.mode = modes['gaussian']
onscreen = screen.interact_with_all_wavelets(onlense2_back)

print("onscreen: " + str(onscreen.n))
screen.add_field_from_wavelets(onscreen)

if plotit:
    # plt.plot(onlense1_back.t0)
    # plt.show()
    #
    # plt.plot(onlense2_back.t0)
    # plt.show()
    #
    # plt.plot(onscreen.t0)
    # plt.show()

    divider = 20#300
    #plt.plot(lense1.front.points[:, 0], pointsource.r[:, 1])
    # for i in range(pointsource.n):
    #     plt.plot(pointsource.r[i, 0], pointsource.r[i, 1], "bo")
    #     plt.arrow(pointsource.r[i, 0], pointsource.r[i, 1], pointsource.k[i, 0] / divider, pointsource.k[i, 1] / divider)
    # plt.plot(lense1.front.points[:, 0], lense1.front.points[:, 1])
    # for i in range(onlense1_front.n):
    #     plt.plot(onlense1_front.r[i, 0], onlense1_front.r[i, 1], "bo")
    #     plt.arrow(onlense1_front.r[i, 0], onlense1_front.r[i, 1], onlense1_front.k[i, 0] / divider, onlense1_front.k[i, 1] / divider)
    # plt.plot(lense1.back.points[:, 0], lense1.back.points[:, 1])
    # for i in range(onlense1_back.n):
    #     plt.plot(onlense1_back.r[i, 0], onlense1_back.r[i, 1], "bo")
    #     plt.arrow(onlense1_back.r[i, 0], onlense1_back.r[i, 1], onlense1_back.k[i, 0] / divider, onlense1_back.k[i, 1] / divider)
    # plt.plot(lense2.front.points[:, 0], lense2.front.points[:, 1])
    # for i in range(onlense2_front.n):
    #     plt.plot(onlense2_front.r[i, 0], onlense2_front.r[i, 1], "bo")
    #     plt.arrow(onlense2_front.r[i, 0], onlense2_front.r[i, 1], onlense2_front.k[i, 0] / divider,
    #               onlense2_front.k[i, 1] / divider)
    plt.plot(lense2.back.points[:, 0], lense2.back.points[:, 1])
    for i in range(onlense2_back.n):
        plt.plot(onlense2_back.r[i, 0], onlense2_back.r[i, 1], "bo")
        plt.arrow(onlense2_back.r[i, 0], onlense2_back.r[i, 1], onlense2_back.k[i, 0] / divider,
                  onlense2_back.k[i, 1] / divider)
    plt.plot(screen.points[:, 0], screen.points[:, 1])
    for i in range(onscreen.n):
        plt.plot(onscreen.r[i, 0], onscreen.r[i, 1], "bo")
        plt.arrow(onscreen.r[i, 0], onscreen.r[i, 1], onscreen.k[i, 0] / divider, onscreen.k[i, 1] / divider)
    #plt.plot(onlense1_back.points[:, 0], onlense1_back.points[:, 1])
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
    onlense2_back = lense2.back.interact_with_all_wavelets(onlense2_front)
    onlense2_back.mode = modes['gaussian']
    onscreen = screen.interact_with_all_wavelets(onlense2_back)

    screen.add_field_from_wavelets(onscreen)

    print(str(i) + " count on screen1: " + str(screen.count))
    prog.next()
    print(str(np.round(prog.percent,1))+'%  ' + str(prog.eta_td))


plt.plot(screen.midpoints[:,1],screen.field ** 2)
plt.savefig("wavelet_lense5_screeny.png", dpi=600)
plt.show()

# plt.plot(screen2.midpoints[:,0],screen2.hits)
# plt.savefig("wavelet_lense_screenx.png", dpi=600)
# plt.show()