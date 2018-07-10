import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float64, int64, void, boolean
import cmath
from wavelets2 import *




num = 512

concave1, concave2 = generate_lens_points(num,2.2)


lense1_front = Surface(concave1, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.5)
lense1_back = Surface(concave2, reflectivity=0.0, transmittance=1.0, n1=1.5, n2=1.0)

num = 200
ys = np.linspace(0,1,num)
xs = np.repeat(2.5,num)
screen = Surface(np.vstack((xs,ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)

# plt.plot(lense1_front.points[:,0],lense1_front.points[:,1])
# plt.plot(lense1_back.points[:,0],lense1_back.points[:,1])
# plt.show()

num = 1000000

# rs = np.zeros((num,2))
# ks = np.zeros((num,2))
# ks[:,1] = np.repeat(-1.0,num)
# alphas = np.linspace(0,2*np.pi,num)
# for i in range(ks.shape[0]):
#     ks[i,:] = rotate_vector(ks[i,:],alphas[i])
# t0s = np.zeros((num))
# phases = np.zeros((num))
# pointsource = Wavelets(r=rs, k=ks, t0=t0s, wavelength=0.1, phases=phases, mode=modes['gaussian'])
# print("pointsource: "+ str(pointsource.n))

rs = np.zeros((num,2))
rs[:,0] = np.repeat(0.0,num)
rs[:,1] = np.linspace(-1, 1, num)
ks = np.zeros((num,2))
ks[:,0] = np.repeat(1.0,num)
t0s = np.zeros((num))
phases = np.zeros((num))
planewave = Wavelets(r=rs, k=ks, t0=t0s, wavelength=0.1, phases=phases, mode=modes['ray'])
print("planewave: "+ str(planewave.n))


# x = np.linspace(0.1, 3, 400)
# y = np.linspace(-1.5, 1.5, 200)
# x2, y2 = np.meshgrid(x,y)
# points = np.vstack((x2.ravel(),y2.ravel())).T
#
# I_plane = pointsource.calc_field(points, 1.0, 1.0)
# I_plane = np.reshape(I_plane,x2.shape)
#
# plt.plot(concave1[:,0],concave1[:,1])
# plt.plot(concave2[:,0],concave2[:,1])
# plt.imshow(I_plane, extent=(x.min(), x.max(), y.max(), y.min()), cmap='RdBu')
# plt.savefig("plane_field_ref.png", dpi=300)
# plt.show()

#onlense1 = lense1_front.interact_with_all_wavelets(pointsource)
onlense1 = lense1_front.interact_with_all_wavelets(planewave)
onlense1.mode = modes['spherical']

print("onlense1: "+ str(onlense1.n))

onlense2 = lense1_back.interact_with_all_wavelets(onlense1)

print("onlense2: "+ str(onlense2.n))

positions, field = screen.intensity_on_surface(onlense2)

plt.plot(lense1_front.points[:,0],lense1_front.points[:,1])
plt.plot(lense1_back.points[:,0],lense1_back.points[:,1])
plt.plot(positions[:,0],positions[:,1])
plt.show()

#print(positions)
#print(field)

yind = positions[:,1]
yind -= yind.min()
yind /= yind.max()
un = len(np.unique(yind))
yind *= un-1
yind = np.array(yind,dtype=np.int)

intensity = np.zeros(un)
hits = np.zeros(un)
for i in yind:
    intensity[i] += field[i]
    hits[i] += 1

plt.plot(intensity)
plt.show()

plt.plot(hits)
plt.show()