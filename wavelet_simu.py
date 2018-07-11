import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float32, int32, void, boolean
import cmath
from wavelets2 import *




num = 256

lense1 = Lense(0.5,0,2.2)

num = 100
ys = np.linspace(-1,1,num)
xs = np.repeat(2.4,num)
screen = Surface(np.vstack((xs,ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)

plt.plot(lense1.front.points[:,0],lense1.front.points[:,1])
plt.plot(lense1.back.points[:,0],lense1.back.points[:,1])
plt.plot(screen.points[:,0],screen.points[:,1])
plt.show()

num = 10000

rs = np.zeros((num,2))
ks = np.zeros((num,2))
ks[:,1] = np.repeat(1.0,num)
alphas = np.linspace(0,-np.pi,num)
for i in range(ks.shape[0]):
    ks[i,:] = rotate_vector(ks[i,:],alphas[i])
t0s = np.zeros((num))
phases = np.zeros((num))
pointsource = Wavelets(r=rs, k=ks, t0=t0s, wavelength=0.1, phases=phases, mode=modes['ray'])
print("pointsource: "+ str(pointsource.n))

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
onlense1 = lense1.front.interact_with_all_wavelets(pointsource)
#onlense1.mode = modes['gaussian']


# plt.plot(lense1.front.points[:,0],lense1.front.points[:,1])
# plt.plot(lense1.back.points[:,0],lense1.back.points[:,1])
# for i in range(pointsource.n):
#     plt.plot(pointsource.r[i,0], pointsource.r[i,1], "bo")
#     plt.arrow(pointsource.r[i,0], pointsource.r[i,1], pointsource.k[i,0], pointsource.k[i,1])
# for i in range(onlense1.n):
#     plt.plot(onlense1.r[i,0], onlense1.r[i,1], "rx")
#     plt.arrow(onlense1.r[i,0], onlense1.r[i,1], onlense1.k[i,0], onlense1.k[i,1])
# plt.show()
#


print("onlense1: "+ str(onlense1.n))

onlense2 = lense1.back.interact_with_all_wavelets(onlense1)

print("onlense2: "+ str(onlense2.n))

onlense2.mode = modes['spherical']


# plt.plot(lense1.front.points[:,0],lense1.front.points[:,1])
# plt.plot(lense1.back.points[:,0],lense1.back.points[:,1])
# for i in range(onlense2.n):
#     plt.plot(onlense2.r[i,0], onlense2.r[i,1], "rx")
#     plt.arrow(onlense2.r[i,0], onlense2.r[i,1], onlense2.k[i,0], onlense2.k[i,1])
# plt.show()
#




x = np.linspace(0.7, 2, 200)
y = np.linspace(-1.0, 1.0, 100)
x2, y2 = np.meshgrid(x,y)
points = np.vstack((x2.ravel(),y2.ravel())).T

I_plane = onlense2.calc_field(points, 1.0,lense1.back.n2)
I_plane = np.reshape(I_plane,x2.shape)

plt.plot(lense1.front.points[:,0],lense1.front.points[:,1])
plt.plot(lense1.back.points[:,0],lense1.back.points[:,1])
plt.plot(onlense2.r[:,0],onlense2.r[:,1],'rx')
plt.imshow(I_plane, extent=(x.min(), x.max(), y.max(), y.min()), cmap='RdBu')
plt.savefig("plane_field_ref.png", dpi=300)
plt.show()



# plt.plot(lense1_front.points[:,0],lense1_front.points[:,1])
# plt.plot(lense1_back.points[:,0],lense1_back.points[:,1])
# plt.plot(positions[:,0],positions[:,1])
# plt.show()

I = onlense2.calc_field(screen.points, 1.0,lense1.back.n2)
plt.plot(I**2)
plt.show()

#positions, field = screen.intensity_on_surface(onlense2)
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