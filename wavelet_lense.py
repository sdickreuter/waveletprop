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
p0 = np.array([0.0, 0.0])
concave1 = gen_concave_points(p0, -2.0, np.pi/3, num)
concave1[:,1]=concave1[:,1]/concave1[:,1].max()
concave1[:,1]=concave1[:,1]*1.2
concave1[:,0]-=concave1[:,0].min()
concave1[:,0]+=1.0

p0 = np.array([0.0, 0.0])
concave2 = gen_concave_points(p0, 2.0, np.pi/3, num)
concave2[:,1]=concave2[:,1]/concave2[:,1].max()
concave2[:,1]=concave2[:,1]*1.2
concave2[:,0]-=concave2[:,0].min()
concave2[:,0]+=concave1[:,0].max()

lense1_front = Surface(concave1, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.5)
lense1_back = Surface(concave2, reflectivity=0.0, transmittance=1.0, n1=1.5, n2=1.0)
# ys = np.linspace(concave1[:,1].min(),concave1[:,1].max(),num)
# xs = np.repeat(concave1[:,0].max(),num)
# lense1_back = Surface(np.vstack((xs,ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.5, n2=1.0)

# plt.plot(lense1_front.points[:,0],lense1_front.points[:,1])
# plt.plot(lense1_back.points[:,0],lense1_back.points[:,1])
# plt.show()

num = 1024

rs = np.zeros((num,2))
rs[:,0] = np.repeat(0.0,num)
rs[:,1] = np.linspace(-1, 1, num)


ks = np.zeros((num,2))
ks[:,0] = np.repeat(1.0,num)

t0s = np.zeros((num))
phases = np.zeros((num))

planewave = Wavelets(r=rs, k=ks, t0=t0s, wavelength=0.1, phases=phases, mode=modes['ray'])

print("pointsource: "+ str(planewave.n))

x = np.linspace(0.1, 3, 400)
y = np.linspace(-1.5, 1.5, 200)
x2, y2 = np.meshgrid(x,y)
points = np.vstack((x2.ravel(),y2.ravel())).T

# I_plane = pointsource.calc_field(points, 1.0)
# I_plane = np.reshape(I_plane,x2.shape)
#
# plt.plot(concave1[:,0],concave1[:,1])
# plt.plot(concave2[:,0],concave2[:,1])
# plt.imshow(I_plane, extent=(x.min(), x.max(), y.max(), y.min()), cmap='RdBu')
# plt.savefig("plane_field_ref.png", dpi=300)
# plt.show()

onlense1 = lense1_front.interact_with_all_wavelets(planewave)

print("onlense1: "+ str(onlense1.n))


# plt.plot(concave1[:,0],concave1[:,1])
# plt.plot(concave2[:,0],concave2[:,1])
# for i in range(pointsource.n):
#     plt.plot(pointsource.r[i,0], pointsource.r[i,1], "bo")
#     plt.arrow(pointsource.r[i,0], pointsource.r[i,1], pointsource.k[i,0], pointsource.k[i,1])
# for i in range(onlense1.n):
#     plt.plot(onlense1.r[i,0], onlense1.r[i,1], "rx")
#     plt.arrow(onlense1.r[i,0], onlense1.r[i,1], onlense1.k[i,0], onlense1.k[i,1])
# plt.show()


I_plane = onlense1.calc_field(points, 1.0,lense1_front.n2)
I_plane = np.reshape(I_plane,x2.shape)

plt.plot(lense1_front.points[:,0],lense1_front.points[:,1])
plt.plot(lense1_back.points[:,0],lense1_back.points[:,1])
plt.plot(onlense1.r[:,0],onlense1.r[:,1],'rx')
plt.imshow(I_plane, extent=(x.min(), x.max(), y.max(), y.min()), cmap='RdBu')
plt.savefig("plane_field_ref.png", dpi=300)
plt.show()

plt.plot(lense1_front.points[:,0],lense1_front.points[:,1])
plt.plot(lense1_back.points[:,0],lense1_back.points[:,1])
plt.imshow(I_plane ** 2, extent=(x.min(), x.max(), y.max(), y.min()), cmap='seismic')
# plt.imshow(I**2,extent=(x.min(), x.max(), y.max(), y.min()))
plt.savefig("plane__int_ref.png", dpi=300)
plt.show()

onlense2 = lense1_back.interact_with_all_wavelets(onlense1)
print(onlense2.n)

x = np.linspace(1.6, 5, 400)
y = np.linspace(-1.5, 1.5, 200)
x2, y2 = np.meshgrid(x,y)
points = np.vstack((x2.ravel(),y2.ravel())).T

I_ref = onlense2.calc_field(points, 1.0,lense1_back.n2)
I_ref = np.reshape(I_ref,x2.shape)

plt.plot(lense1_front.points[:,0],lense1_front.points[:,1])
plt.plot(lense1_back.points[:,0],lense1_back.points[:,1])
plt.imshow(I_ref, extent=(x.min(), x.max(), y.max(), y.min()), cmap='RdBu')
plt.savefig("field_ref.png", dpi=300)
plt.show()

plt.plot(lense1_front.points[:,0],lense1_front.points[:,1])
plt.plot(lense1_back.points[:,0],lense1_back.points[:,1])
plt.imshow(I_ref ** 2, extent=(x.min(), x.max(), y.max(), y.min()),cmap='seismic')
# plt.imshow(I**2,extent=(x.min(), x.max(), y.max(), y.min()))
plt.savefig("intensity_ref.png", dpi=300)
plt.show()



# for wavelet in reflected:
#     plt.plot(wavelet.r[0],wavelet.r[1],"o")
#     plt.arrow(wavelet.r[0],wavelet.r[1],wavelet.k[0],wavelet.k[1])
# plt.savefig("mirror_ref.png", dpi=300)
# plt.show()

# zd, xe, ye = np.histogram2d(yl, xl, bins=10, range=[[view_ymin, view_ymax], [view_xmin, view_xmax]], normed=True)


# num = 100
# p0 = np.array([-0.5, 0.0])
# r0 = np.array([4.0, 0.0])
# concave = gen_concave_points(p0, r0, 1.0, num)
# waves = []
#
#
# for i in range(num):
#     k = np.subtract(r0, concave[i, :])
#     waves.append(Wavelet(r=concave[i, :], k=k, t0=0.0, wavelength=0.1, phase=0.0, spherical=False))
#
# x = np.linspace(1.0, 6.0, 500)
# y = np.linspace(-0.9, 0.9, 200)
# x2, y2 = np.meshgrid(x, y)
# I = np.zeros(x2.shape)
# dy = np.diff(y)[0] / 2
#
# t = 0.0
# for i in range(x.shape[0]):
#     for j in range(y.shape[0]):
#         buf = 0.0
#         for n in range(len(waves)):
#             # for n in [0,num-1]:
#             buf += waves[n].calc_field(np.array([x[i], y[j]]), t) * waves[n].calc_probability(
#                 np.array([x[i], y[j] - dy]), np.array([x[i], y[j] + dy]))
#         I[j, i] += buf
#     print(i)
#
# plt.imshow(I, extent=(x.min(), x.max(), y.max(), y.min()), cmap=sns.diverging_palette(240, 10, as_cmap=True))
# plt.savefig("field.png", dpi=300)
# plt.show()
#
# plt.imshow(I ** 2, extent=(x.min(), x.max(), y.max(), y.min()),
#            cmap=sns.cubehelix_palette(start=0, rot=0.4, gamma=1.0, hue=0.8, light=1.0, dark=0.1, as_cmap=True))
# # plt.imshow(I**2,extent=(x.min(), x.max(), y.max(), y.min()))
# plt.savefig("intensity.png", dpi=300)
# plt.show()





# d3 = Wavelet(r=np.array([0.0, 0.0]), k=np.array([1.0, 0.0]), t0=0.0, wavelength=0.1, phase=0.0, spherical=False)
# eps = 0.1
# print(d3.calc_probability(np.array([1.0,0.0+eps]),np.array([1.0,0.0+2*eps])))
# print(d3.calc_probability(np.array([1.0,0.0-eps]),np.array([1.0,0.0+eps])))
# print(d3.calc_probability(np.array([1.0,0.0-eps]),np.array([1.0,0.0-2*eps])))

# d1 = Wavelet(r=np.array([0.0, 0.1]), k=np.array([1.0, 0.0]), t0=0.0, wavelength=0.05, phase=0.0, spherical=False)
# d2 = Wavelet(r=np.array([0.0, -0.1]), k=np.array([1.0, 0.0]), t0=0.0, wavelength=0.05, phase=0.0, spherical=False)
#
# x = np.linspace(0.1,1,400)
# y = np.linspace(-0.7,0.7,400)
# x2,y2 = np.meshgrid(x,y)
# I = np.zeros(x2.shape)
# dy = np.diff(y)[0]/2
#
# t = 0.0
# for i in range(x.shape[0]):
#     for j in range(y.shape[0]):
#         buf  = d1.calc_field(np.array([x[i],y[j]]),t)*d1.calc_probability(np.array([x[i],y[j]-dy]),np.array([x[i],y[j]+dy]))
#         buf += d2.calc_field(np.array([x[i],y[j]]),t)*d2.calc_probability(np.array([x[i],y[j]-dy]),np.array([x[i],y[j]+dy]))
#         I[i,j] += buf**2
#
# plt.imshow(I.transpose())
# plt.show()
#
# ts = np.linspace(0,np.pi/2,300)
# I = np.zeros(y.shape)
# for t in ts:
#     for j in range(y.shape[0]):
#         buf  = d1.calc_field(np.array([1.0,y[j]]),t)*d1.calc_probability(np.array([1.0,y[j]-dy]),np.array([1.0,y[j]+dy]))
#         buf += d2.calc_field(np.array([1.0,y[j]]),t)*d2.calc_probability(np.array([1.0,y[j]-dy]),np.array([1.0,y[j]+dy]))
#         I[j] += buf**2
#
# plt.plot(I)
# plt.show()

#
# #d1 = DipoleWavelet([0, 0.01], [1, 0], 0, wavelength=0.7e-6)
# #d2 = DipoleWavelet([0, -0.01], [1, 0], 0, wavelength=0.7e-6)
# d1 = Wavelet(r=np.array([0, 0.1]), k=np.array([1.0, 0.0]), t0=0.0, wavelength=0.05, phase=0.0, spherical=True)
# d2 = Wavelet(r=np.array([0, -0.1]), k=np.array([1.0, 0.0]), t0=0.0, wavelength=0.05, phase=0.0, spherical=True)
#
# #print(d1.calc_probability([1, 1], [1, -1]))
#
# n=1000
# height=4.0
#
# s = Surface(np.array([1.0,-height/2]),height,n)
#
# prob1, fields1 = s.interact(d1)
# prob2, fields2 = s.interact(d2)
#
#
# y = np.linspace(-height / 2, height / 2, len(prob1))
# plt.plot(prob1, y)
# plt.plot(prob2, y)
# plt.show()
#
# plt.plot(fields1, y)
# plt.plot(fields2, y)
# plt.show()
#
# #plt.plot(phase1-phase2, y)
# #plt.show()
#
# plt.plot(np.square(fields1 * prob1+fields2 *prob2), y)
# plt.show()
#
# plt.plot(np.square(fields1+fields2), y)
# plt.show()
#
# # iter = 1000
# # intensity = np.zeros(s.points.shape[0]-1)
# # logpoints = np.arange(0,iter,100)
# # for i in range(iter):
# #     phi = (np.random.rand()-0.5)*cmath.pi/2
# #     k = cmath.rect(1.0, phi)
# #     k = [k.real,k.imag]
# #
# #     d1 = Wavelet(r=np.array([0.0, 0.1]), k=np.array([1.0, 0.0]), t0=0, wavelength=0.7, phase=0, spherical=False)
# #     d2 = Wavelet(r=np.array([0.0, -0.1]), k=np.array([1.0, 0.0]), t0=0, wavelength=0.7, phase=0, spherical=False)
# #
# #     prob1, fields1 = s.interact(d1)
# #     prob2, fields2 = s.interact(d2)
# #     intensity += np.square(fields1 * prob1+fields2 *prob2)
# #
# #     if i in logpoints:
# #         print(i)
# #
# # plt.plot(intensity)
# # plt.show()
