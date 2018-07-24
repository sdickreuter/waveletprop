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

# num = 100
# #lense1 = Lense(x=1.0, y=0,r1=np.inf,r2=4.0,height=2.0, num=num)
# lense1 = Lense(x=1.0, y=0,r1=-4.0,r2=np.inf,height=2.0, num=num)
# divider = 3
# plt.plot(lense1.front.points[:, 0], lense1.front.points[:, 1])
# for i in range(lense1.front.normals.shape[0]):
#     plt.plot(lense1.front.midpoints[i, 0], lense1.front.midpoints[i, 1], "bo")
#     plt.arrow(lense1.front.midpoints[i, 0], lense1.front.midpoints[i, 1], lense1.front.normals[i, 0] / divider,
#               lense1.front.normals[i, 1] / divider)
# plt.plot(lense1.back.points[:, 0], lense1.back.points[:, 1])
# for i in range(lense1.back.normals.shape[0]):
#     plt.plot(lense1.back.midpoints[i, 0], lense1.back.midpoints[i, 1], "bo")
#     plt.arrow(lense1.back.midpoints[i, 0], lense1.back.midpoints[i, 1], lense1.back.normals[i, 0] / divider,
#               lense1.back.normals[i, 1] / divider)
# plt.show()


@jit
def make_pointsource(num):
    rs = np.zeros((num, 2))
    ks = np.zeros((num, 2))
    ks[:, 0] = np.repeat(1.0, num)
    alphas = np.linspace(-np.pi / 3, np.pi / 3, num)
    for i in range(ks.shape[0]):
        ks[i, :] = rotate_vector(ks[i, :], alphas[i])
    t0s = np.zeros((num))
    phases = np.zeros((num))
    return Wavelets(r=rs, k=ks, t0=t0s, wavelength=0.03, phases=phases, mode=modes['ray'])

@jit
def weightedChoice(weights):
    """
    Return a random item from objects, with the weighting defined by weights
    (which must sum to 1).
    From: http://stackoverflow.com/a/10803136
    """
    cs = np.cumsum(weights)  # An array of the weights, cumulatively summed.
    idx = np.sum(cs < np.random.rand())  # Find the index of the first weight over a random value.
    return idx


@jit
def make_dipole(theta,alpha_max,num):
    rs = np.zeros((num, 2))
    ks = np.zeros((num, 2))
    ks[:, 0] = np.repeat(1.0, num)
    alphas = np.linspace(-alpha_max, alpha_max, num)
    probabilities = (np.cos(theta - alphas) ** 2)
    probabilities /= np.sum(probabilities)
    b = np.zeros(num)

    for i in range(ks.shape[0]):
        index = weightedChoice(probabilities)
        b[i] = alphas[index]
        ks[i, :] = rotate_vector(ks[i, :], alphas[index])

    phases = np.zeros((num))
    for i in range(len(alphas)):
        if theta-b[i]-np.pi/2 < 0:
            phases[i] = np.pi

    t0s = np.zeros((num))
    return Wavelets(r=rs, k=ks, t0=t0s, wavelength=0.1, phases=phases, mode=modes['ray'])

plotit = True

num = 500

lense1 = Lense(x=0.0, y=0,r1=np.inf,r2=2.0,height=0.5, num=num)
lense2 = Lense(x=0.0, y=0,r1=-2.0,r2=np.inf,height=0.5, num=num)

#lense1.shift(dx=lense1._calc_f_front()+lense1.front.points[:,0].min())
lense1.shift(dx=lense1._calc_f_front()+lense1.front.points[:,0].min())
lense2.shift(dx=lense1.back.points[:,0].max()+lense1._calc_f_back()+lense2._calc_f_front())

#lense1.shift(dx=lense1.f)
#lense2.shift(dx=lense1.back.points[:,0].max()+lense1._calc_f_back()*2)


theta = 0#np.pi/2
alpha_max = angle_between(np.array([1,0]),lense1.front.points[0,:])#np.pi/10
print("alpha_max: "+str(alpha_max)+'  '+str(alpha_max*180/np.pi))
# num = 10000
# ks = np.zeros((num, 2))
# ks[:, 0] = np.repeat(1.0, num)
# alphas = np.linspace(-np.pi/2 , np.pi/2, num)
# probabilities = np.zeros(num)
# probabilities = (np.cos(theta-alphas)**2)
# probabilities /= np.sum(probabilities)
# # plt.plot(theta-alphas,probabilities)
# # plt.show()
# b = np.zeros(num)
# for i in range(b.shape[0]):
#     index = weightedChoice(probabilities)
#     ks[i, :] = rotate_vector(ks[i, :], alphas[index])
#     b[i] = alphas[index]
# plt.hist(b,bins=100)
# plt.show()
#
# phases = np.zeros((num))
# for i in range(len(alphas)):
#     if theta-b[i]-np.pi/2 < 0:
#         phases[i] = np.pi
#
# #plt.plot(theta-alphas-np.pi/2)
# #plt.show()
#
# plt.plot(b,phases,'.b')
# plt.title('phases')
# plt.show()
#
# angles = np.zeros(ks.shape[0])
# for i in range(len(angles)):
#     angles[i] = angle_between(ks[i,:],np.array([1,0])).real
#
# plt.hist(angles,bins=100)
# plt.show()


num = 500
ys = np.linspace(-0.3, 0.3, num)
#xs = np.repeat(lense2.x+lense2._calc_f_back(), num)
#xs = np.repeat(lense2.back.points[:,0].max()+lense2._calc_f_back(), num)
#print('screen x: '+ str(lense2.back.points[:,0].max()+lense2._calc_f_back()))
xs = np.repeat(15.92, num)
screen = Surface(np.vstack((xs, ys)).T, reflectivity=0.0, transmittance=1.0, n1=1.0, n2=1.0)
screen.flip_normals()

# divider = 3
# plt.plot(lense1.front.points[:, 0], lense1.front.points[:, 1])
# for i in range(lense1.front.normals.shape[0]):
#     plt.plot(lense1.front.midpoints[i, 0], lense1.front.midpoints[i, 1], "bo")
#     plt.arrow(lense1.front.midpoints[i, 0], lense1.front.midpoints[i, 1], lense1.front.normals[i, 0] / divider,
#               lense1.front.normals[i, 1] / divider)
# plt.plot(lense1.back.points[:, 0], lense1.back.points[:, 1])
# for i in range(lense1.back.normals.shape[0]):
#     plt.plot(lense1.back.midpoints[i, 0], lense1.back.midpoints[i, 1], "bo")
#     plt.arrow(lense1.back.midpoints[i, 0], lense1.back.midpoints[i, 1], lense1.back.normals[i, 0] / divider,
#               lense1.back.normals[i, 1] / divider)
# plt.plot(lense2.front.points[:, 0], lense2.front.points[:, 1])
# for i in range(lense2.front.normals.shape[0]):
#     plt.plot(lense2.front.midpoints[i, 0], lense2.front.midpoints[i, 1], "bo")
#     plt.arrow(lense2.front.midpoints[i, 0], lense2.front.midpoints[i, 1], lense2.front.normals[i, 0] / divider,
#               lense2.front.normals[i, 1] / divider)
# plt.plot(lense2.back.points[:, 0], lense2.back.points[:, 1])
# for i in range(lense2.back.normals.shape[0]):
#     plt.plot(lense2.back.midpoints[i, 0], lense2.back.midpoints[i, 1], "bo")
#     plt.arrow(lense2.back.midpoints[i, 0], lense2.back.midpoints[i, 1], lense2.back.normals[i, 0] / divider,
#               lense2.back.normals[i, 1] / divider)
# plt.plot(screen.points[:, 0], screen.points[:, 1])
# plt.show()

num = 1000

dipole = make_dipole(theta, alpha_max,num)
#dipole = make_pointsource(num)
print("dipole: " + str(dipole.n))

onlense1_front = lense1.front.interact_with_all_wavelets(dipole)
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
    plt.plot(onlense1_back.t0)
    plt.show()
    plt.plot(onlense2_back.t0)
    plt.show()
    plt.plot(onscreen.t0)
    plt.show()
    # divider = 3
    # for i in range(lense1.front.normals.shape[0]):
    #     plt.plot(lense1.front.midpoints[i, 0], lense1.front.midpoints[i, 1], "b.")
    #     plt.arrow(lense1.front.midpoints[i, 0], lense1.front.midpoints[i, 1], lense1.front.normals[i, 0] / divider,
    #               lense1.front.normals[i, 1] / divider,color='blue')
    # plt.plot(lense1.back.points[:, 0], lense1.back.points[:, 1])
    # for i in range(lense1.back.normals.shape[0]):
    #     plt.plot(lense1.back.midpoints[i, 0], lense1.back.midpoints[i, 1], "b.")
    #     plt.arrow(lense1.back.midpoints[i, 0], lense1.back.midpoints[i, 1], lense1.back.normals[i, 0] / divider,
    #               lense1.back.normals[i, 1] / divider,color='blue')
    # plt.plot(lense2.front.points[:, 0], lense2.front.points[:, 1])
    # for i in range(lense2.front.normals.shape[0]):
    #     plt.plot(lense2.front.midpoints[i, 0], lense2.front.midpoints[i, 1], "b.")
    #     plt.arrow(lense2.front.midpoints[i, 0], lense2.front.midpoints[i, 1], lense2.front.normals[i, 0] / divider,
    #               lense2.front.normals[i, 1] / divider,color='blue')
    # plt.plot(lense2.back.points[:, 0], lense2.back.points[:, 1])
    # for i in range(lense2.back.normals.shape[0]):
    #     plt.plot(lense2.back.midpoints[i, 0], lense2.back.midpoints[i, 1], "b.")
    #     plt.arrow(lense2.back.midpoints[i, 0], lense2.back.midpoints[i, 1], lense2.back.normals[i, 0] / divider,
    #               lense2.back.normals[i, 1] / divider,color='blue')

    divider = 10#200
    plt.plot(dipole.r[:, 0], dipole.r[:, 1])
    for i in range(dipole.r.shape[0]):
        plt.plot(dipole.r[i, 0], dipole.r[i, 1], "bo")
        plt.arrow(dipole.r[i, 0], dipole.r[i, 1], dipole.k[i, 0] / (divider*10), dipole.k[i, 1] / (divider*10))
    plt.plot(lense1.front.points[:, 0], lense1.front.points[:, 1])
    for i in range(onlense1_front.r.shape[0]):
        plt.plot(onlense1_front.r[i, 0], onlense1_front.r[i, 1], "bo")
        plt.arrow(onlense1_front.r[i, 0], onlense1_front.r[i, 1], onlense1_front.k[i, 0] / (divider*10), onlense1_front.k[i, 1] / (divider*10))
    plt.plot(lense1.back.points[:, 0], lense1.back.points[:, 1])
    for i in range(onlense1_back.r.shape[0]):
        plt.plot(onlense1_back.r[i, 0], onlense1_back.r[i, 1], "bo")
        plt.arrow(onlense1_back.r[i, 0], onlense1_back.r[i, 1], onlense1_back.k[i, 0] / divider, onlense1_back.k[i, 1] / divider)
    plt.plot(lense2.front.points[:, 0], lense2.front.points[:, 1])
    for i in range(onlense2_front.r.shape[0]):
        plt.plot(onlense2_front.r[i, 0], onlense2_front.r[i, 1], "bo")
        plt.arrow(onlense2_front.r[i, 0], onlense2_front.r[i, 1], onlense2_front.k[i, 0] / (divider*10),
                  onlense2_front.k[i, 1] / (divider*10))
    plt.plot(lense2.back.points[:, 0], lense2.back.points[:, 1])
    for i in range(onlense2_back.r.shape[0]):
        plt.plot(onlense2_back.r[i, 0], onlense2_back.r[i, 1], "bo")
        plt.arrow(onlense2_back.r[i, 0], onlense2_back.r[i, 1], onlense2_back.k[i, 0] / divider,
                  onlense2_back.k[i, 1] / divider)
    plt.plot(screen.points[:, 0], screen.points[:, 1])
    # for i in range(onscreen.r.shape[0]):
    #     plt.plot(onscreen.r[i, 0], onscreen.r[i, 1], "bo")
    #     plt.arrow(onscreen.r[i, 0], onscreen.r[i, 1], onscreen.k[i, 0] / divider, onscreen.k[i, 1] / divider)
    plt.show()



# x = np.linspace(lense1.back.points[:,0].max()+0.01, lense2.front.points[:,0].min()-0.01, 200)
# y = np.linspace(-1.0, 1.0, 100)
# x2, y2 = np.meshgrid(x,y)
# points = np.vstack((x2.ravel(),y2.ravel())).T
# I = onlense1_back.calc_field(points, 1.0,lense1.back.n2)
# I = np.reshape(I,x2.shape)
# plt.imshow(I,extent=[x.min(),x.max(),y.min(),y.max()])
# plt.show()
#
# x = np.linspace(lense2.back.points[:,0].max()+0.01, 7, 200)
# y = np.linspace(-1.0, 1.0, 100)
# x2, y2 = np.meshgrid(x,y)
# points = np.vstack((x2.ravel(),y2.ravel())).T
# I = onlense2_back.calc_field(points, 1.0,lense2.back.n2)
# I = np.reshape(I,x2.shape)
# plt.imshow(I,extent=[x.min(),x.max(),y.min(),y.max()])
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
num=1000
for i in range(iterations):

    dipole = make_dipole(theta, alpha_max, num)
    onlense1_front = lense1.front.interact_with_all_wavelets(dipole)

    # for j in range(int(iterations/100)):
    #     dipole = make_dipole(theta, alpha_max, num)
    #     onlense1_front.append_wavelets(lense1.front.interact_with_all_wavelets(dipole))

    onlense1_back = lense1.back.interact_with_all_wavelets(onlense1_front)
    onlense2_front = lense2.front.interact_with_all_wavelets(onlense1_back)
    onlense2_back = lense2.back.interact_with_all_wavelets(onlense2_front)

    onlense2_back = lense2.back.interact_with_all_wavelets(onlense2_front)
    onlense2_back.mode = modes['gaussian']
    onscreen = screen.interact_with_all_wavelets(onlense2_back)

    screen.add_field_from_wavelets(onscreen)

    print(str(i) + " count on screen1: " + str(screen.count))
    prog.next()
    print(str(np.round(prog.percent,1))+'%  ' + str(prog.eta_td))


plt.plot(screen.midpoints[:,1],screen.field[:,1] ** 2)
plt.savefig("dipole_horz.png", dpi=600)
plt.show()

# plt.plot(screen2.midpoints[:,0],screen2.hits)
# plt.savefig("wavelet_lense_screenx.png", dpi=600)
# plt.show()