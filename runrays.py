import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float64, int64, void, boolean
import cmath
from raytracing import *
import seaborn as sns

sns.set_style("ticks")


@jit()
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


@jit()
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    v = np.dot(v1_u, v2_u)
    if v > 1.0:
        v = 1.0
    elif v < -1.0:
        v = -1.0
    # return np.arccos(v)
    return cmath.acos(v)


@jit()
def rotate_vector(vector, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return np.dot(R, vector)


@jit()
def gen_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R


@jit()
def gen_concave_points(p0, r0, height, num):
    v = np.subtract(p0, r0)
    # w = np.array(np.dot(gen_rotation_matrix(np.pi/2),v))
    # w = w.reshape(v.shape)
    # w = unit_vector(w)
    # w = np.add(w*height/2,p0)
    # phi = angle_between(v,w).real
    # thetas = np.linspace(phi,-phi,num)
    thetas = np.linspace(height / 2, -height / 2, num)
    points = np.zeros((num, 2))
    for i, theta in enumerate(thetas):
        R = gen_rotation_matrix(theta)
        buf = np.dot(R, v)
        points[i] = buf + p0 + r0
    return points


# # @jit()
# def calc_I(xs, ys, wavelets, t):
#     x2, y2 = np.meshgrid(xs, ys)
#     I = np.zeros(x2.shape)
#     dy = np.diff(ys)[0] / 2
#
#     # t = 1.0
#     for i in range(xs.shape[0]):
#         for j in range(ys.shape[0]):
#             buf = 0.0
#             for n in range(len(wavelets)):
#                 # for n in [0,num-1]:
#                 buf += wavelets[n].calc_field(np.array([x[i], y[j]]), t) * wavelets[n].calc_probability(
#                     np.array([x[i], y[j] - dy]), np.array([x[i], y[j] + dy]))
#             I[j, i] += buf
#         print(i)
#
#     return I


num = 100
p0 = np.array([0, 0.0])
r0 = np.array([4.0, 0.0])
concave1 = gen_concave_points(p0, r0, np.pi/3, num)
p0 = np.array([4.0, 0.0])
r0 = np.array([0.0, 0.0])
concave2 = gen_concave_points(p0, r0, np.pi/3, num)



# plt.plot(concave[:,0],concave[:,1])
# plt.show()

mirror1 = Surface(concave1, reflectivity=1.0, transmittance=0.0, n1=1.0, n2=1.0)
mirror2 = Surface(concave2, reflectivity=1.0, transmittance=0.0, n1=1.0, n2=1.0)


rays = []
for k in mirror2.midpoints:
    for pos in mirror1.midpoints:
        rays.append(
            Ray(p0=pos, k=np.subtract(k, pos), t0=0.0, wavelength=0.1, phase=0.0))


print(len(rays))


#plt.plot(mirror2.points[:,0],mirror2.points[:,1],"rd")
#plt.plot(mirror1.points[:,0],mirror1.points[:,1],"bd")

new_rays = []
for ray in rays:
    #plt.plot(ray.p[0], ray.p[1], "r.")
    #plt.arrow(ray.p[0],ray.p[1],ray.k[0],ray.k[1])

    position,k,t,hit = mirror2.interact(ray)
    if hit:
        #ray.path += np.sqrt(np.sum(np.square(np.subtract(position,ray.p))))
        ray.path += ray.calc_path(position)
        ray.p = position
        ray.k = k
        ray.t0 = t
        ray.phase += np.pi
        #print(position)
        #plt.plot(position[0], position[1], "bo")
        #plt.arrow(position[0], position[1], k[0], k[1])
        new_rays.append(ray)

#plt.show()

rays = new_rays
print(len(rays))

#plt.plot(mirror2.points[:,0],mirror2.points[:,1],"rd")
#plt.plot(mirror1.points[:,0],mirror1.points[:,1],"bd")

new_rays = []
for ray in rays:
    #plt.plot(ray.p[0], ray.p[1], "r.")
    #plt.arrow(ray.p[0], ray.p[1], ray.k[0], ray.k[1])

    position, k, t, hit = mirror1.interact(ray)
    if hit:
        #ray.path += np.sqrt(np.sum(np.square(np.subtract(position,ray.p))))
        #print(position)
        #print(ray.p)
        #print(ray.calc_path(position))
        ray.path += ray.calc_path(position)
        ray.p = position
        ray.k = k
        ray.t0 = t
        # print(position)
        #plt.plot(position[0], position[1], "bo")
        #plt.arrow(position[0], position[1], k[0], k[1])
        new_rays.append(ray)

#plt.show()

rays = new_rays
print(len(rays))

new_rays = []
for ray in rays:
    if ray.is_back_at_origin():
        new_rays.append(ray)

rays = new_rays
print(len(rays))

plt.plot(mirror2.points[:,0],mirror2.points[:,1],"rd")
plt.plot(mirror1.points[:,0],mirror1.points[:,1],"bd")
for ray in rays:
    plt.plot(ray.p[0], ray.p[1], "r.")
    plt.arrow(ray.p[0], ray.p[1], ray.k[0], ray.k[1])


plt.axes().set_aspect('equal', 'datalim')
plt.show()



paths = np.zeros(len(rays))
for i,ray in enumerate(rays):
    paths[i] = ray.path

print(np.sum(np.square(np.diff(paths))))
print(np.mean(paths))

plt.plot(paths)
plt.show()

L = np.mean(paths)



# wavelengths = np.linspace(0.1,1.0,10000)
#
# fields = []
# for i,wl in enumerate(wavelengths):
#     field = 0.0
#     for ray in rays:
#         ray.set_wavelength(wl)
#         field += ray.calc_field(0.0)
#     fields.append(field)
#
# plt.plot(wavelengths,fields)
# plt.show()

