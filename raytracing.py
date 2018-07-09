import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float64, int64, void, boolean
import cmath

#c = 2.998e8  # m/s
c = 1.0  # m/s

spec_Ray = [
    ('p0', float64[:]),
    ('p', float64[:]),
    ('k0', float64[:]),
    ('k', float64[:]),
    ('t0', float64),
    ('t', float64),
    ('phase', float64),
    ('wavelength', float64),
    ('f', float64),
    ('interactions', int64),
    ('path', float64)
]


@jitclass(spec_Ray)
class Ray(object):
    def __init__(self, p0, k, t0, wavelength, phase):
        self.p0 = p0
        self.p = p0
        self.k0 = (k / np.linalg.norm(k)) * 2 * (np.pi / wavelength)
        self.k = self.k0
        self.t0 = t0
        self.phase = phase
        self.wavelength = wavelength
        self.f = c / wavelength
        self.interactions = 0
        self.path = 0.0

    def calc_path(self,point):
        return np.sqrt(np.sum(np.square(np.subtract(self.p, point))))

    def calc_r(self, t):
        r = c * (t - self.t0)
        if r > 0:
            return r
        else:
            return 0.0

    def calc_t(self, point):
        return np.sqrt(np.sum(np.square(np.subtract(self.p, point)))) / c

    # def calc_field(self, point, t):
    #     r = np.linalg.norm(self.p - point)
    #     field = cmath.exp(1j * (np.linalg.norm(self.k) * r - 2 * cmath.pi * self.f * (t - self.t0) + self.phase)).real
    #     return field

    def calc_field(self,t):
        field = cmath.exp(1j * (np.linalg.norm(self.k) * self.path - 2 * cmath.pi * self.f * (t - self.t0) + self.phase)).real
        return field

    def set_wavelength(self, wavelength):
        self.wavelength = wavelength
        self.f = c / wavelength

    def set_frequency(self, f):
        self.f = f
        self.wavelength = c / f

    def is_back_at_origin(self):
        p = self.p/np.linalg.norm(self.p)
        p0 = self.p0/np.linalg.norm(self.p0)
        k = self.k/np.linalg.norm(self.k)
        k0 = self.k0/np.linalg.norm(self.k0)

        if p[0]-p0[0] < 1e-5:
            if p[1] - p0[1] < 1e-5:
                if k[0] - k0[0] < 1e-5:
                    if k[1] - k0[1] < 1e-5:
                        return True
        return False
        #return np.isclose(self.p[0], self.p0[0], rel_tol=0.001) and np.isclose(self.p[1], self.p0[1], rel_tol=0.001) and np.isclose(self.k[0], self.k0[0], rel_tol=0.001) and np.isclose(self.k[1], self.k0[1], rel_tol=0.001)


spec_Surface = [
    ('points', float64[:, :]),
    ('midpoints', float64[:, :]),
    ('normals', float64[:, :]),
    ('reflectivity', float64),
    ('transmittance', float64),
    ('n1', float64),
    ('n2', float64),
]


#@jitclass(spec_Surface)
class Surface(object):
    def __init__(self, points, reflectivity, transmittance, n1=1.0, n2=1.0):
        self.points = points
        self.reflectivity = reflectivity
        self.transmittance = transmittance
        self.n1 = n1
        self.n2 = n2

        self.midpoints = self.calc_midpoints()
        self.normals = self.calc_normals()


    def cross(self, vec1, vec2):
        """ Calculate the cross product of two 3d vectors. """
        result = np.zeros(3)
        a1, a2, a3 = vec1[0], vec1[1], vec1[2]
        b1, b2, b3 = vec2[0], vec2[1], vec2[2]
        result[0] = a2 * b3 - a3 * b2
        result[1] = a3 * b1 - a1 * b3
        result[2] = a1 * b2 - a2 * b1

        return result

    def cross2d(self, vec1, vec2):
        """ Calculate the cross product of two 3d vectors. """
        result = np.zeros(3)
        a1, a2, a3 = vec1[0], vec1[1], 0.0
        b1, b2, b3 = vec2[0], vec2[1], 0.0
        #result[0] = a2 * b3 - a3 * b2
        #result[1] = a3 * b1 - a1 * b3
        result[2] = a1 * b2 - a2 * b1

        return result[2]


    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        # v1_u = self.unit_vector(v1)
        # v2_u = self.unit_vector(v2)
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        v = np.dot(v1_u, v2_u)
        if v > 1.0:
            v = 1.0
        elif v < -1.0:
            v = -1.0
        # return np.arccos(v)
        return cmath.acos(v)

    def reflected_k(self, vector, normal):
        reflected = vector - (2 * (np.dot(normal, vector)) * normal)
        return reflected

    def transmitted_k(self, vector, normal):
        alpha = self.angle_between(vector, normal)
        beta = cmath.sqrt(self.n2 ** 2 - self.n1 ** 2 * cmath.sin(alpha) ** 2).real / self.n2
        transmitted = self.rotate_vector(normal, np.pi - beta)
        return transmitted

    def rotate_vector(self, vector, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.zeros((2, 2))
        R[0, 0] = c
        R[1, 0] = -s
        R[0, 1] = s
        R[1, 1] = c
        return np.dot(R, vector)

    # def interact(self, wavelet):
    #     probabilities = np.zeros(self.points.shape[0] - 1, dtype=np.float64)
    #     fields = np.zeros(self.points.shape[0] - 1)
    #     for i in range(self.points.shape[0] - 1):
    #         probabilities[i] = wavelet.calc_probability(self.points[i], self.points[i + 1])
    #         middle_point = 0.5 * np.add(self.points[i], self.points[i + 1])
    #         fields[i] = wavelet.calc_field(middle_point)
    #
    #     return probabilities, fields

    def calc_midpoints(self):
        midpoints = np.zeros((self.points.shape[0] - 1, self.points.shape[1]))
        for i in range(self.points.shape[0] - 1):
            midpoints[i,:] = 0.5 * np.add(self.points[i], self.points[i + 1])

        return midpoints

    def calc_normals(self):
        normals = np.zeros((self.points.shape[0] - 1, self.points.shape[1]))
        for i in range(self.points.shape[0] - 1):
            normals[i, :] = self.rotate_vector(np.subtract(self.points[i], self.points[i + 1]), np.pi / 2)
            normals[i, :] /= np.linalg.norm(normals[i])

        return normals

    def check_if_vector_between(self, a, b, c):
        """
            check if c is in between a and b
        """
        #return np.cross(a, b) * np.cross(a, c) >= 0.0 and np.cross(c, b) * np.cross(c, a) >= 0.0
        #print((a,b,c))
        #print((a/np.linalg.norm(a),b/np.linalg.norm(b),c/np.linalg.norm(c)))
        #return self.cross2d(a, b) * self.cross2d(a, c) >= 0.0 and self.cross2d(c, b) * self.cross2d(c, a) >= 0.0
        return self.cross2d(a, c) * self.cross2d(b, c) <= 0.0

    def localize_ray(self, ray):
        for i in range(self.midpoints.shape[0]):
            #print(str(i)+' '+str(self.points[i])+' '+str(self.points[i+1]))
            a = np.subtract(self.points[i],ray.p)
            b = np.subtract(self.points[i + 1],ray.p)
            if self.check_if_vector_between(b, a, ray.k):
                #print("bla: "+str(i))
                return self.midpoints[i], self.normals[i], True

        return np.array([0.0, 0.0]), np.array([0.0, 0.0]), False

    def interact(self, ray):
        position, normal, hit = self.localize_ray(ray)
        absorbed = True
        if hit:
            if self.reflectivity > 0.0:
                if np.random.rand() <= self.reflectivity:
                    k = self.reflected_k(ray.k, normal)
                    absorbed = False
            elif self.transmittance > 0.0:
                if np.random.rand() <= self.transmittance:
                    k = self.transmitted_k(ray.k, normal)
                    absorbed = False

            if not absorbed:
                t = ray.calc_t(position)

                return position, k/np.linalg.norm(k), t, True

        return np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0, False




# spec_RayBundle = [
#     ('p0', float64[:,:]),
#     ('p', float64[:,:]),
#     ('k0', float64[:,:]),
#     ('k', float64[:,:]),
#     ('t0', float64),
#     ('t', float64),
#     ('phase', float64),
#     ('wavelength', float64),
#     ('f', float64),
#     ('interactions', int64[:]),
#     ('path', float64[:])
# ]
#
#
# @jitclass(spec_RayBundle)
# class RayBundle(object):
#     def __init__(self, p0, k, t0, wavelength, phase):
#         self.p0 = p0
#         self.p = p0
#         self.k0 = (k / np.linalg.norm(k)) * 2 * (np.pi / wavelength)
#         self.k = self.k0
#         self.t0 = t0
#         self.phase = phase
#         self.wavelength = wavelength
#         self.f = c / wavelength
#         self.interactions = 0
#         self.path = 0.0
#
#     def calc_path(self,point):
#         return np.sqrt(np.sum(np.square(np.subtract(self.p, point))))
#
#     def calc_r(self, t):
#         r = c * (t - self.t0)
#         if r > 0:
#             return r
#         else:
#             return 0.0
#
#     def calc_t(self, point):
#         return np.sqrt(np.sum(np.square(np.subtract(self.p, point)))) / c
#
#     # def calc_field(self, point, t):
#     #     r = np.linalg.norm(self.p - point)
#     #     field = cmath.exp(1j * (np.linalg.norm(self.k) * r - 2 * cmath.pi * self.f * (t - self.t0) + self.phase)).real
#     #     return field
#
#     def calc_field(self,t):
#         field = cmath.exp(1j * (np.linalg.norm(self.k) * self.path - 2 * cmath.pi * self.f * (t - self.t0) + self.phase)).real
#         return field
#
#     def set_wavelength(self, wavelength):
#         self.wavelength = wavelength
#         self.f = c / wavelength
#
#     def set_frequency(self, f):
#         self.f = f
#         self.wavelength = c / f
#
#     def is_back_at_origin(self):
#         p = self.p/np.linalg.norm(self.p)
#         p0 = self.p0/np.linalg.norm(self.p0)
#         k = self.k/np.linalg.norm(self.k)
#         k0 = self.k0/np.linalg.norm(self.k0)
#
#         if p[0]-p0[0] < 1e-5:
#             if p[1] - p0[1] < 1e-5:
#                 if k[0] - k0[0] < 1e-5:
#                     if k[1] - k0[1] < 1e-5:
#                         return True
#         return False
#         #return np.isclose(self.p[0], self.p0[0], rel_tol=0.001) and np.isclose(self.p[1], self.p0[1], rel_tol=0.001) and np.isclose(self.k[0], self.k0[0], rel_tol=0.001) and np.isclose(self.k[1], self.k0[1], rel_tol=0.001)
