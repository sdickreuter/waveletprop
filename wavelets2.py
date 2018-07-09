import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float64, int64, void, boolean, uint64
import cmath

# c = 2.998e8  # m/s
c = 1.0  # m/s

modes = {
    "spherical": 1,
    "gaussian": 2,
    "ray": 3,
}


spec_Wavelets = [
    ('r', float64[:,:]),
    ('k', float64[:,:]),
    ('t0', float64[:]),
    ('phases', float64[:]),
    ('wavelength', float64),
    ('f', float64),
    ('mode', uint64),
    ('n', int64)
]


@jitclass(spec_Wavelets)
class Wavelets(object):
    def __init__(self, r, k, t0, wavelength, phases, mode):
        self.r = r
        self.n = r.shape[0]
        self.k = np.zeros((self.n,2))
        for i in range(self.n):
            self.k[i,:] = k[i,:] / np.linalg.norm(k[i,:]) * 2 * np.pi / wavelength
        self.t0 = t0
        self.phases = phases
        self.wavelength = wavelength
        self.f = c / wavelength
        self.mode = mode

    # def calc_r(self, t):
    #     r = c * (t - self.t0)
    #     if r > 0:
    #         return r
    #     else:
    #         return 0.0

    def calc_t_of_wavelet(self, index, point):
        return np.sqrt(np.sum(np.square(np.subtract(self.r[index,:], point)))) / c

    def calc_field(self, point, t):
        field = 0
        for i in range(self.n):
            r = np.linalg.norm(self.r[i] - point)
            field += cmath.exp(1j * (np.linalg.norm(self.k[i,:]) * r - 2 * cmath.pi * self.f * (t - self.t0[i]) + self.phases[i])).real
        return field

    def calc_probability_of_wavelet(self,index, point1, point2):
        v1 = np.subtract(self.r[index,:], point1)
        v2 = np.subtract(self.r[index,:], point2)
        phi1 = self.angle_between(v1, self.k[index,:]).real
        phi2 = self.angle_between(v1, v2).real
        phi3 = self.angle_between(v2, self.k[index,:]).real
        if self.mode == 1:
            probability = np.abs((phi2) / (np.pi))
        elif self.mode == 2:
            probability = 1 / (2 * np.pi) * (
                np.sin(2 * phi1) + 2 * phi1 - np.sin(2 * (phi1 + phi2)) - 2 * (phi1 + phi2))
            probability = np.abs(probability)
        elif self.mode == 3:
            if (phi1-phi3) < phi2:
                probability = 1.0
            else:
                probability = 0.0
        return probability


    # def unit_vector(self, vector):
    #     """ Returns the unit vector of the vector.  """
    #     return vector / np.linalg.norm(vector)

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

# def angle_between(v1, v2):
#     """ Returns the angle in radians between vectors 'v1' and 'v2'::
#     """
#     # v1_u = self.unit_vector(v1)
#     # v2_u = self.unit_vector(v2)
#     v1_u = v1 / np.linalg.norm(v1)
#     v2_u = v2 / np.linalg.norm(v2)
#     v = np.dot(v1_u, v2_u)
#     if v > 1.0:
#         v = 1.0
#     elif v < -1.0:
#         v = -1.0
#     # return np.arccos(v)
#     return cmath.acos(v)
#
r = np.array([0,0])
k = np.array([1,0])
point1 = np.array([1,0.1])
point2 = np.array([1,-0.1])
v1 = np.subtract(point1,r)
v2 = np.subtract(point2,r)
phi1 = angle_between(v1,k).real
phi2 = angle_between(v1,v2).real
phi3 = angle_between(v2,k).real
phi1
phi2
phi3
phi1-phi3




spec_Surface = [
    ('points', float64[:, :]),
    ('reflectivity', float64),
    ('transmittance', float64),
    ('n1', float64),
    ('n2', float64),
]

@jitclass(spec_Surface)
class Surface(object):
    def __init__(self, points, reflectivity, transmittance, n1=1.0, n2=1.0):
        self.points = points
        self.reflectivity = reflectivity
        self.transmittance = transmittance
        self.n1 = n1
        self.n2 = n2

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

    def weightedChoice(self, weights):
        """Return a random item from objects, with the weighting defined by weights
        (which must sum to 1).
        From: http://stackoverflow.com/a/10803136
        """
        cs = np.cumsum(weights)  # An array of the weights, cumulatively summed.
        idx = np.sum(cs < np.random.rand())  # Find the index of the first weight over a random value.
        return idx

    def localize_wavelet(self, wavelets, index):
        probabilities = np.zeros(self.points.shape[0] - 1, dtype=np.float64)
        for i in range(self.points.shape[0] - 1):
            probabilities[i] = wavelets.calc_probability_of_wavelet(index,self.points[i], self.points[i + 1])

        if np.random.rand() >= np.sum(probabilities):
            probabilities /= np.sum(probabilities)
            index = self.weightedChoice(probabilities)
            position = 0.5 * np.add(self.points[index], self.points[index + 1])
            normal = self.rotate_vector(np.subtract(self.points[index], self.points[index + 1]), np.pi / 2)
            normal /= np.linalg.norm(normal)
            return position, normal, True
        else:
            return np.array([0.0, 0.0]), np.array([0.0, 0.0]), False

    def interact_with_wavelet(self, wavelets, index):
        position, normal, hit = self.localize_wavelet(wavelets, index)
        absorbed = True
        if hit:
            if self.reflectivity > 0.0:
                if np.random.rand() <= self.reflectivity:
                    k = self.reflected_k(wavelets.k[index,:], normal)
                    absorbed = False
            elif self.transmittance > 0.0:
                if np.random.rand() <= self.transmittance:
                    k = self.transmitted_k(wavelets.k[index,:], normal)
                    absorbed = False

            if not absorbed:
                t = wavelets.calc_t_of_wavelet(index,position)
                return position, k, t, True

        return np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0, False

    def interact_with_all_wavelets(self, wavelets):
        hits = np.zeros(wavelets.n, dtype=np.float64)
        rs = np.zeros((wavelets.n,2), dtype=np.float64)
        ks = np.zeros((wavelets.n,2), dtype=np.float64)
        ts = np.zeros(wavelets.n, dtype=np.float64)

        r = np.zeros(2)
        k = np.zeros(2)
        t = 0
        hit = False

        for i in range(wavelets.n):
            r, k, t, hit = self.interact_with_wavelet(wavelets,i)
            hits[i] = hit
            rs[i] = r
            ks[i] = k
            ts[i] = t

        indices = (hits > 0)
        # (self, r, k, t0, wavelength, phases, mode):
        new_wavelets = Wavelets(rs[indices,:],ks[indices,:],ts[indices],wavelets.wavelength,wavelets.phases,wavelets.mode)
        return new_wavelets