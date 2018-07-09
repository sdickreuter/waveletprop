import locale

locale.setlocale(locale.LC_NUMERIC, 'C')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float64, int64, void, boolean
import cmath

# c = 2.998e8  # m/s
c = 1.0  # m/s

spec_Wavelet = [
    ('r', float64[:]),
    ('k', float64[:]),
    ('t0', float64),
    ('phase', float64),
    ('wavelength', float64),
    ('f', float64),
    ('pulsewidth', float64),
    ('spherical', boolean),
]


@jitclass(spec_Wavelet)
class Wavelet(object):
    def __init__(self, r, k, t0, wavelength, phase, pulsewidth, spherical):
        self.r = r
        self.k = k / np.linalg.norm(k) * 2 * np.pi / wavelength
        self.t0 = t0
        self.phase = phase
        self.wavelength = wavelength
        self.f = c / wavelength
        self.pulsewidth = pulsewidth  # seconds
        self.spherical = spherical

    def calc_r(self, t):
        r = c * (t - self.t0)
        if r > 0:
            return r
        else:
            return 0.0

    def calc_t(self, point):
        return np.sqrt(np.sum(np.square(np.subtract(self.r, point)))) / c

    def calc_field(self, point, t):
        # t = self.calc_t(point)
        r = np.linalg.norm(self.r - point)
        # field = 1/r * cmath.exp( 1j * ( np.linalg.norm(self.k)*r + 2*cmath.pi*self.f*(t-self.t0)+self.phase)  ).real
        field = cmath.exp(1j * (np.linalg.norm(self.k) * r - 2 * cmath.pi * self.f * (t - self.t0) + self.phase)).real
        ##r0 = self.calc_r(t)
        ##field = field * 1 / (self.pulsewidth * np.sqrt(2 * np.pi)) * np.exp(
        ##    -0.5 * np.square((r - r0) / self.pulsewidth))
        # field = np.abs(cmath.exp(1j * (np.linalg.norm(self.k) * r - 2 * cmath.pi * self.f * (t - self.t0) + self.phase)))
        return field

    def calc_probability(self, point1, point2):
        v1 = np.subtract(self.r, point1)
        v2 = np.subtract(self.r, point2)
        phi1 = self.angle_between(v1, self.k).real
        phi2 = self.angle_between(v1, v2).real
        if self.spherical:
            probability = np.abs((phi2) / (np.pi))
        else:
            probability = 1 / (2 * np.pi) * (
                np.sin(2 * phi1) + 2 * phi1 - np.sin(2 * (phi1 + phi2)) - 2 * (phi1 + phi2))
            probability = np.abs(probability)
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

    def localize_wavelet(self, wavelet):
        probabilities = np.zeros(self.points.shape[0] - 1, dtype=np.float64)
        for i in range(self.points.shape[0] - 1):
            probabilities[i] = wavelet.calc_probability(self.points[i], self.points[i + 1])

        if np.random.rand() >= np.sum(probabilities):
            probabilities /= np.sum(probabilities)
            index = self.weightedChoice(probabilities)
            position = 0.5 * np.add(self.points[index], self.points[index + 1])
            normal = self.rotate_vector(np.subtract(self.points[index], self.points[index + 1]), np.pi / 2)
            normal /= np.linalg.norm(normal)
            return position, normal, True
        else:
            return np.array([0.0, 0.0]), np.array([0.0, 0.0]), False

    def interact(self, wavelet):
        position, normal, hit = self.localize_wavelet(wavelet)
        absorbed = True
        if hit:
            if self.reflectivity > 0.0:
                if np.random.rand() <= self.reflectivity:
                    k = self.reflected_k(wavelet.k, normal)
                    absorbed = False
            elif self.transmittance > 0.0:
                if np.random.rand() <= self.transmittance:
                    k = self.transmitted_k(wavelet.k, normal)
                    absorbed = False

            if not absorbed:
                t = wavelet.calc_t(position)
                return position, k, t, True

        return np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0, False
