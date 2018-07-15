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
    "dipolar": 4,
}


spec_Wavelets = [
    ('r', float64[:,:]),
    ('k', float64[:,:]),
    ('t0', float64[:]),
    ('phases', float64[:]),
    ('wavelength', float64),
    #('f', float64),
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
        #self.f = c / wavelength
        self.mode = mode

    # def calc_r(self, t):
    #     r = c * (t - self.t0)
    #     if r > 0:
    #         return r
    #     else:
    #         return 0.0

    def calc_t_of_wavelet(self, index, point, n=1.0):
        return np.sqrt(np.sum(np.square(np.subtract(self.r[index,:], point)))) / (c/n)

    def calc_field(self, points, t, n=1.0):
        f = (c/n) / self.wavelength
        field = np.zeros(points.shape[0],dtype=np.complex128)
        for j in range(points.shape[0]):
            for i in range(self.n):
                r = np.linalg.norm(self.r[i,:] - points[j,:])
                #theta = self.angle_between(np.subtract(points[j,:],self.r[i,:]),self.k[i,:]).real
                #field[j] += cmath.cos(theta).real/r * cmath.exp(1j * (np.linalg.norm(self.k[i,:]) * r - 1j*2 * cmath.pi * f * (t - self.t0[i]) + self.phases[i])).real
                field[j] += 1 / r * cmath.exp(1j * ( ( np.linalg.norm(self.k[i, :]) * r
                                                     - 2 * cmath.pi * ((c/n)/self.wavelength) * (t - self.t0[i]) + self.phases[i])))
        return np.real(field)

    def field_at_r(self,index,t):
        n = 1.0
        f = (c/n) / self.wavelength
        field = cmath.exp( - 1j*2 * cmath.pi * f * (t - self.t0[index]) + self.phases[index]).real
        return field


    def calc_probability_of_wavelet(self,index, point1, point2):
        #v1 = np.subtract(self.r[index,:], point1)
        #v2 = np.subtract(self.r[index,:], point2)
        v1 = np.subtract(point1,self.r[index, :])
        v2 = np.subtract(point2,self.r[index, :])

        if self.mode == 1: # spherical
            phi2 = self.angle_between(v1, v2).real
            probability = np.abs((phi2) / (np.pi))

        elif self.mode == 2: # gaussian
            phi1 = self.angle_between(v1, self.k[index, :]).real
            #phi2 = self.angle_between(v1, v2).real
            phi2 = self.angle_between(v2, self.k[index, :]).real
            if phi2 > phi1:
                probability = 1 / (4 * np.pi) * (
                            np.sin(2 * phi1 + np.pi) - 2 * phi1 - np.sin(2 * phi2 + np.pi) + 2 * phi2)
            else:
                probability = 1 / (4 * np.pi) * (
                            np.sin(2 * phi2 + np.pi) - 2 * phi2 - np.sin(2 * phi1 + np.pi) + 2 * phi1)

            #probability = 1 / (4 * np.pi) * (np.sin(2 * phi1 + np.pi) - 2 * phi1 - np.sin(2 * phi2 + np.pi) + 2 * phi2)

        elif self.mode == 3: #ray
            #if (np.dot(np.cross(v1,self.k[index,:]),np.cross(v1,v2) >= 0)) and (np.dot(np.cross(v2,self.k[index,:]),np.cross(v2,v1) >= 0)):
            #if ( (v1[0]*self.k[index,0]-v1[1]*self.k[index,1]) * (v1[0]*v2[0]-v1[1]*v2[1])  ) >= 0 and\
            #        ( (v2[0]*self.k[index,0]-v2[1]*self.k[index,1]) * (v2[0]*v1[0]-v2[1]*v1[1]) ) >= 0 :
            #if (self.k[index,1]*v1[0]-self.k[index,0]*v1[1]) * (self.k[index,1]*v2[0]-self.k[index,0]*v2[1]) < 0:
            if self.is_between(self.k[index, :],v1,v2):
                probability = 1.0
            else:
                probability = 0.0

        elif self.mode == 4:  # dipolar
            phi1 = self.angle_between(v1, self.k[index, :]).real
            phi2 = self.angle_between(v1, v2).real
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

    def angle_between_clockwise(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        def determinant(v, w):
            return v[0] * w[1] - v[1] * w[0]
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        v = np.dot(v1_u, v2_u)
        if determinant(v1,v2) > 0:
            return cmath.acos(v)
        else:
            return 2*np.pi-cmath.acos(v)

    def angle_between_anticlockwise(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        def determinant(v, w):
            return v[0] * w[1] - v[1] * w[0]
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        v = np.dot(v1_u, v2_u)
        if determinant(v1,v2) < 0:
            return cmath.acos(v)
        else:
            return 2*np.pi-cmath.acos(v)

    def is_between(self,k,v1,v2):
        def determinant(v, w):
            return v[0] * w[1] - v[1] * w[0]
        return (determinant(k,v1)>0) ^ (determinant(k,v2)>0)

    def append_wavelets(self, wavelets):
        self.r = np.concatenate((self.r, wavelets.r))
        self.k = np.concatenate((self.k, wavelets.k))
        self.n += wavelets.n
        self.t0 = np.concatenate((self.t0, wavelets.t0))
        self.phases = np.concatenate((self.phases, wavelets.phases))


spec_Surface = [
    ('points', float64[:, :]),
    ('reflectivity', float64),
    ('transmittance', float64),
    ('n1', float64),
    ('n2', float64),
    ('midpoints', float64[:, :]),
    ('field', float64[:]),
    ('normals', float64[:, :]),
    ('count', int64),
]

@jitclass(spec_Surface)
class Surface(object):
    def __init__(self, points, reflectivity, transmittance, n1=1.0, n2=1.0):
        self.points = points
        self.reflectivity = reflectivity
        self.transmittance = transmittance
        self.n1 = n1
        self.n2 = n2
        self.midpoints = np.zeros((self.points.shape[0]-1,2))
        self.normals = np.zeros((self.points.shape[0] - 1,2))
        for i in range(self.points.shape[0]-1):
            self.midpoints[i] = 0.5 * np.add(self.points[i], self.points[i + 1])
            normal = self.rotate_vector(np.subtract(self.points[i], self.points[i + 1]), np.pi / 2)
            normal /= np.linalg.norm(normal)
            self.normals[i] = normal

        self.field = np.zeros(self.midpoints.shape[0])
        self.count = 0


    def flip_normals(self):
        for i in range(self.normals.shape[0]):
            self.normals[i] =  self.rotate_vector(self.normals[i], np.pi )


    def angle_between(self, v1, v2):
        """
        Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        def determinant(v, w):
            return v[0] * w[1] - v[1] * w[0]
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        v = np.dot(v1_u, v2_u)
        if determinant(v1,v2) > 0:
            return cmath.acos(v)
        else:
            return 2*np.pi-cmath.acos(v)


    def reflected_k(self, vector, normal):
        reflected = vector - (2 * (np.dot(normal, vector)) * normal)
        return reflected

    def transmitted_k(self, k, normal):
        alpha =  self.angle_between(k, normal).real
        #beta = cmath.sqrt(self.n2 ** 2 - self.n1 ** 2 * cmath.sin(alpha) ** 2).real / self.n2
        #print((self.n1/self.n2) * np.sin(alpha))
        beta = cmath.asin( (self.n1/self.n2) * np.sin(alpha) ).real
        #transmitted = self.rotate_vector(normal, np.pi - beta)
        transmitted = self.rotate_vector(normal, beta)
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
        """
        Return a random item from objects, with the weighting defined by weights
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

        if (wavelets.mode == 1) or (wavelets.mode == 2):
            if np.random.rand() >= np.sum(probabilities):
                probabilities /= np.sum(probabilities)
                index = self.weightedChoice(probabilities)
                return self.midpoints[index], self.normals[index], True
            else:
                return np.array([0.0, 0.0]), np.array([0.0, 0.0]), False
        elif wavelets.mode == 3:
            index = -1
            for j in range(len(probabilities)):


                if probabilities[j] > 0.99:
                    index = j
            if index >= 0:
                return self.midpoints[index], self.normals[index], True
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
                t = wavelets.calc_t_of_wavelet(index,position,self.n1)
                return position, k, t, True

        return np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0, False

    # def intensity_on_surface(self,wavelets):
    #     hits = np.zeros(wavelets.n, dtype=np.float64)
    #     rs = np.zeros((wavelets.n, 2), dtype=np.float64)
    #     ints = np.zeros(wavelets.n, dtype=np.float64)
    #
    #     r = np.zeros(2)
    #     k = np.zeros(2)
    #     t = 0.0
    #     hit = False
    #
    #     for i in range(wavelets.n):
    #         #r, normal, hit = self.localize_wavelet(wavelets, i)
    #         rs[i] = r
    #         hits[i] = hit
    #         ints[i] = wavelets.field_at_r(i,1.0)
    #
    #     indices = (hits > 0)
    #
    #     return rs[indices,:], ints[indices]

    def add_field_from_wavelets(self,wavelets):
        field = np.zeros(wavelets.n, dtype=np.float64)
        self.count += wavelets.n
        for i in range(wavelets.n):
            field[i] = wavelets.field_at_r(i,1.0)

        for i in range(len(field)):
            for j in range(self.field.shape[0]):
                if ((wavelets.r[i, 1] > self.points[j, 1]) and (wavelets.r[i, 1] < self.points[j + 1, 1])) \
                            or ((wavelets.r[i, 0] > self.points[j, 0]) and (wavelets.r[i, 0] < self.points[j + 1, 0])):
                    self.field[j] += field[i]*np.dot(self.rotate_vector(wavelets.k[i,:],np.pi/2),np.subtract(self.points[j+1,:],self.points[j,:]))
                    #self.field[j] += field[i]*np.dot(wavelets.k[i,:],self.normals[j,:])


    def interact_with_all_wavelets(self, wavelets):
        hits = np.zeros(wavelets.n, dtype=np.float64)
        rs = np.zeros((wavelets.n,2), dtype=np.float64)
        ks = np.zeros((wavelets.n,2), dtype=np.float64)
        ts = np.zeros(wavelets.n, dtype=np.float64)

        r = np.zeros(2)
        k = np.zeros(2)
        t = 0.0
        hit = False

        for i in range(wavelets.n):
            r, k, t, hit = self.interact_with_wavelet(wavelets,i)
            hits[i] = hit
            rs[i] = r
            ks[i] = k
            ts[i] = t

        indices = (hits > 0)
        # (self, r, k, t0, wavelength, phases, mode):
        new_wavelets = Wavelets(rs[indices,:],ks[indices,:],ts[indices],wavelets.wavelength,wavelets.phases[indices],wavelets.mode)
        return new_wavelets

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
def gen_concave_points(p0, r, height, num):
    v = [r,0]
    thetas = np.linspace(height / 2, -height / 2, num)
    points = np.zeros((num, 2))
    for i, theta in enumerate(thetas):
        R = gen_rotation_matrix(theta)
        buf = np.dot(R, v)
        points[i] = p0 + buf
    return points





class Lense(object):

    def __init__(self, x, y, height, reflectivity=0.0,transmittance=1.0, n1=1.0, n2=1.5, num=128):
        self.n1 = n1
        self.n2 = n2
        concave1, concave2 = self._generate_lens_points(num, height)
        concave1[:, 0] += x
        concave2[:, 0] += x
        concave1[:, 1] += y
        concave2[:, 1] += y
        self.front = Surface(concave1, reflectivity=reflectivity, transmittance=transmittance, n1=n1, n2=n2)
        self.back = Surface(concave2, reflectivity=reflectivity, transmittance=transmittance, n1=n2, n2=n1)
        self.front.flip_normals()

    def _generate_lens_points(self, num, height):
        p0 = np.array([0.0, 0.0])
        concave1 = gen_concave_points(p0, -2.0, np.pi / 3, num)
        concave1[:, 1] = concave1[:, 1] / concave1[:, 1].max()
        concave1[:, 1] = concave1[:, 1] * (height / 2)
        concave1[:, 0] -= concave1[:, 0].max()

        p0 = np.array([0.0, 0.0])
        concave2 = gen_concave_points(p0, 2.0, np.pi / 3, num)
        concave2[:, 1] = concave2[:, 1] / concave2[:, 1].max()
        concave2[:, 1] = concave2[:, 1] * (height / 2)
        concave2[:, 0] -= concave2[:, 0].min()

        return concave1, concave2

    def shift(self, dx, dy):
        self.front.points[:, 0] += dx
        self.back.points[:, 0] += dx
        self.front.points[:, 1] += dy
        self.back.points[:, 1] += dy