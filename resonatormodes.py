import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, jitclass, float64, int64, void, boolean
import cmath
from raytracing import *
import seaborn as sns

sns.set_style("ticks")

L = 0.3#1.0
c = 2.998e8#1.0

def gaussian(x, x0, sigma):
    return np.exp(-np.power(x - x0, 2.) / (2 * np.power(sigma, 2.)))

def resonant_frequencies(L,n):
    return n*(c/(2*L))


@jit
def calc_field(f, x, t, t0 = 0):
    wavelength = c/f
    k = 2*np.pi/wavelength
    #return (np.exp(1j * (k * x - 2 * cmath.pi * f * t+np.pi/2))+np.exp(1j * (k * x + 2 * cmath.pi * f * t+np.pi/2))).real
    #return np.cos(np.pi*f*t)*np.cos(k*x)
    return 2*np.sin(k*x)*np.sin(2*np.pi*f*(t-t0))

@jit
def wl2f(wavelength):
    return c/wavelength


x = np.linspace(0,L/1,5000)

wl_mean = 500e-9#m
f_mean = c/wl_mean
mode_number = np.round(f_mean*(2*L/c))
print(mode_number)
frequencies = resonant_frequencies(L,np.arange(1,100,1)+mode_number)

#print(np.diff(frequencies))

envelope = gaussian(frequencies,np.mean(frequencies),(frequencies.max()-frequencies.min())/8)

#plt.plot(envelope)
#plt.show()

# for i in range(len(frequencies)):
#     plt.plot(calc_field(frequencies[i],x,2.0))
# plt.show()


# cmap = plt.get_cmap('viridis')
# colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(frequencies))]
#
# ts = np.linspace(0,2.0,2000)
# for i,f in enumerate(frequencies):
#     plt.plot(ts,np.sin(2*np.pi*f*ts),color=colors[i])
# plt.tight_layout()
# plt.show()


# cmap = plt.get_cmap('viridis')
# colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(frequencies))]
#
# #ts = np.linspace(0,1.0,5)
# ts = np.linspace(0,L/c*1.5,10)
#
# for t in ts:
#
#     f = np.zeros((frequencies.shape[0],x.shape[0]))
#     for i,wl in enumerate(frequencies):
#         f[i,:] = calc_field(wl,x,t)*envelope[i]
#
#     #int = np.square(np.sum(f,axis=0))
#     int = np.sum(f,axis=0)
#     plt.plot(x,int)
#     #for i in range(len(frequencies)):
#     #    plt.plot(calc_field(frequencies[i], x, t)+t*32,color=colors[i])
#
# plt.tight_layout()
# plt.show()



wl_low = 300e-9#m
wl_high = 1000e-9#m
f_high = c/wl_low
f_low = c/wl_high
mode_low = np.round(f_low*(2*L/c))
mode_high = np.round(f_high*(2*L/c))
print(mode_low)
print(mode_high)
frequencies = resonant_frequencies(L,np.arange(mode_low,mode_high,1))
print(len(frequencies))

gain = gaussian(frequencies,c/(520e-9),c/(520e-9)/10)
norm = np.sum(gain)
gain /= norm

# plt.plot(frequencies,gain)
# plt.show()


loss = 0.1 #percent
quantum_efficiency = 0.75
photon_flux = 4.5 * 1e17 # photons/second, ~450nm wavelength
spontaneous_emission_rate = 1/(15e-9) # photons/second
triplet_rate = 1/(15e-6) # rought estimate # photons/second
flow_rate = 1e6 # replacement of dye molecules due to flow
einstein_coefficient = 0.001 # arbitrary for now

n = 1e6 # number of dye molecules
n_ex = 0.0 # number of excited dye molecules
n_trip = 0.0 # number of dye molecules in triplet state
n_ground = n # number of dye molecules in ground state


ts = np.arange(0,100,0.01)*1e-15

print(ts.max())

mode_intensity = np.zeros(gain.shape[0])
#mode_t0 = np.zeros((gain.shape[0],ts.shape[0]))

dt = ts[1]-ts[0]


photons_in_cavity = 0.0
log = np.zeros(ts.shape[0])
for i,t in enumerate(ts):
    r_pump = photon_flux*quantum_efficiency*n_ground
    r_spont = spontaneous_emission_rate*n_ex
    r_triplet = triplet_rate*n_ex


    photons_in_cavity += r_spont*dt


    r_gain = (n_ex-n_ground)*photons_in_cavity*einstein_coefficient
    n_ex += (r_pump-r_spont-r_triplet-r_gain-flow_rate/2)*dt
    n_trip += (r_triplet-flow_rate/2)*dt
    n_ground += (-r_pump+r_gain+r_spont+flow_rate)

    photons_in_cavity *= (1-loss)
    log[i] = n_ex



plt.plot(ts,log)
plt.show()

print(log)
