import numpy as np
import matplotlib.pyplot as plt



def prob(phi1,phi2):
    probability = 1 / 2 * (np.sin(phi1)^2 - np.sin(phi2)^2)
    return probability

def prob(phi1,phi2):
    probability = 1 / (4*np.pi) * (np.sin(2*phi1+np.pi) - 2*phi1 - np.sin(2*phi2+np.pi)+2*phi2)
    return probability

phi1 = np.linspace(-np.pi/2,np.pi/2,100)
phi2 = phi1.copy()
phi2 += np.pi/100

plt.plot(phi1,prob(phi1,phi2))
plt.show()
