import numpy as np

def B(v, T, h=6.626*10**(-34), c=3*10**8, k=1.38*10**(-23)):
    return 2*h*v**3/c**2/(np.exp(h*v/k/T)-1)

def sig(v, T):
    return np.pi*(np.sum(B(v, T)*(v[1]-v[0])))/T**4

v = np.linspace(0.015, 2*10**15, 10**7)
sig_calculation = sig(v, 6000)
sig_exp =  5.670367*10**(-8)
difference = abs(sig_calculation-sig_exp)/sig_exp*100
print(f'The Stefan-Boltzmann constant of my program: {sig_calculation:.7g}\nThe experimental Stefan-Boltzmann constant: {sig_exp:.7g}\nThe difference between them is {difference:.2f}%')