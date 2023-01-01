import numpy as np

def h(x):
    return np.sqrt(1-np.square(x))

def pi_for(N):
    dx = 2/N
    x = -1
    area = 0
    for i in range(N):
        x+=dx
        area+=dx*h(x)
    return area

print(2*pi_for(N=30000))