import numpy as np

def h(x):
    return np.sqrt(1-np.square(x))

def pi_sum(N):
    x = np.linspace(-1,1,N)
    dx = x[1] - x[0]
    height = h(x)
    area = sum(dx*height)
    return area

print(2*pi_sum(N = 30000))