import numpy as np

def h(x):
    return np.sqrt(1-np.square(x))

def pi_np_sum(N):
    x = np.linspace(-1,1,N)
    dx = x[1] - x[0]
    height = h(x)
    area = np.sum(dx*height)
    return area

print(2*pi_np_sum(N = 30000))