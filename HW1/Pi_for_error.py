import numpy as np
import matplotlib.pyplot as plt

def h(x):
    return np.sqrt(1-np.square(x))

def pi_for(N):
    dx = 2/N
    x = -1
    area = 0
    for i in range(N):
        area+=dx*h(x)
        x+=dx
    return area

def error(N):
    areas = []
    X = np.geomspace(1, N, num=int(N/500), dtype=int)
    print(X)
    for x in X:
        areas.append(pi_for(x))
    areas = 2*np.array(areas)
    print(areas)
    plt.plot(X, 100*abs(areas-np.pi)/np.pi)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('steps (N)')
    plt.ylabel('error (%)')
    plt.title(r'Computational $\pi$ accuracy compare to real $\pi$')
    plt.savefig('pi_error.pdf')

error(100000)
