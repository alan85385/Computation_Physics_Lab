import numpy as np
import matplotlib.pyplot as plt
import time

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

def pi_sum(N):
    x = np.linspace(-1,1,N)
    dx = x[1] - x[0]
    height = h(x)
    area = sum(dx*height)
    return area

def pi_np_sum(N):
    x = np.linspace(-1,1,N)
    dx = x[1] - x[0]
    height = h(x)
    area = np.sum(dx*height)
    return area

def test_pi_for(N_max, test_times):
    Ns = np.geomspace(2, N_max+1, num=int(N_max/10), dtype=int)
    t0 = []
    t1 = []
    for N in Ns:
        t0.append(time.time())
        for i in range(test_times):
            pi_for(N)
        t1.append(time.time())
    t0 = np.array(t0)
    t1 = np.array(t1)
    delta_t = (t1-t0)/test_times
    return Ns, delta_t

def test_pi_sum(N_max, test_times):
    Ns = np.geomspace(2, N_max+1, num=int(N_max/10), dtype=int)
    t0 = []
    t1 = []
    for N in Ns:
        t0.append(time.time())
        for i in range(test_times):
            pi_sum(N)
        t1.append(time.time())
    t0 = np.array(t0)
    t1 = np.array(t1)
    delta_t = (t1-t0)/test_times
    return Ns, delta_t

def test_pi_np_sum(N_max, test_times):
    Ns = np.geomspace(2, N_max+1, num=int(N_max/10), dtype=int)
    t0 = []
    t1 = []
    for N in Ns:
        t0.append(time.time())
        for i in range(test_times):
            pi_np_sum(N)
        t1.append(time.time())
    t0 = np.array(t0)
    t1 = np.array(t1)
    delta_t = (t1-t0)/test_times
    return Ns, delta_t


def plot(N_max, test_times):
    Ns, delta_t_pi_for = test_pi_for(N_max, test_times)
    Ns, delta_t_pi_sum = test_pi_sum(N_max, test_times)
    Ns, delta_t_pi_np_sum = test_pi_np_sum(N_max, test_times)
    plt.plot(Ns, delta_t_pi_for, label = 'python for')
    plt.plot(Ns, delta_t_pi_sum, label = 'python sum')
    plt.plot(Ns, delta_t_pi_np_sum, label = 'np sum')
    plt.title('The total time for calculation of the three different numerical methods')
    plt.xlabel('steps (N)')
    plt.ylabel('time (s)')
    plt.legend()
    plt.savefig('test_pi_performance.pdf')

if __name__ == '__main__':
    plot(1000, 10)

        
