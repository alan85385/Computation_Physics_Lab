import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from numba import jit, njit, prange, set_num_threads
import time

"""

This program solve 3D direct N-particles simulations 
under gravitational forces. 

This file contains two classes:

1) Particles: describes the particle properties
2) NbodySimulation: describes the simulation

Usage:

    Step 1: import necessary classes

    from nbody import Particles, NbodySimulation

    Step 2: Write your own initialization function

    
        def initialize(particles:Particles):
            ....
            ....
            particles.set_masses(mass)
            particles.set_positions(pos)
            particles.set_velocities(vel)
            particles.set_accelerations(acc)

            return particles

    Step 3: Initialize your particles.

        particles = Particles(N=100)
        initialize(particles)


    Step 4: Initial, setup and start the simulation

        simulation = Simulation(particles)
        simulation.setip(...)
        simulation.evolve(dt=0.001, tmax=10)


Author: Kuo-Chuan Pan, NTHU 2022.10.30
For the course, computational physics lab

"""

class Particles:
    """
    
    The Particles class handle all particle properties

    for the N-body simulation. 

    """
    def __init__(self,N:int=100):
        """
        Prepare memories for N particles

        :param N: number of particles.

        By default: particle properties include:
                nparticles: int. number of particles
                _masses: (N,1) mass of each particle
                _positions:  (N,3) x,y,z positions of each particle
                _velocities:  (N,3) vx, vy, vz velocities of each particle
                _accelerations:  (N,3) ax, ay, az accelerations of each partciel
                _tags:  (N)   tag of each particle
                _time: float. the simulation time 

        """
        
        self.nparticles     = N
        self._masses        = np.ones((N,1))
        self._positions     = np.zeros((N,3))
        self._velocities    = np.zeros((N,3))
        self._accelerations = np.zeros((N,3))
        self._tags          = np.linspace(1,N,N)
        self._time          = 0.

        return

    def set_masses(self, mass):
        self._masses = mass
        return

    def set_positions(self, pos):
        self._positions = pos
        return
    
    def set_velocities(self, vel):
        self._velocities = vel
        return

    def set_accelerations(self, acc):
        self._accelerations = acc
        return

    def set_particles_tags(self, IDs):
        self._tags = IDs
        return
    
    def get_tags(self):
        return self._tags
    
    def get_masses(self):
        return self._masses
    
    def get_positions(self):
        return self._positions
    
    def get_velocities(self):
        return self._velocities
    
    def get_accelerations(self):
        return self._accelerations

    def output(self,fn, time):
        """
        Write simulation data into a file named "fn"


        """
        mass = self._masses
        pos  = self._positions
        vel  = self._velocities
        acc  = self._accelerations
        tag  = self._tags
        header = """
                ----------------------------------------------------
                Data from a 3D direct N-body simulation. 

                rows are i-particle; 
                coumns are :mass, tag, x ,y, z, vx, vy, vz, ax, ay, az

                NTHU, Computational Physics Lab

                ----------------------------------------------------
                """
        header += "Time = {}".format(time)
        np.savetxt(fn,(tag[:],mass[:,0],pos[:,0],pos[:,1],pos[:,2],
                            vel[:,0],vel[:,1],vel[:,2],
                            acc[:,0],acc[:,1],acc[:,2]),header=header)

        return

class NbodySimulation:
    """
    
    The N-body Simulation class.
    
    """

    def __init__(self,particles:Particles):
        """
        Initialize the N-body simulation with given Particles.

        :param particles: A Particles class.  
        
        """

        # store the particle information
        self.nparticles = particles.nparticles
        self.particles  = particles

        # Store physical information
        self.time  = 0.0  # simulation time

        # Set the default numerical schemes and parameters
        self.setup()
        
        return

    def setup(self, G=1, 
                    rsoft=0.01, 
                    method="RK4", 
                    io_freq=10, 
                    io_title="particles",
                    io_screen=True,
                    visualized=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_title: the output header
        :param io_screen: print message on screen or not.
        :param visualized: on the fly visualization or not. 
        
        """
        
        self.G          = G
        self.rsoft      = rsoft
        self.method     = method
        self.io_freq    = io_freq
        self.io_title   = io_title
        self.io_screen  = io_screen
        self.visualized = visualized
        
        return

    def evolve(self, dt:float=0.01, tmax:float=1):
        """

        Start to evolve the system

        :param dt: time step
        :param tmax: the finial time
        
        """
        
        N = int(np.ceil(tmax/dt))
        times = np.linspace(0, tmax, N)
        particles = self.particles
        nparticles = self.nparticles
        sol = np.zeros((nparticles,3,N))
        io_freq    = self.io_freq
        io_screen  = self.io_screen

        method = self.method
        if method=="Euler":
            _update_particles = self._update_particles_euler
        elif method=="RK2":
            _update_particles = self._update_particles_rk2
        elif method=="RK4":
            _update_particles = self._update_particles_rk4    
        else:
            print("No such update meothd", method)
            quit() 

        # prepare an output folder for lateron output
        io_folder = "data_"+self.io_title
        Path(io_folder).mkdir(parents=True, exist_ok=True)
        
        # ====================================================
        #
        # The main loop of the simulation
        #
        # =====================================================

        for n in range(N):
            #print(n)
            particles = _update_particles(dt, particles)
            sol[:,:,n] = particles.get_positions()
            '''
            if n%io_freq==0:
                if io_screen:
                    print(n)
                    fn = io_folder + '/data_' + self.io_title + '_' + str(n).zfill(5) + '.txt'
                    self.particles.output(fn, times[n])
            '''
            

        print("Done!")
        return sol

    def _calculate_acceleration(self, mass, pos):
        """
        Calculate the acceleration.
        """
        acc = pos.copy()
        
        N = self.nparticles
        ax = np.zeros(N)
        ay = np.zeros(N)
        az = np.zeros(N)
        rsoft      = self.rsoft
        G          = self.G
        
        t0 = time.time()
        ax,ay,az = _calculate_acceleration_kernal(N,pos,mass,G,rsoft,ax,ay,az)
        t1 = time.time()
        #print('time',t1-t0)
        print(ax)
        print(ay)
        print(az)
        '''
        for i in range(N):
            
            for j in range(N):
                
                if (j > i): # because F[I,j] = - F[j,i]
                    
                    seperation = pos[i]-pos[j]
                    distance   = np.linalg.norm(seperation)
                    F = -self.G*mass[i]*mass[j]*(seperation)/(rsoft+distance)**3
                    #print(distance)
                    
                    Fx = F[0]
                    Fy = F[1]
                    Fz = F[2]
                    
                    ax[i] += Fx /mass[i] # same for ay and az
                    ax[j] -= Fx /mass[j] # same for ay and az
                    ay[i] += Fy /mass[i] # same for ay and az
                    ay[j] -= Fy /mass[j] # same for ay and az
                    az[i] += Fz /mass[i] # same for ay and az
                    az[j] -= Fz /mass[j] # same for ay and az
        '''
        
        acc[:,0] = ax
        acc[:,1] = ay
        acc[:,2] = az
                    
        return acc

    def _update_particles_euler(self, dt, particles:Particles):
        
        mass = particles.get_masses()
        pos  = particles.get_positions()
        vel  = particles.get_velocities()
        
        acc  = self._calculate_acceleration(mass, pos)
        
        pos = pos + vel*dt
        vel = vel + acc*dt
        
        particles.set_positions(pos)
        particles.set_velocities(vel)
        particles.set_accelerations(acc)
        
        return particles

    def _update_particles_rk2(self, dt, particles:Particles):
        
        mass = particles.get_masses()
        pos  = particles.get_positions()
        vel  = particles.get_velocities()
        
        acc  = self._calculate_acceleration(mass, pos) # k1[1]
        
        pos2 = pos + vel*dt
        vel2 = vel + acc*dt
        
        acc2 = self._calculate_acceleration(mass, pos2)
        
        next_pos = pos + dt/2*(vel + vel2)
        next_vel = vel + dt/2*(acc + acc2)
        
        particles.set_positions(next_pos)
        particles.set_velocities(next_vel)
        
        return particles

    def _update_particles_rk4(self, dt, particles:Particles):
        
        mass = particles.get_masses()
        pos1  = particles.get_positions()
        vel1  = particles.get_velocities()
        
        acc1  = self._calculate_acceleration(mass, pos1) # k1[1]
        
        pos2 = pos1 + 0.5*vel1*dt
        vel2 = vel1 + 0.5*acc1*dt
        
        acc2  = self._calculate_acceleration(mass, pos2) # k2[1]
        
        pos3 = pos1 + 0.5*vel2*dt
        vel3 = vel1 + 0.5*acc2*dt
        
        acc3  = self._calculate_acceleration(mass, pos3) # k3[1]
        
        pos4 = pos1 + vel3*dt
        vel4 = vel1 + acc3*dt
        
        acc4  = self._calculate_acceleration(mass, pos4) # k4[1]
        
        next_pos = pos1 + (1/6)*dt*(vel1+2*vel2+2*vel3+vel4)
        next_vel = vel1 + (1/6)*dt*(acc1+2*acc2+2*acc3+acc4)
        
        particles.set_positions(next_pos)
        particles.set_velocities(next_vel)
        
        return particles

@jit(nopython=True)
def _calculate_acceleration_kernal(N,pos,mass,G,rsoft,ax,ay,az):
    
    

    for i in range(N):
            
        for j in range(N):
                
            if (j > i): # because F[I,j] = - F[j,i]
                    
                seperation = pos[i]-pos[j]
                distance   = np.linalg.norm(seperation)
                F = -G*mass[i]*mass[j]*(seperation)/(rsoft+distance)**3
                #print(distance)
                
                    
                Fx = F[0]
                Fy = F[1]
                Fz = F[2]
                    
                ax[i] += Fx /mass[i,0] # same for ay and az
                ax[j] -= Fx /mass[j,0] # same for ay and az
                ay[i] += Fy /mass[i,0] # same for ay and az
                ay[j] -= Fy /mass[j,0] # same for ay and az
                az[i] += Fz /mass[i,0] # same for ay and az
                az[j] -= Fz /mass[j,0] # same for ay and az

    
    return ax, ay, az

if __name__=='__main__':
    
    def initialSolarSystem(particles:Particles):
        """
        initial Sun-Earth system
        """
    
        particles = Particles(2)
        particles.set_masses(np.array([[msun],[mearth]]))
    
        M = msun+mearth
        seperation = 1.49598e13
        period = np.sqrt(4*np.pi**2*seperation**3/G/M)
        w = 2*np.pi/period
        x1 = -mearth*seperation/M
        x2 = msun*seperation/M
        v1 = x1*w
        v2 = x2*w
        particles.set_positions(np.array([[x1, 0, 0],[x2, 0, 0]]))
        particles.set_velocities(np.array([[0, v1, 0],[0, v2, 0]]))
        return particles


    # test Particles() here
    particles = Particles(N=2)
    # test NbodySimulation(particles) here
    problem_name = "solar_earth"
    msun   = 1.989e33   # gram
    mearth = 5.97219e27 # gram
    au     = 1.496e13   # cm
    day    = 86400      # sec
    year   = 365*day    # sec
    G      = 6.67e-8   # cgs
    particles = initialSolarSystem(particles)
    sim = NbodySimulation(particles)
    sim.setup(G=G,method="RK4",io_freq=20,io_title=problem_name,io_screen=False,visualized=False)
    sim.evolve(dt=0.1*day,tmax=5*year)
    print("Done")