# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:09:23 2016

@author: arsart and buu
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
The state of the particles is defined by:
[xi,yi,Oi,Wi]
where:
xi = x position
yi = y position
Oi = angle between [-pi,pi]
Wi = angular speed
"""


class ParticleBox:
	
    def __init__(self,
                 init_state = [[1, 0, 0, 0],
                               [-0.5, 0.5, 0, 0],
                               [-0.5, -0.5, -0.5, 0.5]],
                 # size of the box [xmin, xmax, ymin, ymax]                 
                 bounds = [-100, 100, -100, 100],
                 size = 1,
                 maxvel = 5,
                 repulsionRange = 1,
                 orientationRange =8,
                 wallRange = .5,
                 wallRepulsion = 20,
                 k = .5):
        self.init_state = np.asarray(init_state, dtype=float)
        self.size = size
        self.repulsionRange = repulsionRange
        self.orientationRange = orientationRange
        self.wallRange=wallRange
        self.wallRepulsion = wallRepulsion
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.maxvel = maxvel
        self.k = k
        
		  

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        #U is the total force aplied on each particles
        #It has 3 components: a(atraction), o(orientation), r(repulsion)
        U = np.zeros((self.state.shape[0], 2), dtype=float)
        Ur = np.zeros((self.state.shape[0], 2), dtype=float)
        Uo = np.zeros((self.state.shape[0], 2), dtype=float)
        Ua = np.zeros((self.state.shape[0], 2), dtype=float)
        
        #calculate the distance betwen all the particles
        D = squareform(pdist(self.state[:, 0:2]))
        
        #Calculate U
        for i in range (self.state.shape[0]):
            #Sum of Velocities of neighbours
            velSum = np.zeros(2, dtype=float)
            #Sum of the distance of all particles
            distSum = np.zeros(2, dtype = float)
            
            for j in range (self.state.shape[0]):
                if (i != j):
                    #Ur calculation
                    if D[i][j] < self.repulsionRange:
                        Ur[i] += (self.state[j,:2] - self.state[i,:2])/ (D[i][j]**2)
                    #Uo sum
                    if D[i][j] < self.orientationRange:
                        velSum += (np.cos(self.state[j,2]),np.sin(self.state[j,2]))
                    #Ua sum
                    distSum += self.state[j,:2] - self.state[i,:2] 
            
            Uo[i] += ((np.cos(self.state[i,2]),np.sin(self.state[i,2])) + velSum )
            
            #Normaliza Uo and Ua            
            Uo[i] = Uo[i] / np.linalg.norm(Uo[i])
            Ua[i] = distSum / np.linalg.norm(distSum)
            Ur[i] *= -1
        
        #Check boudaries        
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size+self.wallRange)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size-self.wallRange)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size+self.wallRange)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size-self.wallRange)
        
        #Aplying forces to contain particles inside bounds (Using repulsion similar)
        for i in range (self.state.shape[0]):
            if (crossed_x1[i]):
                dif = (self.state[i,0] - self.bounds[0])
                if (dif < 0):
                    dif = .000001
                Ur[i] += (dif / (dif**2), 0)
        for i in range (self.state.shape[0]):
            if (crossed_x2[i]):            
                dif = (self.bounds[1] - self.state[i,0])
                if (dif < 0):
                    dif = .00001
                Ur[i] += (-dif / (dif**2), 0)
        for i in range (self.state.shape[0]):
            if (crossed_y1[i]):
                dif = (self.state[i,1] - self.bounds[2])
                if (dif < 0):
                    dif = .000001
                Ur[i] += (0 , dif / (dif**2))
        for i in range (self.state.shape[0]):
            if (crossed_y2[i]):            
                dif = (self.bounds[3] - self.state[i,1])
                if (dif < 0):
                    dif = .000001
                Ur[i] += (0, -dif / (dif**2))
        
        #Sum forces components  
        U[:] = Ur[:] + Uo[:] + Ua[:]
        
        #Heading calculation and fix
        headingError = np.arctan2(U[:,1],U[:,0]) - self.state[:,2]
        lowerAngleFix = (headingError[:] < -np.pi)
        higherAngleFix = (headingError[:] > np.pi)
        headingError[lowerAngleFix] = 2 * np.pi + headingError[lowerAngleFix]
        headingError[higherAngleFix] += -2 * np.pi
        
        #Update angular speed
        self.state[:,3] = self.k * headingError[:]              
        
        #Update positions
        self.state[:,2] += self.state[:,3] * dt        
        self.state[:,0] += self.maxvel * np.cos(self.state[:,2]) * dt
        self.state[:,1] += self.maxvel * np.sin(self.state[:,2]) * dt
        
                    
#------------------------------------------------------------
# set up initial state
np.random.seed(0)
init_state = -0.5 + np.random.random((50, 4))


init_state[:, 0:2] *= 75
init_state[:, 2] =  (init_state[:, 2]-0.5)* np.pi

init_state[:, 3] = 0

box = ParticleBox(init_state, size=0.04)
dt = 1. / 30 # 30fps


#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-125, 125), ylim=(-125, 125))

# particles holds the locations of the particles
particles, = ax.plot([], [], 'bo', ms=6)
# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax.add_patch(rect)

def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    rect.set_edgecolor('none')
    return particles, rect

def animate(i):
    """perform animation step"""
    global box, rect, dt, ax
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(2)
    return particles, rect

ani = animation.FuncAnimation(fig, animate, frames=30,
                              interval=10, blit=True, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('AltaCoesão.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()