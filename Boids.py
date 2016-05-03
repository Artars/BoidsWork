# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:19:36 2016

@author: arsart and buu
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math 

class ParticleBox:
	
    def __init__(self,
                 init_state = [[1, 0, 0, 0],
                               [-0.5, 0.5, 0, 0],
                               [-0.5, -0.5, -0.5, 0.5]],#3 particles [x0, y0, Vx, Vy,ax,ay]
                 bounds = [-20, 20, -20, 20], # size of the box [xmin, xmax, ymin, ymax]
                 size = 0.04,
                 M = 0.05,
                 maxvel = 5.,
                 separation =0.01,
                 wallRepulsion = 8.,
                 alignment = 0.05,
                 cohesion = .001,
                 visionRange =2.,
                 alignRange =5.,
                 wallRange = 3.,
                 cohesRange = 3):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])#return a new array of given shape filled with ones
        self.size = size
        self.separation = separation
        self.alignment = alignment
        self.cohesion = cohesion
        self.visionRange = visionRange
        self.wallRepulsion = wallRepulsion
        self.alignRange =alignRange
        self.cohesRange=cohesRange
        self.wallRange=wallRange
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.maxvel = maxvel
        
		  

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        dF = np.zeros((self.state.shape[0], 2), dtype=float)
        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, 0:2]))#calculate the distance betwen all the particles
        
        ind1, ind2 = np.where(D <  self.visionRange)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]
        
        #Calculate de Central o mass
        avarageCalc = self.state.copy()
        divisor = np.ones(avarageCalc.shape[0])
        for i in range(len(avarageCalc)):
            avarageCalc[i] *= self.M[i]
            divisor[i] *= self.M[i]
        

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):#make pairs of the data of ind1 with ind2            
            #sum the avarage
            avarageCalc[i1] += self.state[i2] * self.M[i2]
            avarageCalc[i2] += self.state[i1] * self.M[i1]
            divisor[i1] += self.M[i2]
            divisor[i2] += self.M[i1]

            # location vector
            r1x = self.state[i1,0]
            r1y = self.state[i1,1]
            r2x = self.state[i2,0]
            r2y = self.state[i2,1]


            # relative location & velocity vectors
            r_relx = r1x - r2x
            r_rely = r1y - r2y

            #adding the forces to the particles information vector  
            dF[i1]+=(self.separation*r_relx/(D[i1,i2]**3),self.separation*r_rely/(D[i1,i2]**3))
            dF[i2]-=(self.separation*r_relx/(D[i1,i2]**3),self.separation*r_relx/(D[i1,i2]**3))         
        
        #Finish calculating the avarage of mass
        for i in range(len(avarageCalc)):
            
            avarageCalc[i] = avarageCalc[i] / divisor[i]
         
        #Applying Coesion and Align forces
        alpha = np.ones(avarageCalc.shape[0])
        alpha[:]*=np.arctan2(self.state[:,3],self.state[:,2])-np.arctan2(avarageCalc[:,3],avarageCalc[:,2])#angulo do Vmedio para cada partícula
        maior=alpha[:]>math.pi
        menor=alpha[:]<-math.pi
        alpha[maior]=alpha[maior]-2*math.pi
        alpha[menor]=2*math.pi +alpha[menor]
        alpha[:] *= self.alignment
        dF[:] += self.cohesion * (avarageCalc[:,:2] - self.state[:,:2])
        
        # check for crossing boundary               
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size+self.wallRange)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size-self.wallRange)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size+self.wallRange)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size-self.wallRange)
        
        dF[crossed_x1,0] += (self.wallRepulsion )/(self.wallRange**2)
        dF[crossed_x2,0] -= (self.wallRepulsion )/(self.wallRange**2)
        
        dF[crossed_y1,1] += (self.wallRepulsion )/(self.wallRange**2)
        dF[crossed_y2,1] -= (self.wallRepulsion )/(self.wallRange**2)
        
        #making a new Mass vector with duplicate information to fit the x and y equations
        
        Mduplicate=np.array([self.M,]*2)
        Mduplicate.shape=(self.state.shape[0], 2)

        # update positions
        self.state[:, :2] += self.state[:,2:4]*dt+dF[:]/Mduplicate[:]*(dt**2)/2#S=So+Vo.t+a.t²/2
        
        # assign new velocities
        self.state[:,2:]+=(dF[:]/Mduplicate[:])*dt
        

        #Checking if velocity is up the max velocity
        CrossedVel= (self.state[:,2]**2 +self.state[:,3]**2)**0.5> self.maxvel        
        self.state[CrossedVel,2]/= (self.state[CrossedVel,2]**2 +self.state[CrossedVel,3]**2)**0.5/self.maxvel
        self.state[CrossedVel,3]/= (self.state[CrossedVel,2]**2 +self.state[CrossedVel,3]**2)**0.5/self.maxvel
        for i in range(len(self.state)):
            self.state[i,2:]=(np.cos(alpha[i])*self.state[i,2]+np.sin(alpha[i])*self.state[i,3],-np.sin(alpha[i])*self.state[i,2]+np.cos(alpha[i])*self.state[i,3])
#------------------------------------------------------------
# set up initial state
np.random.seed(0)
init_state = -0.5 + np.random.random((100, 4))


init_state[:, 0:2] *= 7
init_state[:, 2:] *=10

box = ParticleBox(init_state, size=0.04)
dt = 1. / 60 # 30fps


#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-22, 22), ylim=(-22, 22))

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
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    
    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    return particles, rect

ani = animation.FuncAnimation(fig, animate, frames=60,
                              interval=10, blit=True, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('AltaCoesão.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()