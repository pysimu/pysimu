# -*- coding: utf-8 -*-
"""

Main pysimu module

Created on Thu Aug 14 20:21:56 2014


/home/jmmauricio/Documents/private/pyWork11/PyPsat/src



@author: jmmauricio-m
"""


import numpy as np
from scipy.integrate import ode

class sim:
    '''
    
    Class to perform simuations
    
    
    '''    
    

    def __init__(self):
        
        self.x = np.array([])
        self.t = 0.0
        
        self.T = np.array([])
        self.X = np.array([])
        self.Y = np.array([])

        self.max_step = 0.1
        self.nsteps = 5000

        
    def h(self,x):
        
        return x
    
    def odefun(self,t,x):
        self.x = x
        return self.f(t,x)
        
    def odeout(self,t,x):
        
        self.T = np.hstack((self.T,t))
        self.X = np.vstack((self.X,x))
        
        self.Y = np.vstack((self.Y,self.h(t,self.x)))
        
        return self.h(t,self.x)

    def run(self, t_end):
        
        r = ode(self.odefun)
        r.set_integrator('dopri5', max_step=self.max_step,  nsteps = self.nsteps)
        r.set_solout(self.odeout)
        if len(self.X)==0:
            self.X = self.x_0
            self.T = np.array(self.t)
            self.Y = np.array(self.h(self.t,self.x_0))
            
        r.set_initial_value(self.x_0, self.t)
        
        r.integrate(t_end)
        self.t = t_end
        self.r = r
        self.x = r.y



from numba import jit



    
@jit(cache=True)
def f(x):
 
    L = 0.1
    R = 0.1
    V = 1.0
    
    dx = 1.0/L*(V - R*x[0])
    
    return np.array([dx])

@jit(cache=True)
def forward_euler(x,Dt):     
    x[0] = x[0] + Dt*(f(x))
    return x        
        
        
@jit
def simulate():
    Dt = 0.0001
    x_0 = np.array([0.0])
    x= x_0
    t_end = 1.0
    decimation = 100  

    it_decimation =0
    
    t=0
    out = [[t,x[0]]]
    for t in np.arange(0.0,t_end,Dt):
        x = forward_euler(x,Dt)
        
        it_decimation += 1
        if it_decimation>decimation:
            it_decimation==0
            out += [[t,x[0]]]
    
     
    return np.array(out)
        
if __name__ == '__main__':
    
    import time
    
    t_0 = time.time()
    out = simulate()
    print(time.time()-t_0)
    
#    simu_rl = sim()
#    
#    # parameters
#    R = 1.0
#    L = 50.0e-3
#    v = 1.0
#
#    # dynamic system
#    def f(t,x):
#        
#        i = x[0]   
#        
#        di = 1.0/L*(v - R*i)
#        
#        return [di]
#        
#    # outputs functions
#    def h(t,x):
#    
#        i = x[0]
#       
#        p = i*v
#        
#        return np.array(p)
#        
#    # initialization
#    i_0 = 0
#    x_0 = np.array([i_0])
#    
#    # system definition
#    simu_rl.f = f
#    simu_rl.x_0 = x_0
#    simu_rl.h = h
#    simu_rl.run(1.0)
#
    # plot results
    import matplotlib.pyplot as plt
    fig = plt.figure( figsize=(6, 4))       
    ax = fig.add_subplot(1,1,1)    
    ax.plot(out[:,0],out[:,1], linewidth=2)    
    fig.show()
