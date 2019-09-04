import numpy as np
import numba
from pysimu.nummath import interp


class freq_1_class: 
    def __init__(self): 

        self.t_end = 20.000000 
        self.Dt = 0.500000 
        self.decimation = 1.000000 
        self.itol = 0.010000 
        self.solvern = 2 
        self.imax = 100 
        self.update() 

    def update(self): 

        self.N_steps = int(np.ceil(self.t_end/self.Dt)) 
        self.N_store = int(np.ceil(self.N_steps/self.decimation))
        dt =  [  
        ('t_end', np.float64),
        ('Dt', np.float64),
        ('decimation', np.float64),
        ('itol', np.float64),
        ('solvern', np.int64),
        ('imax', np.int64),
                    ('N_steps', np.int64),
                    ('N_store', np.int64),
                ('T_g', np.float64),
                ('H', np.float64),
                ('D', np.float64),
                ('Droop', np.float64),
                ('p_l', np.float64),
                ('p_nc', np.float64),
                    ('N_x', np.int64),
                    ('idx', np.int64),
                    ('f', np.float64, (2,1)),
                    ('x', np.float64, (2,1)),
                    ('x_0', np.float64, (2,1)),
                    ('h', np.float64, (1,1)),
                    ('Fx', np.float64, (2,2)),
                    ('T', np.float64, (self.N_store+1,1)),
                    ('X', np.float64, (self.N_store+1,2)),
                ]  

        values = [
                self.t_end,
                self.Dt,
                self.decimation,
                self.itol,
                self.solvern,
                self.imax,
                self.N_steps,
                self.N_store,
                5.0,   # T_g 
                 6.5,   # H 
                 1.0,   # D 
                 0.05,   # Droop 
                 0.0,   # p_l 
                 0.0,   # p_nc 
                 2,
                0,
                np.zeros((2,1)),
                np.zeros((2,1)),
                np.zeros((2,1)),
                np.zeros((1,1)),
                                np.zeros((2,2)),
                np.zeros((self.N_store+1,1)),
                np.zeros((self.N_store+1,2)),
                ]  




        self.struct = np.rec.array([tuple(values)], dtype=np.dtype(dt))


@numba.jit(nopython=True, cache=True)
def freq_1(t,struct, mode):

    it = 0

    # parameters 
    T_g = struct[it].T_g
    H = struct[it].H
    D = struct[it].D
    Droop = struct[it].Droop

    # inputs 
    p_l = struct[it].p_l
    p_nc = struct[it].p_nc

    # states 
    Df   = struct[it].x[0,0] 
    p_g  = struct[it].x[1,0] 

    if mode==1: # initialization 

        p_g = (p_l - p_nc)/(D*Droop + 1)
        Df = -Droop*(p_l - p_nc)/(D*Droop + 1)
        struct[it].x[1,0] = p_g  
        struct[it].x[0,0] = Df  

    if mode==2: # derivatives 

        dDf  = 1/(2*H)*(p_g + p_nc - p_l - D*Df) 
        dp_g = 1/T_g*(-Df/Droop - p_g) 

        struct[it].f[0,0] = dDf    
        struct[it].f[1,0] = dp_g   

    if mode==3: # outputs 

        struct[it].h[0,0] = Df  
    

    if mode==10: # Fx 

        struct[it].Fx[0,0] = -D/(2*H) 
        struct[it].Fx[0,1] = 1/(2*H) 
        struct[it].Fx[1,0] = -1/(Droop*T_g) 
        struct[it].Fx[1,1] = -1/T_g 


@numba.njit(cache=True) 
def solver(struct): 
    i = 0 

    Dt = struct[i].Dt 
    N_steps = struct[i].N_steps 
    N_store = struct[i].N_store 
    N_x = 2 
    N_outs = 1 
    decimation = struct[i].decimation 
    # initialization 
    t = 0.0 
    freq_1(0.0,struct, 1) 
    it_store = 0 
    struct[i]['T'][0] = t 
    struct[i].X[0,:] = struct[i].x[:,0]  
    for it in range(N_steps-1): 
        t += Dt 
 
        perturbations(t,struct) 
        solver = struct[i].solvern 
        if solver == 1: 
            # forward euler solver  
            freq_1(t,struct, 2)  
            struct[i].x[:] += Dt*struct[i].f  
 
        if solver == 2: 
            # bacward euler solver
            x_0 = np.copy(struct[i].x[:]) 
            for j in range(struct[i].imax): 
                freq_1(t,struct, 2) 
                freq_1(t,struct, 10)  
                phi =  x_0 + Dt*struct[i].f - struct[i].x 
                Dx = np.linalg.solve(-(Dt*struct[i].Fx - np.eye(N_x)), phi) 
                struct[i].x[:] += Dx[:] 
                if np.max(np.abs(Dx)) < struct[i].itol: break 
 
        if solver == 3: 
            # trapezoidal solver
            freq_1(t,struct, 2) 
            f_0 = np.copy(struct[i].f[:]) 
            x_0 = np.copy(struct[i].x[:]) 
            for j in range(struct[i].imax): 
                freq_1(t,struct, 10)  
                phi =  x_0 + 0.5*Dt*(f_0 + struct[i].f) - struct[i].x 
                Dx = np.linalg.solve(-(0.5*Dt*struct[i].Fx - np.eye(N_x)), phi) 
                struct[i].x[:] += Dx[:] 
                freq_1(t,struct, 2) 
                if np.max(np.abs(Dx)) < struct[i].itol: break 
        
        # channels 
        if it >= it_store*decimation: 
          struct[i]['T'][it_store+1] = t 
          struct[i].X[it_store+1,:] = struct[i].x[:,0] 
          it_store += 1 
        
    return struct[i]['T'][:], struct[i].X[:] 


@numba.njit(cache=True) 
def perturbations(t,struct): 
    if t>1.000000: struct[0].p_l = 1.000000
    if t>2.000000: struct[0].p_nc = 0.500000



if __name__ == "__main__":
    sys = {'name': 'freq_1', 't_end': 20.0, 'Dt': 0.5, 'solver': 'backward-euler', 'decimation': 1, 'itol': 0.01, 'models': [{'params': {'T_g': 5.0, 'H': 6.5, 'D': 1.0, 'Droop': 0.05}, 'f': ['dDf  = 1/(2*H)*(p_g + p_nc - p_l - D*Df)', 'dp_g = 1/T_g*(-Df/Droop - p_g)'], 'u': {'p_l': 0.0, 'p_nc': 0.0}, 'h': ['Df']}], 'perturbations': [{'type': 'step', 'time': 1.0, 'var': 'p_l', 'final': 1.0}, {'type': 'step', 'time': 2.0, 'var': 'p_nc', 'final': 0.5}], 'imax': 100, 'solvern': 2}
    syst =  freq_1_class()
    T,X = solver(syst.struct)
    import matplotlib.pyplot as plt
    fig, (ax0,ax1) = plt.subplots(nrows=2)   # creates a figure with one axe
    ax0.plot(T,X[:,0])
    ax1.plot(T,X[:,1])
    ax0.set_ylabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_xlabel('Time (s)')
    plt.show()