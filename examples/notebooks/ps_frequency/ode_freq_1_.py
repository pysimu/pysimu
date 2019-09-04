import numpy as np
import numba



class freq_1_class: 
        def __init__(self): 

            dt = np.dtype([  
                    ('t_end', np.float64),
                    ('Dt', np.float64),
                    ('solver', np.int64),
                ('T_g', np.float64),
                ('H', np.float64),
                ('D', np.float64),
                ('Droop', np.float64),
                ('p_l', np.float64),
                    ('N_x', np.int64),
                    ('idx', np.int64),
                    ('f', np.float64, (2,1)),
                    ('x', np.float64, (2,1)),
                    ('x_0', np.float64, (2,1)),
                    ('h', np.float64, (1,1)),
                    ('Fx', np.float64, (2,2)),
                    ('T', np.float64, (2000,1)),
                    ('X', np.float64, (2000,2)),
                ])

            values = [(  
                20.000000,
                0.010000,
                1,
                0.1,   # T_g 
                 6.5,   # H 
                 1.0,   # D 
                 0.05,   # Droop 
                 0.6,   # p_l 
                 2,
                0,
                np.zeros((2,1)),
                np.zeros((2,1)),
                np.zeros((2,1)),
                np.zeros((1,1)),
                np.zeros((2,2)),
np.zeros((2000,1)),
np.zeros((2000,2)),
                ),  
                ]  
            self.struct = np.rec.array(values, dtype=dt)


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

    # states 
    Df   = struct[it].x[0,0] 
    p_g  = struct[it].x[1,0] 

    if mode==1: # initialization 

        p_g = p_l/(D*Droop + 1)
        Df = -Droop*p_l/(D*Droop + 1)
        struct[it].x[1,0] = p_g  
        struct[it].x[0,0] = Df  

    if mode==2: # derivatives 

        dDf  = 1/(2*H)*(p_g - p_l - D*Df) 
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
    N_steps = 2000 
    N_x = 2 
    N_outs = 1 
    T = np.zeros(N_steps) 
    X = np.zeros((N_steps,N_x)) 
    Y = np.zeros((N_steps,N_outs)) 
    # initialization 
    t = 0.0 
    freq_1(0.0,struct, 1) 
    T[0] = t 
    struct[i].X[0,:] = struct[i].x[:,0]  
    for it in range(N_steps-1): 
        t += Dt 
        perturbations(t,struct)
        # euler solver 
        freq_1(t,struct, 2) 
        struct[i].x[:] += Dt*struct[i].f 
        
        # channels 
        struct[i].T[it+1] = t 
        struct[i].X[it+1,:] = struct[i].x[:,0] 
        
    return struct[i].T[:], struct[i].X[:] 


@numba.njit(cache=True) 
def perturbations(t,struct): 
    struct[0].p_l = 0.0
    if t>1.0:
        struct[0].p_l = 0.1
        
    
    
if __name__ == "__main__":

    fr1 = freq_1_class()
    fr1.struct[0].p_l = 0.0 
    freq_1(0.0,fr1.struct,2)
    freq_1(0.0,fr1.struct,3)
    freq_1(0.0,fr1.struct,10)
    T,X = solver(fr1.struct)
    
    
    