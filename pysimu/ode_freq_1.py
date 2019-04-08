import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def freq_1(t,params):

    N = 2
    for it in range(N):

        # Parameters 
        T_g = params[it].T_g
        H = params[it].H
        D = params[it].D
        Droop = params[it].Droop
        p_l = params[it].p_l

        # States 
        Df   = params[it].x[0,0] 
        p_g  = params[it].x[1,0] 

        # Equations 
        dDf  = 1/(2*H)*(p_g - p_l - D*Df) 
        dp_g = 1/T_g*(-Df/Droop - p_g) 

        # Derivatives 
        params[it].f[0,0] = dDf    
        params[it].f[1,0] = dp_g   
dt = np.dtype([  
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
                ])

values = [(  
                0.1,   # T_g 
                 6.5,   # H 
                 1.0,   # D 
                 0.05,   # Droop 
                 0.05,   # p_l 
                 2,
                0,
                np.zeros((2,1)),
                np.zeros((2,1)),
                np.zeros((2,1)),
),  
(  
                0.6,   # T_g 
                 0.6,   # H 
                 0.6,   # D 
                 0.6,   # Droop 
                 0.6,   # p_l 
                 2,
                2,
                np.zeros((2,1)),
                np.zeros((2,1)),
                np.zeros((2,1)),
),  
]  
params= np.rec.array(values, dtype=dt)