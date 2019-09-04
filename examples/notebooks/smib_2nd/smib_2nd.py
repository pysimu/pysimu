import numpy as np
import numba
from pysimu.nummath import interp


class smib_2nd_class: 
    def __init__(self): 

        self.t_end = 20.000000 
        self.Dt = 0.010000 
        self.decimation = 10.000000 
        self.itol = 0.000000 
        self.solvern = 1 
        self.imax = 100 
        self.N_x = 2 
        self.N_y = 9 
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
                ('X_d', np.float64),
                ('X1d', np.float64),
                ('T1d0', np.float64),
                ('X_q', np.float64),
                ('X1q', np.float64),
                ('T1q0', np.float64),
                ('R_a', np.float64),
                ('X_l', np.float64),
                ('H', np.float64),
                ('Omega_b', np.float64),
                ('B_t0', np.float64),
                ('G_t_inf', np.float64),
                ('T_r', np.float64),
                ('theta_inf', np.float64),
                ('T_pss_1', np.float64),
                ('T_pss_2', np.float64),
                ('T_w', np.float64),
                ('D', np.float64),
                ('K_a', np.float64),
                ('K_stab', np.float64),
                ('B_t_inf', np.float64),
                ('G_t0', np.float64),
                ('V_inf', np.float64),
                ('e1q', np.float64),
                ('e1d', np.float64),
                ('p_m', np.float64),
                ('V_ref', np.float64),
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
                1.81,   # X_d 
                 0.3,   # X1d 
                 8.0,   # T1d0 
                 1.76,   # X_q 
                 0.65,   # X1q 
                 1.0,   # T1q0 
                 0.003,   # R_a 
                 0.15,   # X_l 
                 3.5,   # H 
                 376.99111843077515,   # Omega_b 
                 0.0,   # B_t0 
                 0.0,   # G_t_inf 
                 0.05,   # T_r 
                 0.0,   # theta_inf 
                 1.281,   # T_pss_1 
                 0.013,   # T_pss_2 
                 5.0,   # T_w 
                 1.0,   # D 
                 200.0,   # K_a 
                 10,   # K_stab 
                 -2.10448859455482,   # B_t_inf 
                 0.01,   # G_t0 
                 0.9008,   # V_inf 
                 1.1,   # e1q 
                 0.0,   # e1d 
                 0.9,   # p_m 
                 1.0,   # V_ref 
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
        ini_struct(dt,values)

        dt +=     [('t', np.float64)]
        values += [0.0]

        dt +=     [('g', np.float64, (9,1))]
        values += [np.zeros((9,1))]
        dt +=     [('y', np.float64, (9,1))]
        values += [np.zeros((9,1))]
        dt +=     [('Fy', np.float64, (2,9))]
        values += [np.zeros((2,9))]
        dt +=     [('Gx', np.float64, (9,2))]
        values += [np.zeros((9,2))]
        dt +=     [('Gy', np.float64, (9,9))]
        values += [np.zeros((9,9))]




        self.struct = np.rec.array([tuple(values)], dtype=np.dtype(dt))


    def ini_problem(self,x):
        self.struct[0].x_ini[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        initialization(self.struct)
        fg = np.vstack((self.struct[0].f_ini,self.struct[0].g_ini))[:,0]
        return fg

    def run_problem(self,x):
        t = self.struct[0].t
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run(t,self.struct,2)
        run(t,self.struct,3)
        run(t,self.struct,10)
        run(t,self.struct,11)
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg
@numba.jit(nopython=True, cache=True)
def run(t,struct, mode):

    sin = np.sin
    cos = np.cos
    it = 0

    # parameters 
    X_d = struct[it].X_d
    X1d = struct[it].X1d
    T1d0 = struct[it].T1d0
    X_q = struct[it].X_q
    X1q = struct[it].X1q
    T1q0 = struct[it].T1q0
    R_a = struct[it].R_a
    X_l = struct[it].X_l
    H = struct[it].H
    Omega_b = struct[it].Omega_b
    B_t0 = struct[it].B_t0
    G_t_inf = struct[it].G_t_inf
    T_r = struct[it].T_r
    theta_inf = struct[it].theta_inf
    T_pss_1 = struct[it].T_pss_1
    T_pss_2 = struct[it].T_pss_2
    T_w = struct[it].T_w
    D = struct[it].D
    K_a = struct[it].K_a
    K_stab = struct[it].K_stab
    B_t_inf = struct[it].B_t_inf
    G_t0 = struct[it].G_t0
    V_inf = struct[it].V_inf
    e1q = struct[it].e1q
    e1d = struct[it].e1d

    # inputs 
    p_m = struct[it].p_m
    V_ref = struct[it].V_ref

    # states 
    delta = struct[it].x[0,0] 
    omega  = struct[it].x[1,0] 


    # algebraic states 
    i_d = struct[it].y[0,0] 
    i_q = struct[it].y[1,0] 
    p_e = struct[it].y[2,0] 
    v_d = struct[it].y[3,0] 
    v_q = struct[it].y[4,0] 
    P_t = struct[it].y[5,0] 
    Q_t = struct[it].y[6,0] 
    theta_t = struct[it].y[7,0] 
    V_t = struct[it].y[8,0] 


    if mode==2: # derivatives 

        ddelta=Omega_b*(omega - 1) 
        domega = -(p_e - p_m + D*(omega - 1))/(2*H) 

        struct[it].f[0,0] = ddelta  
        struct[it].f[1,0] = domega   

    if mode==3: # algebraic equations 

        struct[it].g[0,0] = R_a*i_q - e1q + i_d*(X1d - X_l) + v_q  
        struct[it].g[1,0] = R_a*i_d - e1d - i_q*(X1q - X_l) + v_d  
        struct[it].g[2,0] = -i_d*(R_a*i_d + v_d) - i_q*(R_a*i_q + v_q) + p_e  
        struct[it].g[3,0] = -V_t*sin(delta - theta_t) + v_d  
        struct[it].g[4,0] = -V_t*cos(delta - theta_t) + v_q  
        struct[it].g[5,0] = -P_t + i_d*v_d + i_q*v_q  
        struct[it].g[6,0] = -Q_t + i_d*v_q - i_q*v_d  
        struct[it].g[7,0] = -P_t - V_inf*V_t*(-B_t_inf*sin(theta_inf - theta_t) + G_t_inf*cos(theta_inf - theta_t)) + V_t**2*(G_t0 + G_t_inf)  
        struct[it].g[8,0] = -Q_t + V_inf*V_t*(B_t_inf*cos(theta_inf - theta_t) + G_t_inf*sin(theta_inf - theta_t)) + V_t**2*(-B_t0 - B_t_inf)  

    if mode==3: # outputs 

        struct[it].h[0,0] = omega  
    

    if mode==10: # Fx 

        struct[it].Fx[0,0] = 0 
        struct[it].Fx[0,1] = Omega_b 
        struct[it].Fx[1,0] = 0 
        struct[it].Fx[1,1] = -D/(2*H) 
    

    if mode==11: # Fy,Gx,Gy 

        struct[it].Fy[0,0] = 0 
        struct[it].Fy[0,1] = 0 
        struct[it].Fy[0,2] = 0 
        struct[it].Fy[0,3] = 0 
        struct[it].Fy[0,4] = 0 
        struct[it].Fy[0,5] = 0 
        struct[it].Fy[0,6] = 0 
        struct[it].Fy[0,7] = 0 
        struct[it].Fy[0,8] = 0 
        struct[it].Fy[1,0] = 0 
        struct[it].Fy[1,1] = 0 
        struct[it].Fy[1,2] = -1/(2*H) 
        struct[it].Fy[1,3] = 0 
        struct[it].Fy[1,4] = 0 
        struct[it].Fy[1,5] = 0 
        struct[it].Fy[1,6] = 0 
        struct[it].Fy[1,7] = 0 
        struct[it].Fy[1,8] = 0 
    

        struct[it].Gx[0,0] = 0 
        struct[it].Gx[0,1] = 0 
        struct[it].Gx[1,0] = 0 
        struct[it].Gx[1,1] = 0 
        struct[it].Gx[2,0] = 0 
        struct[it].Gx[2,1] = 0 
        struct[it].Gx[3,0] = -V_t*cos(delta - theta_t) 
        struct[it].Gx[3,1] = 0 
        struct[it].Gx[4,0] = V_t*sin(delta - theta_t) 
        struct[it].Gx[4,1] = 0 
        struct[it].Gx[5,0] = 0 
        struct[it].Gx[5,1] = 0 
        struct[it].Gx[6,0] = 0 
        struct[it].Gx[6,1] = 0 
        struct[it].Gx[7,0] = 0 
        struct[it].Gx[7,1] = 0 
        struct[it].Gx[8,0] = 0 
        struct[it].Gx[8,1] = 0 
    

        struct[it].Gy[0,0] = X1d - X_l 
        struct[it].Gy[0,1] = R_a 
        struct[it].Gy[0,2] = 0 
        struct[it].Gy[0,3] = 0 
        struct[it].Gy[0,4] = 1 
        struct[it].Gy[0,5] = 0 
        struct[it].Gy[0,6] = 0 
        struct[it].Gy[0,7] = 0 
        struct[it].Gy[0,8] = 0 
        struct[it].Gy[1,0] = R_a 
        struct[it].Gy[1,1] = -X1q + X_l 
        struct[it].Gy[1,2] = 0 
        struct[it].Gy[1,3] = 1 
        struct[it].Gy[1,4] = 0 
        struct[it].Gy[1,5] = 0 
        struct[it].Gy[1,6] = 0 
        struct[it].Gy[1,7] = 0 
        struct[it].Gy[1,8] = 0 
        struct[it].Gy[2,0] = -2*R_a*i_d - v_d 
        struct[it].Gy[2,1] = -2*R_a*i_q - v_q 
        struct[it].Gy[2,2] = 1 
        struct[it].Gy[2,3] = -i_d 
        struct[it].Gy[2,4] = -i_q 
        struct[it].Gy[2,5] = 0 
        struct[it].Gy[2,6] = 0 
        struct[it].Gy[2,7] = 0 
        struct[it].Gy[2,8] = 0 
        struct[it].Gy[3,0] = 0 
        struct[it].Gy[3,1] = 0 
        struct[it].Gy[3,2] = 0 
        struct[it].Gy[3,3] = 1 
        struct[it].Gy[3,4] = 0 
        struct[it].Gy[3,5] = 0 
        struct[it].Gy[3,6] = 0 
        struct[it].Gy[3,7] = V_t*cos(delta - theta_t) 
        struct[it].Gy[3,8] = -sin(delta - theta_t) 
        struct[it].Gy[4,0] = 0 
        struct[it].Gy[4,1] = 0 
        struct[it].Gy[4,2] = 0 
        struct[it].Gy[4,3] = 0 
        struct[it].Gy[4,4] = 1 
        struct[it].Gy[4,5] = 0 
        struct[it].Gy[4,6] = 0 
        struct[it].Gy[4,7] = -V_t*sin(delta - theta_t) 
        struct[it].Gy[4,8] = -cos(delta - theta_t) 
        struct[it].Gy[5,0] = v_d 
        struct[it].Gy[5,1] = v_q 
        struct[it].Gy[5,2] = 0 
        struct[it].Gy[5,3] = i_d 
        struct[it].Gy[5,4] = i_q 
        struct[it].Gy[5,5] = -1 
        struct[it].Gy[5,6] = 0 
        struct[it].Gy[5,7] = 0 
        struct[it].Gy[5,8] = 0 
        struct[it].Gy[6,0] = v_q 
        struct[it].Gy[6,1] = -v_d 
        struct[it].Gy[6,2] = 0 
        struct[it].Gy[6,3] = -i_q 
        struct[it].Gy[6,4] = i_d 
        struct[it].Gy[6,5] = 0 
        struct[it].Gy[6,6] = -1 
        struct[it].Gy[6,7] = 0 
        struct[it].Gy[6,8] = 0 
        struct[it].Gy[7,0] = 0 
        struct[it].Gy[7,1] = 0 
        struct[it].Gy[7,2] = 0 
        struct[it].Gy[7,3] = 0 
        struct[it].Gy[7,4] = 0 
        struct[it].Gy[7,5] = -1 
        struct[it].Gy[7,6] = 0 
        struct[it].Gy[7,7] = -V_inf*V_t*(B_t_inf*cos(theta_inf - theta_t) + G_t_inf*sin(theta_inf - theta_t)) 
        struct[it].Gy[7,8] = -V_inf*(-B_t_inf*sin(theta_inf - theta_t) + G_t_inf*cos(theta_inf - theta_t)) + 2*V_t*(G_t0 + G_t_inf) 
        struct[it].Gy[8,0] = 0 
        struct[it].Gy[8,1] = 0 
        struct[it].Gy[8,2] = 0 
        struct[it].Gy[8,3] = 0 
        struct[it].Gy[8,4] = 0 
        struct[it].Gy[8,5] = 0 
        struct[it].Gy[8,6] = -1 
        struct[it].Gy[8,7] = V_inf*V_t*(B_t_inf*sin(theta_inf - theta_t) - G_t_inf*cos(theta_inf - theta_t)) 
        struct[it].Gy[8,8] = V_inf*(B_t_inf*cos(theta_inf - theta_t) + G_t_inf*sin(theta_inf - theta_t)) + 2*V_t*(-B_t0 - B_t_inf) 


@numba.njit(cache=True)
def initialization(struct):

    sin = np.sin
    cos = np.cos
    X_d = struct[0].X_d 
    X1d = struct[0].X1d 
    T1d0 = struct[0].T1d0 
    X_q = struct[0].X_q 
    X1q = struct[0].X1q 
    T1q0 = struct[0].T1q0 
    R_a = struct[0].R_a 
    X_l = struct[0].X_l 
    H = struct[0].H 
    Omega_b = struct[0].Omega_b 
    B_t0 = struct[0].B_t0 
    G_t_inf = struct[0].G_t_inf 
    T_r = struct[0].T_r 
    theta_inf = struct[0].theta_inf 
    T_pss_1 = struct[0].T_pss_1 
    T_pss_2 = struct[0].T_pss_2 
    T_w = struct[0].T_w 
    D = struct[0].D 
    K_a = struct[0].K_a 
    K_stab = struct[0].K_stab 
    B_t_inf = struct[0].B_t_inf 
    G_t0 = struct[0].G_t0 
    V_inf = struct[0].V_inf 
    e1q = struct[0].e1q 
    e1d = struct[0].e1d 
    p_m = struct[0].p_m 
    V_ref = struct[0].V_ref 
    delta = struct[0].x_ini[0,0] 
    omega = struct[0].x_ini[1,0] 
    i_d = struct[0].y_ini[0,0] 
    i_q = struct[0].y_ini[1,0] 
    p_e = struct[0].y_ini[2,0] 
    v_d = struct[0].y_ini[3,0] 
    v_q = struct[0].y_ini[4,0] 
    P_t = struct[0].y_ini[5,0] 
    Q_t = struct[0].y_ini[6,0] 
    theta_t = struct[0].y_ini[7,0] 
    V_t = struct[0].y_ini[8,0] 
    struct[0].f_ini[0,0] = Omega_b*(omega - 1) 
    struct[0].f_ini[1,0] = (-D*(omega - 1) - p_e + p_m)/(2*H) 
    struct[0].g_ini[0,0]  = R_a*i_q - e1q + i_d*(X1d - X_l) + v_q  
    struct[0].g_ini[1,0]  = R_a*i_d - e1d - i_q*(X1q - X_l) + v_d  
    struct[0].g_ini[2,0]  = -i_d*(R_a*i_d + v_d) - i_q*(R_a*i_q + v_q) + p_e  
    struct[0].g_ini[3,0]  = -V_t*sin(delta - theta_t) + v_d  
    struct[0].g_ini[4,0]  = -V_t*cos(delta - theta_t) + v_q  
    struct[0].g_ini[5,0]  = -P_t + i_d*v_d + i_q*v_q  
    struct[0].g_ini[6,0]  = -Q_t + i_d*v_q - i_q*v_d  
    struct[0].g_ini[7,0]  = -P_t - V_inf*V_t*(-B_t_inf*sin(theta_inf - theta_t) + G_t_inf*cos(theta_inf - theta_t)) + V_t**2*(G_t0 + G_t_inf)  
    struct[0].g_ini[8,0]  = -Q_t + V_inf*V_t*(B_t_inf*cos(theta_inf - theta_t) + G_t_inf*sin(theta_inf - theta_t)) + V_t**2*(-B_t0 - B_t_inf)  
    struct[0].Fx_ini[0,0] = 0 
    struct[0].Fx_ini[0,1] = Omega_b 
    struct[0].Fx_ini[1,0] = 0 
    struct[0].Fx_ini[1,1] = -D/(2*H) 
    struct[0].Fy_ini[0,0] = 0 
    struct[0].Fy_ini[0,1] = 0 
    struct[0].Fy_ini[0,2] = 0 
    struct[0].Fy_ini[0,3] = 0 
    struct[0].Fy_ini[0,4] = 0 
    struct[0].Fy_ini[0,5] = 0 
    struct[0].Fy_ini[0,6] = 0 
    struct[0].Fy_ini[0,7] = 0 
    struct[0].Fy_ini[0,8] = 0 
    struct[0].Fy_ini[1,0] = 0 
    struct[0].Fy_ini[1,1] = 0 
    struct[0].Fy_ini[1,2] = -1/(2*H) 
    struct[0].Fy_ini[1,3] = 0 
    struct[0].Fy_ini[1,4] = 0 
    struct[0].Fy_ini[1,5] = 0 
    struct[0].Fy_ini[1,6] = 0 
    struct[0].Fy_ini[1,7] = 0 
    struct[0].Fy_ini[1,8] = 0 
    struct[0].Gx_ini[0,0] = 0 
    struct[0].Gx_ini[0,1] = 0 
    struct[0].Gx_ini[1,0] = 0 
    struct[0].Gx_ini[1,1] = 0 
    struct[0].Gx_ini[2,0] = 0 
    struct[0].Gx_ini[2,1] = 0 
    struct[0].Gx_ini[3,0] = -V_t*cos(delta - theta_t) 
    struct[0].Gx_ini[3,1] = 0 
    struct[0].Gx_ini[4,0] = V_t*sin(delta - theta_t) 
    struct[0].Gx_ini[4,1] = 0 
    struct[0].Gx_ini[5,0] = 0 
    struct[0].Gx_ini[5,1] = 0 
    struct[0].Gx_ini[6,0] = 0 
    struct[0].Gx_ini[6,1] = 0 
    struct[0].Gx_ini[7,0] = 0 
    struct[0].Gx_ini[7,1] = 0 
    struct[0].Gx_ini[8,0] = 0 
    struct[0].Gx_ini[8,1] = 0 
    struct[0].Gy_ini[0,0] = X1d - X_l 
    struct[0].Gy_ini[0,1] = R_a 
    struct[0].Gy_ini[0,2] = 0 
    struct[0].Gy_ini[0,3] = 0 
    struct[0].Gy_ini[0,4] = 1 
    struct[0].Gy_ini[0,5] = 0 
    struct[0].Gy_ini[0,6] = 0 
    struct[0].Gy_ini[0,7] = 0 
    struct[0].Gy_ini[0,8] = 0 
    struct[0].Gy_ini[1,0] = R_a 
    struct[0].Gy_ini[1,1] = -X1q + X_l 
    struct[0].Gy_ini[1,2] = 0 
    struct[0].Gy_ini[1,3] = 1 
    struct[0].Gy_ini[1,4] = 0 
    struct[0].Gy_ini[1,5] = 0 
    struct[0].Gy_ini[1,6] = 0 
    struct[0].Gy_ini[1,7] = 0 
    struct[0].Gy_ini[1,8] = 0 
    struct[0].Gy_ini[2,0] = -2*R_a*i_d - v_d 
    struct[0].Gy_ini[2,1] = -2*R_a*i_q - v_q 
    struct[0].Gy_ini[2,2] = 1 
    struct[0].Gy_ini[2,3] = -i_d 
    struct[0].Gy_ini[2,4] = -i_q 
    struct[0].Gy_ini[2,5] = 0 
    struct[0].Gy_ini[2,6] = 0 
    struct[0].Gy_ini[2,7] = 0 
    struct[0].Gy_ini[2,8] = 0 
    struct[0].Gy_ini[3,0] = 0 
    struct[0].Gy_ini[3,1] = 0 
    struct[0].Gy_ini[3,2] = 0 
    struct[0].Gy_ini[3,3] = 1 
    struct[0].Gy_ini[3,4] = 0 
    struct[0].Gy_ini[3,5] = 0 
    struct[0].Gy_ini[3,6] = 0 
    struct[0].Gy_ini[3,7] = V_t*cos(delta - theta_t) 
    struct[0].Gy_ini[3,8] = -sin(delta - theta_t) 
    struct[0].Gy_ini[4,0] = 0 
    struct[0].Gy_ini[4,1] = 0 
    struct[0].Gy_ini[4,2] = 0 
    struct[0].Gy_ini[4,3] = 0 
    struct[0].Gy_ini[4,4] = 1 
    struct[0].Gy_ini[4,5] = 0 
    struct[0].Gy_ini[4,6] = 0 
    struct[0].Gy_ini[4,7] = -V_t*sin(delta - theta_t) 
    struct[0].Gy_ini[4,8] = -cos(delta - theta_t) 
    struct[0].Gy_ini[5,0] = v_d 
    struct[0].Gy_ini[5,1] = v_q 
    struct[0].Gy_ini[5,2] = 0 
    struct[0].Gy_ini[5,3] = i_d 
    struct[0].Gy_ini[5,4] = i_q 
    struct[0].Gy_ini[5,5] = -1 
    struct[0].Gy_ini[5,6] = 0 
    struct[0].Gy_ini[5,7] = 0 
    struct[0].Gy_ini[5,8] = 0 
    struct[0].Gy_ini[6,0] = v_q 
    struct[0].Gy_ini[6,1] = -v_d 
    struct[0].Gy_ini[6,2] = 0 
    struct[0].Gy_ini[6,3] = -i_q 
    struct[0].Gy_ini[6,4] = i_d 
    struct[0].Gy_ini[6,5] = 0 
    struct[0].Gy_ini[6,6] = -1 
    struct[0].Gy_ini[6,7] = 0 
    struct[0].Gy_ini[6,8] = 0 
    struct[0].Gy_ini[7,0] = 0 
    struct[0].Gy_ini[7,1] = 0 
    struct[0].Gy_ini[7,2] = 0 
    struct[0].Gy_ini[7,3] = 0 
    struct[0].Gy_ini[7,4] = 0 
    struct[0].Gy_ini[7,5] = -1 
    struct[0].Gy_ini[7,6] = 0 
    struct[0].Gy_ini[7,7] = -V_inf*V_t*(B_t_inf*cos(theta_inf - theta_t) + G_t_inf*sin(theta_inf - theta_t)) 
    struct[0].Gy_ini[7,8] = -V_inf*(-B_t_inf*sin(theta_inf - theta_t) + G_t_inf*cos(theta_inf - theta_t)) + 2*V_t*(G_t0 + G_t_inf) 
    struct[0].Gy_ini[8,0] = 0 
    struct[0].Gy_ini[8,1] = 0 
    struct[0].Gy_ini[8,2] = 0 
    struct[0].Gy_ini[8,3] = 0 
    struct[0].Gy_ini[8,4] = 0 
    struct[0].Gy_ini[8,5] = 0 
    struct[0].Gy_ini[8,6] = -1 
    struct[0].Gy_ini[8,7] = V_inf*V_t*(B_t_inf*sin(theta_inf - theta_t) - G_t_inf*cos(theta_inf - theta_t)) 
    struct[0].Gy_ini[8,8] = V_inf*(B_t_inf*cos(theta_inf - theta_t) + G_t_inf*sin(theta_inf - theta_t)) + 2*V_t*(-B_t0 - B_t_inf) 


def ini_struct(dt,values):

    dt +=     [('x_ini', np.float64, (2,1))]
    values += [np.zeros((2,1))]
    dt +=     [('y_ini', np.float64, (9,1))]
    values += [np.zeros((9,1))]
    dt +=     [('f_ini', np.float64, (2,1))]
    values += [np.zeros((2,1))]
    dt +=     [('g_ini', np.float64, (9,1))]
    values += [np.zeros((9,1))]
    dt +=     [('Fx_ini', np.float64, (2,2))]
    values += [np.zeros((2,2))]
    dt +=     [('Fy_ini', np.float64, (2,9))]
    values += [np.zeros((2,9))]
    dt +=     [('Gx_ini', np.float64, (9,2))]
    values += [np.zeros((9,2))]
    dt +=     [('Gy_ini', np.float64, (9,9))]
    values += [np.zeros((9,9))]


@numba.njit(cache=True) 
def solver(struct): 
    sin = np.sin
    cos = np.cos
    i = 0 

    Dt = struct[i].Dt 
    N_steps = struct[i].N_steps 
    N_store = struct[i].N_store 
    N_x = 2 
    N_outs = 1 
    decimation = struct[i].decimation 
    # initialization 
    t = 0.0 
    smib_2nd(0.0,struct, 1) 
    it_store = 0 
    struct[i]['T'][0] = t 
    struct[i].X[0,:] = struct[i].x[:,0]  
    for it in range(N_steps-1): 
        t += Dt 
 
        perturbations(t,struct) 
        solver = struct[i].solvern 
        if solver == 1: 
            # forward euler solver  
            smib_2nd(t,struct, 2)  
            struct[i].x[:] += Dt*struct[i].f  
 
        if solver == 2: 
            # bacward euler solver
            x_0 = np.copy(struct[i].x[:]) 
            for j in range(struct[i].imax): 
                smib_2nd(t,struct, 2) 
                smib_2nd(t,struct, 10)  
                phi =  x_0 + Dt*struct[i].f - struct[i].x 
                Dx = np.linalg.solve(-(Dt*struct[i].Fx - np.eye(N_x)), phi) 
                struct[i].x[:] += Dx[:] 
                if np.max(np.abs(Dx)) < struct[i].itol: break 
 
        if solver == 3: 
            # trapezoidal solver
            smib_2nd(t,struct, 2) 
            f_0 = np.copy(struct[i].f[:]) 
            x_0 = np.copy(struct[i].x[:]) 
            for j in range(struct[i].imax): 
                smib_2nd(t,struct, 10)  
                phi =  x_0 + 0.5*Dt*(f_0 + struct[i].f) - struct[i].x 
                Dx = np.linalg.solve(-(0.5*Dt*struct[i].Fx - np.eye(N_x)), phi) 
                struct[i].x[:] += Dx[:] 
                smib_2nd(t,struct, 2) 
                if np.max(np.abs(Dx)) < struct[i].itol: break 
        
        # channels 
        if it >= it_store*decimation: 
          struct[i]['T'][it_store+1] = t 
          struct[i].X[it_store+1,:] = struct[i].x[:,0] 
          it_store += 1 
        
    return struct[i]['T'][:], struct[i].X[:] 


@numba.njit(cache=True) 
def perturbations(t,struct): 
    if t>1.000000: struct[0].V_ref = 1.010000



if __name__ == "__main__":
    sys = {'t_end': 20.0, 'Dt': 0.01, 'solver': 'forward-euler', 'decimation': 10, 'name': 'smib_2nd', 'models': [{'params': {'X_d': 1.81, 'X1d': 0.3, 'T1d0': 8.0, 'X_q': 1.76, 'X1q': 0.65, 'T1q0': 1.0, 'R_a': 0.003, 'X_l': 0.15, 'H': 3.5, 'Omega_b': 376.99111843077515, 'B_t0': 0.0, 'G_t_inf': 0.0, 'T_r': 0.05, 'theta_inf': 0.0, 'T_pss_1': 1.281, 'T_pss_2': 0.013, 'T_w': 5.0, 'D': 1.0, 'K_a': 200.0, 'K_stab': 10, 'B_t_inf': -2.10448859455482, 'G_t0': 0.01, 'V_inf': 0.9008, 'e1q': 1.1, 'e1d': 0.0}, 'f': ['ddelta=Omega_b*(omega - 1)', 'domega = -(p_e - p_m + D*(omega - 1))/(2*H)'], 'g': ['i_d@ v_q - e1q + R_a*i_q + i_d*(X1d - X_l)', 'i_q@ v_d - e1d + R_a*i_d - i_q*(X1q - X_l)', 'p_e@ p_e - i_d*(v_d + R_a*i_d) - i_q*(v_q + R_a*i_q) ', 'v_d@ v_d - V_t*sin(delta - theta_t)', 'v_q@ v_q - V_t*cos(delta - theta_t)', 'P_t@ i_d*v_d - P_t + i_q*v_q', 'Q_t@ i_d*v_q - Q_t - i_q*v_d', 'theta_t @(G_t0 + G_t_inf)*V_t**2 - V_inf*(G_t_inf*cos(theta_t - theta_inf) + B_t_inf*sin(theta_t - theta_inf))*V_t - P_t', 'V_t@ (- B_t0 - B_t_inf)*V_t**2 + V_inf*(B_t_inf*cos(theta_t - theta_inf) - G_t_inf*sin(theta_t - theta_inf))*V_t - Q_t'], 'u': {'p_m': 0.9, 'V_ref': 1.0}, 'u_ini': {}, 'y_ini': ['i_d', 'i_q', 'p_e', 'v_d', 'v_q', 'P_t', 'Q_t', 'theta_t', 'V_t'], 'h': ['omega']}], 'perturbations': [{'type': 'step', 'time': 1.0, 'var': 'V_ref', 'final': 1.01}], 'itol': 1e-08, 'imax': 100, 'solvern': 1}
    syst =  smib_2nd_class()
    T,X = solver(syst.struct)
    from scipy.optimize import fsolve
    x0 = np.ones(syst.N_x+syst.N_y)
    s = fsolve(syst.ini_problem,x0 )
    print(s)