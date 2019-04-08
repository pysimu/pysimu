import numpy as np
import numba
from pysimu.nummath import interp


class smib_4rd_avr_pss_rocof_1_class: 
    def __init__(self): 

        self.t_end = 20.000000 
        self.Dt = 0.010000 
        self.decimation = 10.000000 
        self.itol = 0.000000 
        self.solvern = 1 
        self.imax = 100 
        self.N_x = 9 
        self.N_y = 14 
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
                ('T_pss_1', np.float64),
                ('T_pss_2', np.float64),
                ('T_w', np.float64),
                ('D', np.float64),
                ('K_a', np.float64),
                ('K_stab', np.float64),
                ('B_t_inf', np.float64),
                ('G_t0', np.float64),
                ('V_inf', np.float64),
                ('p_m', np.float64),
                ('V_ref', np.float64),
                ('RoCoFpu', np.float64),
                    ('N_x', np.int64),
                    ('idx', np.int64),
                    ('f', np.float64, (9,1)),
                    ('x', np.float64, (9,1)),
                    ('x_0', np.float64, (9,1)),
                    ('h', np.float64, (1,1)),
                    ('Fx', np.float64, (9,9)),
                    ('T', np.float64, (self.N_store+1,1)),
                    ('X', np.float64, (self.N_store+1,9)),
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
                 1.281,   # T_pss_1 
                 0.013,   # T_pss_2 
                 5.0,   # T_w 
                 0.0,   # D 
                 200.0,   # K_a 
                 10,   # K_stab 
                 -2.10448859455482,   # B_t_inf 
                 0.01,   # G_t0 
                 0.9008,   # V_inf 
                 0.9,   # p_m 
                 1.0,   # V_ref 
                 0.0,   # RoCoFpu 
                 9,
                0,
                np.zeros((9,1)),
                np.zeros((9,1)),
                np.zeros((9,1)),
                np.zeros((1,1)),
                                np.zeros((9,9)),
                np.zeros((self.N_store+1,1)),
                np.zeros((self.N_store+1,9)),
                ]  
        ini_struct(dt,values)

        dt +=     [('t', np.float64)]
        values += [0.0]

        dt +=     [('N_y', np.int64)]
        values += [self.N_y]

        dt +=     [('g', np.float64, (14,1))]
        values += [np.zeros((14,1))]
        dt +=     [('y', np.float64, (14,1))]
        values += [np.zeros((14,1))]
        dt +=     [('Fy', np.float64, (9,14))]
        values += [np.zeros((9,14))]
        dt +=     [('Gx', np.float64, (14,9))]
        values += [np.zeros((14,9))]
        dt +=     [('Gy', np.float64, (14,14))]
        values += [np.zeros((14,14))]




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
    sqrt = np.sqrt
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
    T_pss_1 = struct[it].T_pss_1
    T_pss_2 = struct[it].T_pss_2
    T_w = struct[it].T_w
    D = struct[it].D
    K_a = struct[it].K_a
    K_stab = struct[it].K_stab
    B_t_inf = struct[it].B_t_inf
    G_t0 = struct[it].G_t0
    V_inf = struct[it].V_inf

    # inputs 
    p_m = struct[it].p_m
    V_ref = struct[it].V_ref
    RoCoFpu = struct[it].RoCoFpu

    # states 
    delta = struct[it].x[0,0] 
    omega  = struct[it].x[1,0] 
    e1q  = struct[it].x[2,0] 
    e1d  = struct[it].x[3,0] 
    v_c  = struct[it].x[4,0] 
    x_pss  = struct[it].x[5,0] 
    x_w  = struct[it].x[6,0] 
    phi_inf  = struct[it].x[7,0] 
    omega_inf  = struct[it].x[8,0] 


    # algebraic states 
    p_h = struct[it].y[0,0] 
    p_m0 = struct[it].y[1,0] 
    i_d = struct[it].y[2,0] 
    i_q = struct[it].y[3,0] 
    p_e = struct[it].y[4,0] 
    v_d = struct[it].y[5,0] 
    v_q = struct[it].y[6,0] 
    P_t = struct[it].y[7,0] 
    Q_t = struct[it].y[8,0] 
    theta_t = struct[it].y[9,0] 
    V_t = struct[it].y[10,0] 
    v_f = struct[it].y[11,0] 
    v_pss = struct[it].y[12,0] 
    omega_w = struct[it].y[13,0] 


    if mode==2: # derivatives 

        ddelta=Omega_b*(omega - 1) 
        domega = -(p_e - p_m + D*(omega - 1))/(2*H) 
        de1q =(v_f - e1q + i_d*(X1d - X_d))/T1d0 
        de1d = -(e1d + i_q*(X1q - X_q))/T1q0 
        dv_c =   (V_t - v_c)/T_r 
        dx_pss = (omega_w - x_pss)/T_pss_2 
        dx_w =  (omega - x_w)/T_w 
        dphi_inf = Omega_b*(omega_inf - 1) -1e-6*phi_inf 
        domega_inf = RoCoFpu - 0.0001*(omega_inf - 1) 

        struct[it].f[0,0] = ddelta  
        struct[it].f[1,0] = domega   
        struct[it].f[2,0] = de1q   
        struct[it].f[3,0] = de1d   
        struct[it].f[4,0] = dv_c   
        struct[it].f[5,0] = dx_pss   
        struct[it].f[6,0] = dx_w   
        struct[it].f[7,0] = dphi_inf   
        struct[it].f[8,0] = domega_inf   

    if mode==3: # algebraic equations 

        struct[it].g[0,0] = 2*H*RoCoFpu + p_h  
        struct[it].g[1,0] = -p_m + p_m0  
        struct[it].g[2,0] = R_a*i_q - e1q + i_d*(X1d - X_l) + v_q  
        struct[it].g[3,0] = R_a*i_d - e1d - i_q*(X1q - X_l) + v_d  
        struct[it].g[4,0] = -i_d*(R_a*i_d + v_d) - i_q*(R_a*i_q + v_q) + p_e  
        struct[it].g[5,0] = -V_t*sin(delta - theta_t) + v_d  
        struct[it].g[6,0] = -V_t*cos(delta - theta_t) + v_q  
        struct[it].g[7,0] = -P_t + i_d*v_d + i_q*v_q  
        struct[it].g[8,0] = -Q_t + i_d*v_q - i_q*v_d  
        struct[it].g[9,0] = -P_t - V_inf*V_t*(-B_t_inf*sin(phi_inf - theta_t) + G_t_inf*cos(phi_inf - theta_t)) + V_t**2*(G_t0 + G_t_inf)  
        struct[it].g[10,0] = -Q_t + V_inf*V_t*(B_t_inf*cos(phi_inf - theta_t) + G_t_inf*sin(phi_inf - theta_t)) + V_t**2*(-B_t0 - B_t_inf)  
        struct[it].g[11,0] = K_a*(V_ref - v_c + v_pss) - v_f  
        struct[it].g[12,0] = K_stab*(-T_pss_1*omega_w/T_pss_2 + x_pss*(-1 + 1/T_pss_2)) + v_pss  
        struct[it].g[13,0] = -omega + omega_w + x_w  

    if mode==4: # outputs 

        struct[it].h[0,0] = omega  
    

    if mode==10: # Fx 

        struct[it].Fx[0,1] = Omega_b 
        struct[it].Fx[1,1] = -D/(2*H) 
        struct[it].Fx[2,2] = -1/T1d0 
        struct[it].Fx[3,3] = -1/T1q0 
        struct[it].Fx[4,4] = -1/T_r 
        struct[it].Fx[5,5] = -1/T_pss_2 
        struct[it].Fx[6,1] = 1/T_w 
        struct[it].Fx[6,6] = -1/T_w 
        struct[it].Fx[7,7] = -1.00000000000000e-6 
        struct[it].Fx[7,8] = Omega_b 
        struct[it].Fx[8,8] = -0.000100000000000000 
    

    if mode==11: # Fy,Gx,Gy 

        struct[it].Fy[1,4] = -1/(2*H) 
        struct[it].Fy[2,2] = (X1d - X_d)/T1d0 
        struct[it].Fy[2,11] = 1/T1d0 
        struct[it].Fy[3,3] = (-X1q + X_q)/T1q0 
        struct[it].Fy[4,10] = 1/T_r 
        struct[it].Fy[5,13] = 1/T_pss_2 
    

        struct[it].Gx[2,2] = -1 
        struct[it].Gx[3,3] = -1 
        struct[it].Gx[5,0] = -V_t*cos(delta - theta_t) 
        struct[it].Gx[6,0] = V_t*sin(delta - theta_t) 
        struct[it].Gx[9,7] = -V_inf*V_t*(-B_t_inf*cos(phi_inf - theta_t) - G_t_inf*sin(phi_inf - theta_t)) 
        struct[it].Gx[10,7] = V_inf*V_t*(-B_t_inf*sin(phi_inf - theta_t) + G_t_inf*cos(phi_inf - theta_t)) 
        struct[it].Gx[11,4] = -K_a 
        struct[it].Gx[12,5] = K_stab*(-1 + 1/T_pss_2) 
        struct[it].Gx[13,1] = -1 
        struct[it].Gx[13,6] = 1 
    

        struct[it].Gy[0,0] = 1 
        struct[it].Gy[1,1] = 1 
        struct[it].Gy[2,2] = X1d - X_l 
        struct[it].Gy[2,3] = R_a 
        struct[it].Gy[2,6] = 1 
        struct[it].Gy[3,2] = R_a 
        struct[it].Gy[3,3] = -X1q + X_l 
        struct[it].Gy[3,5] = 1 
        struct[it].Gy[4,2] = -2*R_a*i_d - v_d 
        struct[it].Gy[4,3] = -2*R_a*i_q - v_q 
        struct[it].Gy[4,4] = 1 
        struct[it].Gy[4,5] = -i_d 
        struct[it].Gy[4,6] = -i_q 
        struct[it].Gy[5,5] = 1 
        struct[it].Gy[5,9] = V_t*cos(delta - theta_t) 
        struct[it].Gy[5,10] = -sin(delta - theta_t) 
        struct[it].Gy[6,6] = 1 
        struct[it].Gy[6,9] = -V_t*sin(delta - theta_t) 
        struct[it].Gy[6,10] = -cos(delta - theta_t) 
        struct[it].Gy[7,2] = v_d 
        struct[it].Gy[7,3] = v_q 
        struct[it].Gy[7,5] = i_d 
        struct[it].Gy[7,6] = i_q 
        struct[it].Gy[7,7] = -1 
        struct[it].Gy[8,2] = v_q 
        struct[it].Gy[8,3] = -v_d 
        struct[it].Gy[8,5] = -i_q 
        struct[it].Gy[8,6] = i_d 
        struct[it].Gy[8,8] = -1 
        struct[it].Gy[9,7] = -1 
        struct[it].Gy[9,9] = -V_inf*V_t*(B_t_inf*cos(phi_inf - theta_t) + G_t_inf*sin(phi_inf - theta_t)) 
        struct[it].Gy[9,10] = -V_inf*(-B_t_inf*sin(phi_inf - theta_t) + G_t_inf*cos(phi_inf - theta_t)) + 2*V_t*(G_t0 + G_t_inf) 
        struct[it].Gy[10,8] = -1 
        struct[it].Gy[10,9] = V_inf*V_t*(B_t_inf*sin(phi_inf - theta_t) - G_t_inf*cos(phi_inf - theta_t)) 
        struct[it].Gy[10,10] = V_inf*(B_t_inf*cos(phi_inf - theta_t) + G_t_inf*sin(phi_inf - theta_t)) + 2*V_t*(-B_t0 - B_t_inf) 
        struct[it].Gy[11,11] = -1 
        struct[it].Gy[11,12] = K_a 
        struct[it].Gy[12,12] = 1 
        struct[it].Gy[12,13] = -K_stab*T_pss_1/T_pss_2 
        struct[it].Gy[13,13] = 1 


@numba.njit(cache=True)
def initialization(struct):

    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
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
    T_pss_1 = struct[0].T_pss_1 
    T_pss_2 = struct[0].T_pss_2 
    T_w = struct[0].T_w 
    D = struct[0].D 
    K_a = struct[0].K_a 
    K_stab = struct[0].K_stab 
    B_t_inf = struct[0].B_t_inf 
    G_t0 = struct[0].G_t0 
    V_inf = struct[0].V_inf 
    p_m = struct[0].p_m 
    V_ref = struct[0].V_ref 
    RoCoFpu = struct[0].RoCoFpu 
    delta = struct[0].x_ini[0,0] 
    omega = struct[0].x_ini[1,0] 
    e1q = struct[0].x_ini[2,0] 
    e1d = struct[0].x_ini[3,0] 
    v_c = struct[0].x_ini[4,0] 
    x_pss = struct[0].x_ini[5,0] 
    x_w = struct[0].x_ini[6,0] 
    phi_inf = struct[0].x_ini[7,0] 
    omega_inf = struct[0].x_ini[8,0] 
    p_h = struct[0].y_ini[0,0] 
    p_m0 = struct[0].y_ini[1,0] 
    i_d = struct[0].y_ini[2,0] 
    i_q = struct[0].y_ini[3,0] 
    p_e = struct[0].y_ini[4,0] 
    v_d = struct[0].y_ini[5,0] 
    v_q = struct[0].y_ini[6,0] 
    P_t = struct[0].y_ini[7,0] 
    Q_t = struct[0].y_ini[8,0] 
    theta_t = struct[0].y_ini[9,0] 
    V_t = struct[0].y_ini[10,0] 
    v_f = struct[0].y_ini[11,0] 
    v_pss = struct[0].y_ini[12,0] 
    omega_w = struct[0].y_ini[13,0] 
    struct[0].f_ini[0,0] = Omega_b*(omega - 1) 
    struct[0].f_ini[1,0] = (-D*(omega - 1) - p_e + p_m)/(2*H) 
    struct[0].f_ini[2,0] = (-e1q + i_d*(X1d - X_d) + v_f)/T1d0 
    struct[0].f_ini[3,0] = (-e1d - i_q*(X1q - X_q))/T1q0 
    struct[0].f_ini[4,0] = (V_t - v_c)/T_r 
    struct[0].f_ini[5,0] = (omega_w - x_pss)/T_pss_2 
    struct[0].f_ini[6,0] = (omega - x_w)/T_w 
    struct[0].f_ini[7,0] = Omega_b*(omega_inf - 1) - 1.0e-6*phi_inf 
    struct[0].f_ini[8,0] = RoCoFpu - 0.0001*omega_inf + 0.0001 
    struct[0].g_ini[0,0]  = 2*H*RoCoFpu + p_h  
    struct[0].g_ini[1,0]  = -p_m + p_m0  
    struct[0].g_ini[2,0]  = R_a*i_q - e1q + i_d*(X1d - X_l) + v_q  
    struct[0].g_ini[3,0]  = R_a*i_d - e1d - i_q*(X1q - X_l) + v_d  
    struct[0].g_ini[4,0]  = -i_d*(R_a*i_d + v_d) - i_q*(R_a*i_q + v_q) + p_e  
    struct[0].g_ini[5,0]  = -V_t*sin(delta - theta_t) + v_d  
    struct[0].g_ini[6,0]  = -V_t*cos(delta - theta_t) + v_q  
    struct[0].g_ini[7,0]  = -P_t + i_d*v_d + i_q*v_q  
    struct[0].g_ini[8,0]  = -Q_t + i_d*v_q - i_q*v_d  
    struct[0].g_ini[9,0]  = -P_t - V_inf*V_t*(-B_t_inf*sin(phi_inf - theta_t) + G_t_inf*cos(phi_inf - theta_t)) + V_t**2*(G_t0 + G_t_inf)  
    struct[0].g_ini[10,0]  = -Q_t + V_inf*V_t*(B_t_inf*cos(phi_inf - theta_t) + G_t_inf*sin(phi_inf - theta_t)) + V_t**2*(-B_t0 - B_t_inf)  
    struct[0].g_ini[11,0]  = K_a*(V_ref - v_c + v_pss) - v_f  
    struct[0].g_ini[12,0]  = K_stab*(-T_pss_1*omega_w/T_pss_2 + x_pss*(-1 + 1/T_pss_2)) + v_pss  
    struct[0].g_ini[13,0]  = -omega + omega_w + x_w  
    struct[0].Fx_ini[0,1] = Omega_b 
    struct[0].Fx_ini[1,1] = -D/(2*H) 
    struct[0].Fx_ini[2,2] = -1/T1d0 
    struct[0].Fx_ini[3,3] = -1/T1q0 
    struct[0].Fx_ini[4,4] = -1/T_r 
    struct[0].Fx_ini[5,5] = -1/T_pss_2 
    struct[0].Fx_ini[6,1] = 1/T_w 
    struct[0].Fx_ini[6,6] = -1/T_w 
    struct[0].Fx_ini[7,7] = -1.00000000000000e-6 
    struct[0].Fx_ini[7,8] = Omega_b 
    struct[0].Fx_ini[8,8] = -0.000100000000000000 
    struct[0].Fy_ini[1,4] = -1/(2*H) 
    struct[0].Fy_ini[2,2] = (X1d - X_d)/T1d0 
    struct[0].Fy_ini[2,11] = 1/T1d0 
    struct[0].Fy_ini[3,3] = (-X1q + X_q)/T1q0 
    struct[0].Fy_ini[4,10] = 1/T_r 
    struct[0].Fy_ini[5,13] = 1/T_pss_2 
    struct[0].Gx_ini[2,2] = -1 
    struct[0].Gx_ini[3,3] = -1 
    struct[0].Gx_ini[5,0] = -V_t*cos(delta - theta_t) 
    struct[0].Gx_ini[6,0] = V_t*sin(delta - theta_t) 
    struct[0].Gx_ini[9,7] = -V_inf*V_t*(-B_t_inf*cos(phi_inf - theta_t) - G_t_inf*sin(phi_inf - theta_t)) 
    struct[0].Gx_ini[10,7] = V_inf*V_t*(-B_t_inf*sin(phi_inf - theta_t) + G_t_inf*cos(phi_inf - theta_t)) 
    struct[0].Gx_ini[11,4] = -K_a 
    struct[0].Gx_ini[12,5] = K_stab*(-1 + 1/T_pss_2) 
    struct[0].Gx_ini[13,1] = -1 
    struct[0].Gx_ini[13,6] = 1 
    struct[0].Gy_ini[0,0] = 1 
    struct[0].Gy_ini[1,1] = 1 
    struct[0].Gy_ini[2,2] = X1d - X_l 
    struct[0].Gy_ini[2,3] = R_a 
    struct[0].Gy_ini[2,6] = 1 
    struct[0].Gy_ini[3,2] = R_a 
    struct[0].Gy_ini[3,3] = -X1q + X_l 
    struct[0].Gy_ini[3,5] = 1 
    struct[0].Gy_ini[4,2] = -2*R_a*i_d - v_d 
    struct[0].Gy_ini[4,3] = -2*R_a*i_q - v_q 
    struct[0].Gy_ini[4,4] = 1 
    struct[0].Gy_ini[4,5] = -i_d 
    struct[0].Gy_ini[4,6] = -i_q 
    struct[0].Gy_ini[5,5] = 1 
    struct[0].Gy_ini[5,9] = V_t*cos(delta - theta_t) 
    struct[0].Gy_ini[5,10] = -sin(delta - theta_t) 
    struct[0].Gy_ini[6,6] = 1 
    struct[0].Gy_ini[6,9] = -V_t*sin(delta - theta_t) 
    struct[0].Gy_ini[6,10] = -cos(delta - theta_t) 
    struct[0].Gy_ini[7,2] = v_d 
    struct[0].Gy_ini[7,3] = v_q 
    struct[0].Gy_ini[7,5] = i_d 
    struct[0].Gy_ini[7,6] = i_q 
    struct[0].Gy_ini[7,7] = -1 
    struct[0].Gy_ini[8,2] = v_q 
    struct[0].Gy_ini[8,3] = -v_d 
    struct[0].Gy_ini[8,5] = -i_q 
    struct[0].Gy_ini[8,6] = i_d 
    struct[0].Gy_ini[8,8] = -1 
    struct[0].Gy_ini[9,7] = -1 
    struct[0].Gy_ini[9,9] = -V_inf*V_t*(B_t_inf*cos(phi_inf - theta_t) + G_t_inf*sin(phi_inf - theta_t)) 
    struct[0].Gy_ini[9,10] = -V_inf*(-B_t_inf*sin(phi_inf - theta_t) + G_t_inf*cos(phi_inf - theta_t)) + 2*V_t*(G_t0 + G_t_inf) 
    struct[0].Gy_ini[10,8] = -1 
    struct[0].Gy_ini[10,9] = V_inf*V_t*(B_t_inf*sin(phi_inf - theta_t) - G_t_inf*cos(phi_inf - theta_t)) 
    struct[0].Gy_ini[10,10] = V_inf*(B_t_inf*cos(phi_inf - theta_t) + G_t_inf*sin(phi_inf - theta_t)) + 2*V_t*(-B_t0 - B_t_inf) 
    struct[0].Gy_ini[11,11] = -1 
    struct[0].Gy_ini[11,12] = K_a 
    struct[0].Gy_ini[12,12] = 1 
    struct[0].Gy_ini[12,13] = -K_stab*T_pss_1/T_pss_2 
    struct[0].Gy_ini[13,13] = 1 


def ini_struct(dt,values):

    dt +=     [('x_ini', np.float64, (9,1))]
    values += [np.zeros((9,1))]
    dt +=     [('y_ini', np.float64, (14,1))]
    values += [np.zeros((14,1))]
    dt +=     [('f_ini', np.float64, (9,1))]
    values += [np.zeros((9,1))]
    dt +=     [('g_ini', np.float64, (14,1))]
    values += [np.zeros((14,1))]
    dt +=     [('Fx_ini', np.float64, (9,9))]
    values += [np.zeros((9,9))]
    dt +=     [('Fy_ini', np.float64, (9,14))]
    values += [np.zeros((9,14))]
    dt +=     [('Gx_ini', np.float64, (14,9))]
    values += [np.zeros((14,9))]
    dt +=     [('Gy_ini', np.float64, (14,14))]
    values += [np.zeros((14,14))]


@numba.njit(cache=True) 
def solver(struct): 
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    i = 0 

    Dt = struct[i].Dt 
    N_steps = struct[i].N_steps 
    N_store = struct[i].N_store 
    N_x = 9 
    N_outs = 1 
    decimation = struct[i].decimation 
    # initialization 
    t = 0.0 
    run(0.0,struct, 1) 
    it_store = 0 
    struct[i]['T'][0] = t 
    struct[i].X[0,:] = struct[i].x[:,0]  
    for it in range(N_steps-1): 
        t += Dt 
 
        perturbations(t,struct) 
        solver = struct[i].solvern 
        if solver == 1: 
            # forward euler solver  
            run(t,struct, 2)  
            struct[i].x[:] += Dt*struct[i].f  
 
        if solver == 2: 
            # bacward euler solver
            x_0 = np.copy(struct[i].x[:]) 
            for j in range(struct[i].imax): 
                run(t,struct, 2) 
                run(t,struct, 10)  
                phi =  x_0 + Dt*struct[i].f - struct[i].x 
                Dx = np.linalg.solve(-(Dt*struct[i].Fx - np.eye(N_x)), phi) 
                struct[i].x[:] += Dx[:] 
                if np.max(np.abs(Dx)) < struct[i].itol: break 
 
        if solver == 3: 
            # trapezoidal solver
            run(t,struct, 2) 
            f_0 = np.copy(struct[i].f[:]) 
            x_0 = np.copy(struct[i].x[:]) 
            for j in range(struct[i].imax): 
                run(t,struct, 10)  
                phi =  x_0 + 0.5*Dt*(f_0 + struct[i].f) - struct[i].x 
                Dx = np.linalg.solve(-(0.5*Dt*struct[i].Fx - np.eye(N_x)), phi) 
                struct[i].x[:] += Dx[:] 
                run(t,struct, 2) 
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
    sys = {'t_end': 20.0, 'Dt': 0.01, 'solver': 'forward-euler', 'decimation': 10, 'name': 'smib_4rd_avr_pss_rocof_1', 'models': [{'params': {'X_d': 1.81, 'X1d': 0.3, 'T1d0': 8.0, 'X_q': 1.76, 'X1q': 0.65, 'T1q0': 1.0, 'R_a': 0.003, 'X_l': 0.15, 'H': 3.5, 'Omega_b': 376.99111843077515, 'B_t0': 0.0, 'G_t_inf': 0.0, 'T_r': 0.05, 'T_pss_1': 1.281, 'T_pss_2': 0.013, 'T_w': 5.0, 'D': 0.0, 'K_a': 200.0, 'K_stab': 10, 'B_t_inf': -2.10448859455482, 'G_t0': 0.01, 'V_inf': 0.9008}, 'f': ['ddelta=Omega_b*(omega - 1)', 'domega = -(p_e - p_m + D*(omega - 1))/(2*H)', 'de1q =(v_f - e1q + i_d*(X1d - X_d))/T1d0', 'de1d = -(e1d + i_q*(X1q - X_q))/T1q0', 'dv_c =   (V_t - v_c)/T_r', 'dx_pss = (omega_w - x_pss)/T_pss_2', 'dx_w =  (omega - x_w)/T_w', 'dphi_inf = Omega_b*(omega_inf - 1) -1e-6*phi_inf', 'domega_inf = RoCoFpu - 0.0001*(omega_inf - 1)'], 'g': ['p_h@ p_h + 2*H*RoCoFpu', 'p_m0 @ p_m0 - p_m', 'i_d@ v_q - e1q + R_a*i_q + i_d*(X1d - X_l)', 'i_q@ v_d - e1d + R_a*i_d - i_q*(X1q - X_l)', 'p_e@ p_e - i_d*(v_d + R_a*i_d) - i_q*(v_q + R_a*i_q) ', 'v_d@ v_d - V_t*sin(delta - theta_t)', 'v_q@ v_q - V_t*cos(delta - theta_t)', 'P_t@ i_d*v_d - P_t + i_q*v_q', 'Q_t@ i_d*v_q - Q_t - i_q*v_d', 'theta_t @(G_t0 + G_t_inf)*V_t**2 - V_inf*(G_t_inf*cos(theta_t - phi_inf) + B_t_inf*sin(theta_t - phi_inf))*V_t - P_t', 'V_t@ (- B_t0 - B_t_inf)*V_t**2 + V_inf*(B_t_inf*cos(theta_t - phi_inf) - G_t_inf*sin(theta_t - phi_inf))*V_t - Q_t', 'v_f@K_a*(V_ref - v_c + v_pss) - v_f', 'v_pss@v_pss + K_stab*(x_pss*(1/T_pss_2 - 1) - (T_pss_1*omega_w)/T_pss_2)', 'omega_w@ omega_w - omega + x_w'], 'u': {'p_m': 0.9, 'V_ref': 1.0, 'RoCoFpu': 0.0}, 'u_ini': {}, 'y_ini': ['p_h', 'p_m0', 'i_d', 'i_q', 'p_e', 'v_d', 'v_q', 'P_t', 'Q_t', 'theta_t', 'V_t', 'v_f', 'v_pss', 'omega_w'], 'h': ['omega']}], 'perturbations': [{'type': 'step', 'time': 1.0, 'var': 'V_ref', 'final': 1.01}], 'itol': 1e-08, 'imax': 100, 'solvern': 1}
    syst =  smib_4rd_avr_pss_rocof_1_class()
    T,X = solver(syst.struct)
    from scipy.optimize import fsolve
    x0 = np.ones(syst.N_x+syst.N_y)
    s = fsolve(syst.ini_problem,x0 )
    print(s)