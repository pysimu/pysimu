import numpy as np
import numba
from pysimu.nummath import interp


class vsg_monk_class: 
    def __init__(self): 

        self.t_end = 20.000000 
        self.Dt = 0.010000 
        self.decimation = 10.000000 
        self.itol = 0.000000 
        self.solvern = 1 
        self.imax = 100 
        self.N_x = 7 
        self.N_y = 27 
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
                ('H', np.float64),
                ('D', np.float64),
                ('X_s', np.float64),
                ('R_s', np.float64),
                ('K_p', np.float64),
                ('T_p', np.float64),
                ('K_q', np.float64),
                ('T_q', np.float64),
                ('Omega_b', np.float64),
                ('R_g', np.float64),
                ('X_g', np.float64),
                ('V_g', np.float64),
                ('K_pll', np.float64),
                ('p_m', np.float64),
                ('q_s_ref', np.float64),
                ('RoCoFpu', np.float64),
                    ('N_x', np.int64),
                    ('idx', np.int64),
                    ('f', np.float64, (7,1)),
                    ('x', np.float64, (7,1)),
                    ('x_0', np.float64, (7,1)),
                    ('h', np.float64, (1,1)),
                    ('Fx', np.float64, (7,7)),
                    ('T', np.float64, (self.N_store+1,1)),
                    ('X', np.float64, (self.N_store+1,7)),
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
                6.0,   # H 
                 20,   # D 
                 0.1,   # X_s 
                 0.0,   # R_s 
                 1.0,   # K_p 
                 1.0,   # T_p 
                 1.0,   # K_q 
                 1.0,   # T_q 
                 376.99111843077515,   # Omega_b 
                 0.01,   # R_g 
                 0.05,   # X_g 
                 1.0,   # V_g 
                 0.01592,   # K_pll 
                 0.8,   # p_m 
                 0.1,   # q_s_ref 
                 0.0,   # RoCoFpu 
                 7,
                0,
                np.zeros((7,1)),
                np.zeros((7,1)),
                np.zeros((7,1)),
                np.zeros((1,1)),
                                np.zeros((7,7)),
                np.zeros((self.N_store+1,1)),
                np.zeros((self.N_store+1,7)),
                ]  
        ini_struct(dt,values)

        dt +=     [('t', np.float64)]
        values += [0.0]

        dt +=     [('N_y', np.int64)]
        values += [self.N_y]

        dt +=     [('g', np.float64, (27,1))]
        values += [np.zeros((27,1))]
        dt +=     [('y', np.float64, (27,1))]
        values += [np.zeros((27,1))]
        dt +=     [('Fy', np.float64, (7,27))]
        values += [np.zeros((7,27))]
        dt +=     [('Gx', np.float64, (27,7))]
        values += [np.zeros((27,7))]
        dt +=     [('Gy', np.float64, (27,27))]
        values += [np.zeros((27,27))]




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
    H = struct[it].H
    D = struct[it].D
    X_s = struct[it].X_s
    R_s = struct[it].R_s
    K_p = struct[it].K_p
    T_p = struct[it].T_p
    K_q = struct[it].K_q
    T_q = struct[it].T_q
    Omega_b = struct[it].Omega_b
    R_g = struct[it].R_g
    X_g = struct[it].X_g
    V_g = struct[it].V_g
    K_pll = struct[it].K_pll

    # inputs 
    p_m = struct[it].p_m
    q_s_ref = struct[it].q_s_ref
    RoCoFpu = struct[it].RoCoFpu

    # states 
    delta  = struct[it].x[0,0] 
    omega_t  = struct[it].x[1,0] 
    xi_p   = struct[it].x[2,0] 
    xi_q   = struct[it].x[3,0] 
    phi_pll  = struct[it].x[4,0] 
    phi_g  = struct[it].x[5,0] 
    omega_g  = struct[it].x[6,0] 


    # algebraic states 
    p_h = struct[it].y[0,0] 
    p_m0 = struct[it].y[1,0] 
    v_gr = struct[it].y[2,0] 
    v_gi = struct[it].y[3,0] 
    i_sr = struct[it].y[4,0] 
    i_si = struct[it].y[5,0] 
    i_sd_s = struct[it].y[6,0] 
    i_sq_s = struct[it].y[7,0] 
    i_sd = struct[it].y[8,0] 
    i_sq = struct[it].y[9,0] 
    v_sd = struct[it].y[10,0] 
    v_sq = struct[it].y[11,0] 
    v_sd_s = struct[it].y[12,0] 
    v_sq_s = struct[it].y[13,0] 
    v_si = struct[it].y[14,0] 
    v_sr = struct[it].y[15,0] 
    epsilon_p = struct[it].y[16,0] 
    epsilon_q = struct[it].y[17,0] 
    e = struct[it].y[18,0] 
    v_td = struct[it].y[19,0] 
    v_tq = struct[it].y[20,0] 
    p_e = struct[it].y[21,0] 
    p_s = struct[it].y[22,0] 
    q_s = struct[it].y[23,0] 
    v_sd_pll = struct[it].y[24,0] 
    v_sq_pll = struct[it].y[25,0] 
    omega_pll = struct[it].y[26,0] 


    if mode==2: # derivatives 

        ddelta = Omega_b*(omega_t - 1) 
        domega_t = K_p*(epsilon_p + xi_p/T_p) 
        dxi_p  = epsilon_p 
        dxi_q  = epsilon_q 
        dphi_pll = Omega_b*omega_pll 
        dphi_g = Omega_b*(omega_g - 1) -1e-5*phi_g 
        domega_g = RoCoFpu - 1e-5*(omega_g - 1) 

        struct[it].f[0,0] = ddelta   
        struct[it].f[1,0] = domega_t   
        struct[it].f[2,0] = dxi_p    
        struct[it].f[3,0] = dxi_q    
        struct[it].f[4,0] = dphi_pll   
        struct[it].f[5,0] = dphi_g   
        struct[it].f[6,0] = domega_g   

    if mode==3: # algebraic equations 

        struct[it].g[0,0] = 2*H*RoCoFpu + p_h  
        struct[it].g[1,0] = -p_m + p_m0  
        struct[it].g[2,0] = V_g*cos(phi_g) - v_gr  
        struct[it].g[3,0] = V_g*sin(phi_g) - v_gi  
        struct[it].g[4,0] = -i_sr - (R_g*(-v_gr + v_sr) + X_g*(-v_gi + v_si))/(R_g**2 + X_g**2)  
        struct[it].g[5,0] = -i_si - (R_g*(-v_gi + v_si) - X_g*(-v_gr + v_sr))/(R_g**2 + X_g**2)  
        struct[it].g[6,0] = -i_sd_s + i_si  
        struct[it].g[7,0] = -i_sq_s - i_sr  
        struct[it].g[8,0] = -i_sd + i_sd_s*cos(delta) + i_sq_s*sin(delta)  
        struct[it].g[9,0] = -i_sd_s*sin(delta) - i_sq + i_sq_s*cos(delta)  
        struct[it].g[10,0] = -R_s*i_sd + X_s*i_sq + v_sd - v_td  
        struct[it].g[11,0] = -R_s*i_sq - X_s*i_sd + v_sq - v_tq  
        struct[it].g[12,0] = v_sd*cos(delta) - v_sd_s - v_sq*sin(delta)  
        struct[it].g[13,0] = v_sd*sin(delta) + v_sq*cos(delta) - v_sq_s  
        struct[it].g[14,0] = -v_sd_s + v_si  
        struct[it].g[15,0] = -v_sq_s - v_sr  
        struct[it].g[16,0] = -epsilon_p - p_e + p_m  
        struct[it].g[17,0] = -epsilon_q - q_s + q_s_ref  
        struct[it].g[18,0] = K_q*(epsilon_q + xi_q/T_q) - e - 1  
        struct[it].g[19,0] = -v_td  
        struct[it].g[20,0] = e - v_tq  
        struct[it].g[21,0] = i_sd*v_td + i_sq*v_tq - p_e  
        struct[it].g[22,0] = i_sd*v_sd + i_sq*v_sq - p_s  
        struct[it].g[23,0] = i_sd*v_sq - i_sq*v_sd - q_s  
        struct[it].g[24,0] = -v_sd_pll + v_si*sin(phi_pll) + v_sr*cos(phi_pll)  
        struct[it].g[25,0] = v_si*cos(phi_pll) - v_sq_pll - v_sr*sin(phi_pll)  
        struct[it].g[26,0] = K_pll*v_sq_pll - omega_pll  

    if mode==4: # outputs 

        struct[it].h[0,0] = omega_t  
    

    if mode==10: # Fx 

        struct[it].Fx[0,1] = Omega_b 
        struct[it].Fx[1,2] = K_p/T_p 
        struct[it].Fx[5,5] = -1.00000000000000e-5 
        struct[it].Fx[5,6] = Omega_b 
        struct[it].Fx[6,6] = -1.00000000000000e-5 
    

    if mode==11: # Fy,Gx,Gy 

        struct[it].Fy[1,16] = K_p 
        struct[it].Fy[2,16] = 1 
        struct[it].Fy[3,17] = 1 
        struct[it].Fy[4,26] = Omega_b 
    

        struct[it].Gx[2,5] = -V_g*sin(phi_g) 
        struct[it].Gx[3,5] = V_g*cos(phi_g) 
        struct[it].Gx[8,0] = -i_sd_s*sin(delta) + i_sq_s*cos(delta) 
        struct[it].Gx[9,0] = -i_sd_s*cos(delta) - i_sq_s*sin(delta) 
        struct[it].Gx[12,0] = -v_sd*sin(delta) - v_sq*cos(delta) 
        struct[it].Gx[13,0] = v_sd*cos(delta) - v_sq*sin(delta) 
        struct[it].Gx[18,3] = K_q/T_q 
        struct[it].Gx[24,4] = v_si*cos(phi_pll) - v_sr*sin(phi_pll) 
        struct[it].Gx[25,4] = -v_si*sin(phi_pll) - v_sr*cos(phi_pll) 
    

        struct[it].Gy[0,0] = 1 
        struct[it].Gy[1,1] = 1 
        struct[it].Gy[2,2] = -1 
        struct[it].Gy[3,3] = -1 
        struct[it].Gy[4,2] = R_g/(R_g**2 + X_g**2) 
        struct[it].Gy[4,3] = X_g/(R_g**2 + X_g**2) 
        struct[it].Gy[4,4] = -1 
        struct[it].Gy[4,14] = -X_g/(R_g**2 + X_g**2) 
        struct[it].Gy[4,15] = -R_g/(R_g**2 + X_g**2) 
        struct[it].Gy[5,2] = -X_g/(R_g**2 + X_g**2) 
        struct[it].Gy[5,3] = R_g/(R_g**2 + X_g**2) 
        struct[it].Gy[5,5] = -1 
        struct[it].Gy[5,14] = -R_g/(R_g**2 + X_g**2) 
        struct[it].Gy[5,15] = X_g/(R_g**2 + X_g**2) 
        struct[it].Gy[6,5] = 1 
        struct[it].Gy[6,6] = -1 
        struct[it].Gy[7,4] = -1 
        struct[it].Gy[7,7] = -1 
        struct[it].Gy[8,6] = cos(delta) 
        struct[it].Gy[8,7] = sin(delta) 
        struct[it].Gy[8,8] = -1 
        struct[it].Gy[9,6] = -sin(delta) 
        struct[it].Gy[9,7] = cos(delta) 
        struct[it].Gy[9,9] = -1 
        struct[it].Gy[10,8] = -R_s 
        struct[it].Gy[10,9] = X_s 
        struct[it].Gy[10,10] = 1 
        struct[it].Gy[10,19] = -1 
        struct[it].Gy[11,8] = -X_s 
        struct[it].Gy[11,9] = -R_s 
        struct[it].Gy[11,11] = 1 
        struct[it].Gy[11,20] = -1 
        struct[it].Gy[12,10] = cos(delta) 
        struct[it].Gy[12,11] = -sin(delta) 
        struct[it].Gy[12,12] = -1 
        struct[it].Gy[13,10] = sin(delta) 
        struct[it].Gy[13,11] = cos(delta) 
        struct[it].Gy[13,13] = -1 
        struct[it].Gy[14,12] = -1 
        struct[it].Gy[14,14] = 1 
        struct[it].Gy[15,13] = -1 
        struct[it].Gy[15,15] = -1 
        struct[it].Gy[16,16] = -1 
        struct[it].Gy[16,21] = -1 
        struct[it].Gy[17,17] = -1 
        struct[it].Gy[17,23] = -1 
        struct[it].Gy[18,17] = K_q 
        struct[it].Gy[18,18] = -1 
        struct[it].Gy[19,19] = -1 
        struct[it].Gy[20,18] = 1 
        struct[it].Gy[20,20] = -1 
        struct[it].Gy[21,8] = v_td 
        struct[it].Gy[21,9] = v_tq 
        struct[it].Gy[21,19] = i_sd 
        struct[it].Gy[21,20] = i_sq 
        struct[it].Gy[21,21] = -1 
        struct[it].Gy[22,8] = v_sd 
        struct[it].Gy[22,9] = v_sq 
        struct[it].Gy[22,10] = i_sd 
        struct[it].Gy[22,11] = i_sq 
        struct[it].Gy[22,22] = -1 
        struct[it].Gy[23,8] = v_sq 
        struct[it].Gy[23,9] = -v_sd 
        struct[it].Gy[23,10] = -i_sq 
        struct[it].Gy[23,11] = i_sd 
        struct[it].Gy[23,23] = -1 
        struct[it].Gy[24,14] = sin(phi_pll) 
        struct[it].Gy[24,15] = cos(phi_pll) 
        struct[it].Gy[24,24] = -1 
        struct[it].Gy[25,14] = cos(phi_pll) 
        struct[it].Gy[25,15] = -sin(phi_pll) 
        struct[it].Gy[25,25] = -1 
        struct[it].Gy[26,25] = K_pll 
        struct[it].Gy[26,26] = -1 


@numba.njit(cache=True)
def initialization(struct):

    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    H = struct[0].H 
    D = struct[0].D 
    X_s = struct[0].X_s 
    R_s = struct[0].R_s 
    K_p = struct[0].K_p 
    T_p = struct[0].T_p 
    K_q = struct[0].K_q 
    T_q = struct[0].T_q 
    Omega_b = struct[0].Omega_b 
    R_g = struct[0].R_g 
    X_g = struct[0].X_g 
    V_g = struct[0].V_g 
    K_pll = struct[0].K_pll 
    p_m = struct[0].p_m 
    q_s_ref = struct[0].q_s_ref 
    RoCoFpu = struct[0].RoCoFpu 
    delta = struct[0].x_ini[0,0] 
    omega_t = struct[0].x_ini[1,0] 
    xi_p = struct[0].x_ini[2,0] 
    xi_q = struct[0].x_ini[3,0] 
    phi_pll = struct[0].x_ini[4,0] 
    phi_g = struct[0].x_ini[5,0] 
    omega_g = struct[0].x_ini[6,0] 
    p_h = struct[0].y_ini[0,0] 
    p_m0 = struct[0].y_ini[1,0] 
    v_gr = struct[0].y_ini[2,0] 
    v_gi = struct[0].y_ini[3,0] 
    i_sr = struct[0].y_ini[4,0] 
    i_si = struct[0].y_ini[5,0] 
    i_sd_s = struct[0].y_ini[6,0] 
    i_sq_s = struct[0].y_ini[7,0] 
    i_sd = struct[0].y_ini[8,0] 
    i_sq = struct[0].y_ini[9,0] 
    v_sd = struct[0].y_ini[10,0] 
    v_sq = struct[0].y_ini[11,0] 
    v_sd_s = struct[0].y_ini[12,0] 
    v_sq_s = struct[0].y_ini[13,0] 
    v_sr = struct[0].y_ini[14,0] 
    v_si = struct[0].y_ini[15,0] 
    epsilon_p = struct[0].y_ini[16,0] 
    epsilon_q = struct[0].y_ini[17,0] 
    e = struct[0].y_ini[18,0] 
    v_td = struct[0].y_ini[19,0] 
    v_tq = struct[0].y_ini[20,0] 
    p_e = struct[0].y_ini[21,0] 
    p_s = struct[0].y_ini[22,0] 
    q_s = struct[0].y_ini[23,0] 
    v_sd_pll = struct[0].y_ini[24,0] 
    v_sq_pll = struct[0].y_ini[25,0] 
    omega_pll = struct[0].y_ini[26,0] 
    struct[0].f_ini[0,0] = Omega_b*(omega_t - 1) 
    struct[0].f_ini[1,0] = K_p*(epsilon_p + xi_p/T_p) 
    struct[0].f_ini[2,0] = epsilon_p 
    struct[0].f_ini[3,0] = epsilon_q 
    struct[0].f_ini[4,0] = Omega_b*omega_pll 
    struct[0].f_ini[5,0] = Omega_b*(omega_g - 1) - 1.0e-5*phi_g 
    struct[0].f_ini[6,0] = RoCoFpu - 1.0e-5*omega_g + 1.0e-5 
    struct[0].g_ini[0,0]  = 2*H*RoCoFpu + p_h  
    struct[0].g_ini[1,0]  = -p_m + p_m0  
    struct[0].g_ini[2,0]  = V_g*cos(phi_g) - v_gr  
    struct[0].g_ini[3,0]  = V_g*sin(phi_g) - v_gi  
    struct[0].g_ini[4,0]  = -i_sr - (R_g*(-v_gr + v_sr) + X_g*(-v_gi + v_si))/(R_g**2 + X_g**2)  
    struct[0].g_ini[5,0]  = -i_si - (R_g*(-v_gi + v_si) - X_g*(-v_gr + v_sr))/(R_g**2 + X_g**2)  
    struct[0].g_ini[6,0]  = -i_sd_s + i_si  
    struct[0].g_ini[7,0]  = -i_sq_s - i_sr  
    struct[0].g_ini[8,0]  = -i_sd + i_sd_s*cos(delta) + i_sq_s*sin(delta)  
    struct[0].g_ini[9,0]  = -i_sd_s*sin(delta) - i_sq + i_sq_s*cos(delta)  
    struct[0].g_ini[10,0]  = -R_s*i_sd + X_s*i_sq + v_sd - v_td  
    struct[0].g_ini[11,0]  = -R_s*i_sq - X_s*i_sd + v_sq - v_tq  
    struct[0].g_ini[12,0]  = v_sd*cos(delta) - v_sd_s - v_sq*sin(delta)  
    struct[0].g_ini[13,0]  = v_sd*sin(delta) + v_sq*cos(delta) - v_sq_s  
    struct[0].g_ini[14,0]  = -v_sd_s + v_si  
    struct[0].g_ini[15,0]  = -v_sq_s - v_sr  
    struct[0].g_ini[16,0]  = -epsilon_p - p_e + p_m  
    struct[0].g_ini[17,0]  = -epsilon_q - q_s + q_s_ref  
    struct[0].g_ini[18,0]  = K_q*(epsilon_q + xi_q/T_q) - e - 1  
    struct[0].g_ini[19,0]  = -v_td  
    struct[0].g_ini[20,0]  = e - v_tq  
    struct[0].g_ini[21,0]  = i_sd*v_td + i_sq*v_tq - p_e  
    struct[0].g_ini[22,0]  = i_sd*v_sd + i_sq*v_sq - p_s  
    struct[0].g_ini[23,0]  = i_sd*v_sq - i_sq*v_sd - q_s  
    struct[0].g_ini[24,0]  = -v_sd_pll + v_si*sin(phi_pll) + v_sr*cos(phi_pll)  
    struct[0].g_ini[25,0]  = v_si*cos(phi_pll) - v_sq_pll - v_sr*sin(phi_pll)  
    struct[0].g_ini[26,0]  = K_pll*v_sq_pll - omega_pll  
    struct[0].Fx_ini[0,1] = Omega_b 
    struct[0].Fx_ini[1,2] = K_p/T_p 
    struct[0].Fx_ini[5,5] = -1.00000000000000e-5 
    struct[0].Fx_ini[5,6] = Omega_b 
    struct[0].Fx_ini[6,6] = -1.00000000000000e-5 
    struct[0].Fy_ini[1,16] = K_p 
    struct[0].Fy_ini[2,16] = 1 
    struct[0].Fy_ini[3,17] = 1 
    struct[0].Fy_ini[4,26] = Omega_b 
    struct[0].Gx_ini[2,5] = -V_g*sin(phi_g) 
    struct[0].Gx_ini[3,5] = V_g*cos(phi_g) 
    struct[0].Gx_ini[8,0] = -i_sd_s*sin(delta) + i_sq_s*cos(delta) 
    struct[0].Gx_ini[9,0] = -i_sd_s*cos(delta) - i_sq_s*sin(delta) 
    struct[0].Gx_ini[12,0] = -v_sd*sin(delta) - v_sq*cos(delta) 
    struct[0].Gx_ini[13,0] = v_sd*cos(delta) - v_sq*sin(delta) 
    struct[0].Gx_ini[18,3] = K_q/T_q 
    struct[0].Gx_ini[24,4] = v_si*cos(phi_pll) - v_sr*sin(phi_pll) 
    struct[0].Gx_ini[25,4] = -v_si*sin(phi_pll) - v_sr*cos(phi_pll) 
    struct[0].Gy_ini[0,0] = 1 
    struct[0].Gy_ini[1,1] = 1 
    struct[0].Gy_ini[2,2] = -1 
    struct[0].Gy_ini[3,3] = -1 
    struct[0].Gy_ini[4,2] = R_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[4,3] = X_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[4,4] = -1 
    struct[0].Gy_ini[4,14] = -R_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[4,15] = -X_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[5,2] = -X_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[5,3] = R_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[5,5] = -1 
    struct[0].Gy_ini[5,14] = X_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[5,15] = -R_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[6,5] = 1 
    struct[0].Gy_ini[6,6] = -1 
    struct[0].Gy_ini[7,4] = -1 
    struct[0].Gy_ini[7,7] = -1 
    struct[0].Gy_ini[8,6] = cos(delta) 
    struct[0].Gy_ini[8,7] = sin(delta) 
    struct[0].Gy_ini[8,8] = -1 
    struct[0].Gy_ini[9,6] = -sin(delta) 
    struct[0].Gy_ini[9,7] = cos(delta) 
    struct[0].Gy_ini[9,9] = -1 
    struct[0].Gy_ini[10,8] = -R_s 
    struct[0].Gy_ini[10,9] = X_s 
    struct[0].Gy_ini[10,10] = 1 
    struct[0].Gy_ini[10,19] = -1 
    struct[0].Gy_ini[11,8] = -X_s 
    struct[0].Gy_ini[11,9] = -R_s 
    struct[0].Gy_ini[11,11] = 1 
    struct[0].Gy_ini[11,20] = -1 
    struct[0].Gy_ini[12,10] = cos(delta) 
    struct[0].Gy_ini[12,11] = -sin(delta) 
    struct[0].Gy_ini[12,12] = -1 
    struct[0].Gy_ini[13,10] = sin(delta) 
    struct[0].Gy_ini[13,11] = cos(delta) 
    struct[0].Gy_ini[13,13] = -1 
    struct[0].Gy_ini[14,12] = -1 
    struct[0].Gy_ini[14,15] = 1 
    struct[0].Gy_ini[15,13] = -1 
    struct[0].Gy_ini[15,14] = -1 
    struct[0].Gy_ini[16,16] = -1 
    struct[0].Gy_ini[16,21] = -1 
    struct[0].Gy_ini[17,17] = -1 
    struct[0].Gy_ini[17,23] = -1 
    struct[0].Gy_ini[18,17] = K_q 
    struct[0].Gy_ini[18,18] = -1 
    struct[0].Gy_ini[19,19] = -1 
    struct[0].Gy_ini[20,18] = 1 
    struct[0].Gy_ini[20,20] = -1 
    struct[0].Gy_ini[21,8] = v_td 
    struct[0].Gy_ini[21,9] = v_tq 
    struct[0].Gy_ini[21,19] = i_sd 
    struct[0].Gy_ini[21,20] = i_sq 
    struct[0].Gy_ini[21,21] = -1 
    struct[0].Gy_ini[22,8] = v_sd 
    struct[0].Gy_ini[22,9] = v_sq 
    struct[0].Gy_ini[22,10] = i_sd 
    struct[0].Gy_ini[22,11] = i_sq 
    struct[0].Gy_ini[22,22] = -1 
    struct[0].Gy_ini[23,8] = v_sq 
    struct[0].Gy_ini[23,9] = -v_sd 
    struct[0].Gy_ini[23,10] = -i_sq 
    struct[0].Gy_ini[23,11] = i_sd 
    struct[0].Gy_ini[23,23] = -1 
    struct[0].Gy_ini[24,14] = cos(phi_pll) 
    struct[0].Gy_ini[24,15] = sin(phi_pll) 
    struct[0].Gy_ini[24,24] = -1 
    struct[0].Gy_ini[25,14] = -sin(phi_pll) 
    struct[0].Gy_ini[25,15] = cos(phi_pll) 
    struct[0].Gy_ini[25,25] = -1 
    struct[0].Gy_ini[26,25] = K_pll 
    struct[0].Gy_ini[26,26] = -1 


def ini_struct(dt,values):

    dt +=     [('x_ini', np.float64, (7,1))]
    values += [np.zeros((7,1))]
    dt +=     [('y_ini', np.float64, (27,1))]
    values += [np.zeros((27,1))]
    dt +=     [('f_ini', np.float64, (7,1))]
    values += [np.zeros((7,1))]
    dt +=     [('g_ini', np.float64, (27,1))]
    values += [np.zeros((27,1))]
    dt +=     [('Fx_ini', np.float64, (7,7))]
    values += [np.zeros((7,7))]
    dt +=     [('Fy_ini', np.float64, (7,27))]
    values += [np.zeros((7,27))]
    dt +=     [('Gx_ini', np.float64, (27,7))]
    values += [np.zeros((27,7))]
    dt +=     [('Gy_ini', np.float64, (27,27))]
    values += [np.zeros((27,27))]


@numba.njit(cache=True) 
def solver(struct): 
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    i = 0 

    Dt = struct[i].Dt 
    N_steps = struct[i].N_steps 
    N_store = struct[i].N_store 
    N_x = 7 
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
    if t>100.000000: struct[0].p_m = 1.010000



if __name__ == "__main__":
    sys = {'t_end': 20.0, 'Dt': 0.01, 'solver': 'forward-euler', 'decimation': 10, 'name': 'vsg_monk', 'models': [{'params': {'H': 6.0, 'D': 20, 'X_s': 0.1, 'R_s': 0.0, 'K_p': 1.0, 'T_p': 1.0, 'K_q': 1.0, 'T_q': 1.0, 'Omega_b': 376.99111843077515, 'R_g': 0.01, 'X_g': 0.05, 'V_g': 1.0, 'K_pll': 0.01592}, 'f': ['ddelta = Omega_b*(omega_t - 1)', 'domega_t = K_p*(epsilon_p + xi_p/T_p)', 'dxi_p  = epsilon_p', 'dxi_q  = epsilon_q', 'dphi_pll = Omega_b*omega_pll', 'dphi_g = Omega_b*(omega_g - 1) -1e-5*phi_g', 'domega_g = RoCoFpu - 1e-5*(omega_g - 1)'], 'g': ['p_h@ p_h + 2*H*RoCoFpu', 'p_m0 @ p_m0 - p_m', 'v_gr@-v_gr + V_g*cos(phi_g)', 'v_gi@-v_gi + V_g*sin(phi_g)', 'i_sr @ - i_sr -(R_g*(v_sr - v_gr) + X_g*(v_si -v_gi))/(X_g**2 + R_g**2)', 'i_si @ - i_si -(R_g*(v_si - v_gi) - X_g*(v_sr -v_gr))/(X_g**2 + R_g**2)', 'i_sd_s@-i_sd_s + i_si', 'i_sq_s@-i_sq_s - i_sr', 'i_sd @-i_sd + cos(delta)*i_sd_s + cos(delta-pi/2)*i_sq_s', 'i_sq @-i_sq - sin(delta)*i_sd_s - sin(delta-pi/2)*i_sq_s', 'v_sd @ v_sd - v_td - R_s*i_sd + X_s*i_sq', 'v_sq @ v_sq - v_tq - R_s*i_sq - X_s*i_sd', 'v_sd_s @ -v_sd_s + cos(-delta)*v_sd + cos(-delta-pi/2)*v_sq', 'v_sq_s @ -v_sq_s - sin(-delta)*v_sd - sin(-delta-pi/2)*v_sq', 'v_si@-v_sd_s + v_si', 'v_sr@-v_sq_s - v_sr', 'epsilon_p@-epsilon_p + p_m - p_e', 'epsilon_q@-epsilon_q + q_s_ref - q_s', 'e    @ -e -1+ K_q*(epsilon_q + xi_q/T_q) ', 'v_td@ -v_td + 0.0', 'v_tq@ -v_tq +e', 'p_e@-p_e+ i_sd*v_td + i_sq*v_tq', 'p_s@-p_s+ i_sd*v_sd + i_sq*v_sq', 'q_s@-q_s+ i_sd*v_sq - i_sq*v_sd', 'v_sd_pll@ -v_sd_pll + cos(phi_pll)*v_sr + cos(phi_pll-pi/2)*v_si', 'v_sq_pll@ -v_sq_pll - sin(phi_pll)*v_sr - sin(phi_pll-pi/2)*v_si', 'omega_pll@-omega_pll + K_pll*v_sq_pll'], 'u': {'p_m': 0.8, 'q_s_ref': 0.1, 'RoCoFpu': 0.0}, 'y': ['p_h', 'p_m0', 'v_gr', 'v_gi', 'i_sr', 'i_si', 'i_sd_s', 'i_sq_s', 'i_sd', 'i_sq', 'v_sd', 'v_sq', 'v_sd_s', 'v_sq_s', 'v_sr', 'v_si', 'epsilon_p', 'epsilon_q', 'e', 'v_td', 'v_tq', 'p_e', 'p_s', 'q_s', 'v_sd_pll', 'v_sq_pll', 'omega_pll'], 'y_ini': ['p_h', 'p_m0', 'v_gr', 'v_gi', 'i_sr', 'i_si', 'i_sd_s', 'i_sq_s', 'i_sd', 'i_sq', 'v_sd', 'v_sq', 'v_sd_s', 'v_sq_s', 'v_sr', 'v_si', 'epsilon_p', 'epsilon_q', 'e', 'v_td', 'v_tq', 'p_e', 'p_s', 'q_s', 'v_sd_pll', 'v_sq_pll', 'omega_pll'], 'h': ['omega_t']}], 'perturbations': [{'type': 'step', 'time': 100.0, 'var': 'p_m', 'final': 1.01}], 'itol': 1e-08, 'imax': 100, 'solvern': 1}
    syst =  vsg_monk_class()
    T,X = solver(syst.struct)
    from scipy.optimize import fsolve
    x0 = np.ones(syst.N_x+syst.N_y)
    s = fsolve(syst.ini_problem,x0 )
    print(s)