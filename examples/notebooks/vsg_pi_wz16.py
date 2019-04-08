import numpy as np
import numba
from pysimu.nummath import interp


class vsg_pi_wz16_class: 
    def __init__(self): 

        self.t_end = 20.000000 
        self.Dt = 0.010000 
        self.decimation = 10.000000 
        self.itol = 0.000000 
        self.solvern = 1 
        self.imax = 100 
        self.N_x = 5 
        self.N_y = 22 
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
                ('K_p', np.float64),
                ('K_i', np.float64),
                ('K_q', np.float64),
                ('T_q', np.float64),
                ('Omega_b', np.float64),
                ('R_g', np.float64),
                ('X_g', np.float64),
                ('V_g', np.float64),
                ('K_f', np.float64),
                ('K_s', np.float64),
                ('H', np.float64),
                ('p_m', np.float64),
                ('q_s_ref', np.float64),
                ('RoCoFpu', np.float64),
                    ('N_x', np.int64),
                    ('idx', np.int64),
                    ('f', np.float64, (5,1)),
                    ('x', np.float64, (5,1)),
                    ('x_0', np.float64, (5,1)),
                    ('h', np.float64, (1,1)),
                    ('Fx', np.float64, (5,5)),
                    ('T', np.float64, (self.N_store+1,1)),
                    ('X', np.float64, (self.N_store+1,5)),
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
                0.01,   # K_p 
                 0.1,   # K_i 
                 1.0,   # K_q 
                 1.0,   # T_q 
                 376.99111843077515,   # Omega_b 
                 0.01,   # R_g 
                 0.05,   # X_g 
                 1.0,   # V_g 
                 10.0,   # K_f 
                 1.0,   # K_s 
                 5.0,   # H 
                 0.8,   # p_m 
                 0.1,   # q_s_ref 
                 0.0,   # RoCoFpu 
                 5,
                0,
                np.zeros((5,1)),
                np.zeros((5,1)),
                np.zeros((5,1)),
                np.zeros((1,1)),
                                np.zeros((5,5)),
                np.zeros((self.N_store+1,1)),
                np.zeros((self.N_store+1,5)),
                ]  
        ini_struct(dt,values)

        dt +=     [('t', np.float64)]
        values += [0.0]

        dt +=     [('N_y', np.int64)]
        values += [self.N_y]

        dt +=     [('g', np.float64, (22,1))]
        values += [np.zeros((22,1))]
        dt +=     [('y', np.float64, (22,1))]
        values += [np.zeros((22,1))]
        dt +=     [('Fy', np.float64, (5,22))]
        values += [np.zeros((5,22))]
        dt +=     [('Gx', np.float64, (22,5))]
        values += [np.zeros((22,5))]
        dt +=     [('Gy', np.float64, (22,22))]
        values += [np.zeros((22,22))]




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
    K_p = struct[it].K_p
    K_i = struct[it].K_i
    K_q = struct[it].K_q
    T_q = struct[it].T_q
    Omega_b = struct[it].Omega_b
    R_g = struct[it].R_g
    X_g = struct[it].X_g
    V_g = struct[it].V_g
    K_f = struct[it].K_f
    K_s = struct[it].K_s
    H = struct[it].H

    # inputs 
    p_m = struct[it].p_m
    q_s_ref = struct[it].q_s_ref
    RoCoFpu = struct[it].RoCoFpu

    # states 
    phi_s  = struct[it].x[0,0] 
    xi_p   = struct[it].x[1,0] 
    xi_q   = struct[it].x[2,0] 
    phi_g  = struct[it].x[3,0] 
    omega_g  = struct[it].x[4,0] 


    # algebraic states 
    omega_s = struct[it].y[0,0] 
    p_s_ref = struct[it].y[1,0] 
    p_h = struct[it].y[2,0] 
    p_m0 = struct[it].y[3,0] 
    v_gr = struct[it].y[4,0] 
    v_gi = struct[it].y[5,0] 
    i_sr = struct[it].y[6,0] 
    i_si = struct[it].y[7,0] 
    i_sd_s = struct[it].y[8,0] 
    i_sq_s = struct[it].y[9,0] 
    i_sd = struct[it].y[10,0] 
    i_sq = struct[it].y[11,0] 
    v_sd = struct[it].y[12,0] 
    v_sq = struct[it].y[13,0] 
    v_sd_s = struct[it].y[14,0] 
    v_sq_s = struct[it].y[15,0] 
    v_si = struct[it].y[16,0] 
    v_sr = struct[it].y[17,0] 
    epsilon_p = struct[it].y[18,0] 
    epsilon_q = struct[it].y[19,0] 
    p_s = struct[it].y[20,0] 
    q_s = struct[it].y[21,0] 


    if mode==2: # derivatives 

        dphi_s = K_s*Omega_b*(omega_s - 1) 
        dxi_p  = epsilon_p 
        dxi_q  = epsilon_q 
        dphi_g = Omega_b*(omega_g - 1) -1e-5*phi_g 
        domega_g = RoCoFpu - 1e-5*(omega_g - 1) 

        struct[it].f[0,0] = dphi_s   
        struct[it].f[1,0] = dxi_p    
        struct[it].f[2,0] = dxi_q    
        struct[it].f[3,0] = dphi_g   
        struct[it].f[4,0] = domega_g   

    if mode==3: # algebraic equations 

        struct[it].g[0,0] = K_i*xi_p + K_p*epsilon_p - omega_s + 1  
        struct[it].g[1,0] = K_f*(omega_s - 1) + p_m - p_s_ref  
        struct[it].g[2,0] = 2*H*RoCoFpu + p_h  
        struct[it].g[3,0] = -p_m + p_m0  
        struct[it].g[4,0] = V_g*cos(phi_g) - v_gr  
        struct[it].g[5,0] = V_g*sin(phi_g) - v_gi  
        struct[it].g[6,0] = -i_sr - (R_g*(-v_gr + v_sr) + X_g*(-v_gi + v_si))/(R_g**2 + X_g**2)  
        struct[it].g[7,0] = -i_si - (R_g*(-v_gi + v_si) - X_g*(-v_gr + v_sr))/(R_g**2 + X_g**2)  
        struct[it].g[8,0] = -i_sd_s + i_si  
        struct[it].g[9,0] = -i_sq_s - i_sr  
        struct[it].g[10,0] = -i_sd + i_sd_s*cos(phi_s) + i_sq_s*sin(phi_s)  
        struct[it].g[11,0] = -i_sd_s*sin(phi_s) - i_sq + i_sq_s*cos(phi_s)  
        struct[it].g[12,0] = -v_sd  
        struct[it].g[13,0] = K_q*(epsilon_q + xi_q/T_q) - v_sq - 1  
        struct[it].g[14,0] = v_sd*cos(phi_s) - v_sd_s - v_sq*sin(phi_s)  
        struct[it].g[15,0] = v_sd*sin(phi_s) + v_sq*cos(phi_s) - v_sq_s  
        struct[it].g[16,0] = v_sd_s - v_si  
        struct[it].g[17,0] = -v_sq_s - v_sr  
        struct[it].g[18,0] = -epsilon_p + p_s + p_s_ref  
        struct[it].g[19,0] = -epsilon_q - q_s + q_s_ref  
        struct[it].g[20,0] = i_sd*v_sd + i_sq*v_sq - p_s  
        struct[it].g[21,0] = i_sd*v_sq - i_sq*v_sd - q_s  

    if mode==4: # outputs 

        struct[it].h[0,0] = p_m  
    

    if mode==10: # Fx 

        struct[it].Fx[3,3] = -1.00000000000000e-5 
        struct[it].Fx[3,4] = Omega_b 
        struct[it].Fx[4,4] = -1.00000000000000e-5 
    

    if mode==11: # Fy,Gx,Gy 

        struct[it].Fy[0,0] = K_s*Omega_b 
        struct[it].Fy[1,18] = 1 
        struct[it].Fy[2,19] = 1 
    

        struct[it].Gx[0,1] = K_i 
        struct[it].Gx[4,3] = -V_g*sin(phi_g) 
        struct[it].Gx[5,3] = V_g*cos(phi_g) 
        struct[it].Gx[10,0] = -i_sd_s*sin(phi_s) + i_sq_s*cos(phi_s) 
        struct[it].Gx[11,0] = -i_sd_s*cos(phi_s) - i_sq_s*sin(phi_s) 
        struct[it].Gx[13,2] = K_q/T_q 
        struct[it].Gx[14,0] = -v_sd*sin(phi_s) - v_sq*cos(phi_s) 
        struct[it].Gx[15,0] = v_sd*cos(phi_s) - v_sq*sin(phi_s) 
    

        struct[it].Gy[0,0] = -1 
        struct[it].Gy[0,18] = K_p 
        struct[it].Gy[1,0] = K_f 
        struct[it].Gy[1,1] = -1 
        struct[it].Gy[2,2] = 1 
        struct[it].Gy[3,3] = 1 
        struct[it].Gy[4,4] = -1 
        struct[it].Gy[5,5] = -1 
        struct[it].Gy[6,4] = R_g/(R_g**2 + X_g**2) 
        struct[it].Gy[6,5] = X_g/(R_g**2 + X_g**2) 
        struct[it].Gy[6,6] = -1 
        struct[it].Gy[6,16] = -X_g/(R_g**2 + X_g**2) 
        struct[it].Gy[6,17] = -R_g/(R_g**2 + X_g**2) 
        struct[it].Gy[7,4] = -X_g/(R_g**2 + X_g**2) 
        struct[it].Gy[7,5] = R_g/(R_g**2 + X_g**2) 
        struct[it].Gy[7,7] = -1 
        struct[it].Gy[7,16] = -R_g/(R_g**2 + X_g**2) 
        struct[it].Gy[7,17] = X_g/(R_g**2 + X_g**2) 
        struct[it].Gy[8,7] = 1 
        struct[it].Gy[8,8] = -1 
        struct[it].Gy[9,6] = -1 
        struct[it].Gy[9,9] = -1 
        struct[it].Gy[10,8] = cos(phi_s) 
        struct[it].Gy[10,9] = sin(phi_s) 
        struct[it].Gy[10,10] = -1 
        struct[it].Gy[11,8] = -sin(phi_s) 
        struct[it].Gy[11,9] = cos(phi_s) 
        struct[it].Gy[11,11] = -1 
        struct[it].Gy[12,12] = -1 
        struct[it].Gy[13,13] = -1 
        struct[it].Gy[13,19] = K_q 
        struct[it].Gy[14,12] = cos(phi_s) 
        struct[it].Gy[14,13] = -sin(phi_s) 
        struct[it].Gy[14,14] = -1 
        struct[it].Gy[15,12] = sin(phi_s) 
        struct[it].Gy[15,13] = cos(phi_s) 
        struct[it].Gy[15,15] = -1 
        struct[it].Gy[16,14] = 1 
        struct[it].Gy[16,16] = -1 
        struct[it].Gy[17,15] = -1 
        struct[it].Gy[17,17] = -1 
        struct[it].Gy[18,1] = 1 
        struct[it].Gy[18,18] = -1 
        struct[it].Gy[18,20] = 1 
        struct[it].Gy[19,19] = -1 
        struct[it].Gy[19,21] = -1 
        struct[it].Gy[20,10] = v_sd 
        struct[it].Gy[20,11] = v_sq 
        struct[it].Gy[20,12] = i_sd 
        struct[it].Gy[20,13] = i_sq 
        struct[it].Gy[20,20] = -1 
        struct[it].Gy[21,10] = v_sq 
        struct[it].Gy[21,11] = -v_sd 
        struct[it].Gy[21,12] = -i_sq 
        struct[it].Gy[21,13] = i_sd 
        struct[it].Gy[21,21] = -1 


@numba.njit(cache=True)
def initialization(struct):

    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    K_p = struct[0].K_p 
    K_i = struct[0].K_i 
    K_q = struct[0].K_q 
    T_q = struct[0].T_q 
    Omega_b = struct[0].Omega_b 
    R_g = struct[0].R_g 
    X_g = struct[0].X_g 
    V_g = struct[0].V_g 
    K_f = struct[0].K_f 
    K_s = struct[0].K_s 
    H = struct[0].H 
    p_m = struct[0].p_m 
    q_s_ref = struct[0].q_s_ref 
    RoCoFpu = struct[0].RoCoFpu 
    phi_s = struct[0].x_ini[0,0] 
    xi_p = struct[0].x_ini[1,0] 
    xi_q = struct[0].x_ini[2,0] 
    phi_g = struct[0].x_ini[3,0] 
    omega_g = struct[0].x_ini[4,0] 
    omega_s = struct[0].y_ini[0,0] 
    p_s_ref = struct[0].y_ini[1,0] 
    p_h = struct[0].y_ini[2,0] 
    p_m0 = struct[0].y_ini[3,0] 
    v_gr = struct[0].y_ini[4,0] 
    v_gi = struct[0].y_ini[5,0] 
    i_sr = struct[0].y_ini[6,0] 
    i_si = struct[0].y_ini[7,0] 
    i_sd_s = struct[0].y_ini[8,0] 
    i_sq_s = struct[0].y_ini[9,0] 
    i_sd = struct[0].y_ini[10,0] 
    i_sq = struct[0].y_ini[11,0] 
    v_sd = struct[0].y_ini[12,0] 
    v_sq = struct[0].y_ini[13,0] 
    v_sd_s = struct[0].y_ini[14,0] 
    v_sq_s = struct[0].y_ini[15,0] 
    v_sr = struct[0].y_ini[16,0] 
    v_si = struct[0].y_ini[17,0] 
    epsilon_p = struct[0].y_ini[18,0] 
    epsilon_q = struct[0].y_ini[19,0] 
    p_s = struct[0].y_ini[20,0] 
    q_s = struct[0].y_ini[21,0] 
    struct[0].f_ini[0,0] = K_s*Omega_b*(omega_s - 1) 
    struct[0].f_ini[1,0] = epsilon_p 
    struct[0].f_ini[2,0] = epsilon_q 
    struct[0].f_ini[3,0] = Omega_b*(omega_g - 1) - 1.0e-5*phi_g 
    struct[0].f_ini[4,0] = RoCoFpu - 1.0e-5*omega_g + 1.0e-5 
    struct[0].g_ini[0,0]  = K_i*xi_p + K_p*epsilon_p - omega_s + 1  
    struct[0].g_ini[1,0]  = K_f*(omega_s - 1) + p_m - p_s_ref  
    struct[0].g_ini[2,0]  = 2*H*RoCoFpu + p_h  
    struct[0].g_ini[3,0]  = -p_m + p_m0  
    struct[0].g_ini[4,0]  = V_g*cos(phi_g) - v_gr  
    struct[0].g_ini[5,0]  = V_g*sin(phi_g) - v_gi  
    struct[0].g_ini[6,0]  = -i_sr - (R_g*(-v_gr + v_sr) + X_g*(-v_gi + v_si))/(R_g**2 + X_g**2)  
    struct[0].g_ini[7,0]  = -i_si - (R_g*(-v_gi + v_si) - X_g*(-v_gr + v_sr))/(R_g**2 + X_g**2)  
    struct[0].g_ini[8,0]  = -i_sd_s + i_si  
    struct[0].g_ini[9,0]  = -i_sq_s - i_sr  
    struct[0].g_ini[10,0]  = -i_sd + i_sd_s*cos(phi_s) + i_sq_s*sin(phi_s)  
    struct[0].g_ini[11,0]  = -i_sd_s*sin(phi_s) - i_sq + i_sq_s*cos(phi_s)  
    struct[0].g_ini[12,0]  = -v_sd  
    struct[0].g_ini[13,0]  = K_q*(epsilon_q + xi_q/T_q) - v_sq - 1  
    struct[0].g_ini[14,0]  = v_sd*cos(phi_s) - v_sd_s - v_sq*sin(phi_s)  
    struct[0].g_ini[15,0]  = v_sd*sin(phi_s) + v_sq*cos(phi_s) - v_sq_s  
    struct[0].g_ini[16,0]  = v_sd_s - v_si  
    struct[0].g_ini[17,0]  = -v_sq_s - v_sr  
    struct[0].g_ini[18,0]  = -epsilon_p + p_s + p_s_ref  
    struct[0].g_ini[19,0]  = -epsilon_q - q_s + q_s_ref  
    struct[0].g_ini[20,0]  = i_sd*v_sd + i_sq*v_sq - p_s  
    struct[0].g_ini[21,0]  = i_sd*v_sq - i_sq*v_sd - q_s  
    struct[0].Fx_ini[3,3] = -1.00000000000000e-5 
    struct[0].Fx_ini[3,4] = Omega_b 
    struct[0].Fx_ini[4,4] = -1.00000000000000e-5 
    struct[0].Fy_ini[0,0] = K_s*Omega_b 
    struct[0].Fy_ini[1,18] = 1 
    struct[0].Fy_ini[2,19] = 1 
    struct[0].Gx_ini[0,1] = K_i 
    struct[0].Gx_ini[4,3] = -V_g*sin(phi_g) 
    struct[0].Gx_ini[5,3] = V_g*cos(phi_g) 
    struct[0].Gx_ini[10,0] = -i_sd_s*sin(phi_s) + i_sq_s*cos(phi_s) 
    struct[0].Gx_ini[11,0] = -i_sd_s*cos(phi_s) - i_sq_s*sin(phi_s) 
    struct[0].Gx_ini[13,2] = K_q/T_q 
    struct[0].Gx_ini[14,0] = -v_sd*sin(phi_s) - v_sq*cos(phi_s) 
    struct[0].Gx_ini[15,0] = v_sd*cos(phi_s) - v_sq*sin(phi_s) 
    struct[0].Gy_ini[0,0] = -1 
    struct[0].Gy_ini[0,18] = K_p 
    struct[0].Gy_ini[1,0] = K_f 
    struct[0].Gy_ini[1,1] = -1 
    struct[0].Gy_ini[2,2] = 1 
    struct[0].Gy_ini[3,3] = 1 
    struct[0].Gy_ini[4,4] = -1 
    struct[0].Gy_ini[5,5] = -1 
    struct[0].Gy_ini[6,4] = R_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[6,5] = X_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[6,6] = -1 
    struct[0].Gy_ini[6,16] = -R_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[6,17] = -X_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[7,4] = -X_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[7,5] = R_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[7,7] = -1 
    struct[0].Gy_ini[7,16] = X_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[7,17] = -R_g/(R_g**2 + X_g**2) 
    struct[0].Gy_ini[8,7] = 1 
    struct[0].Gy_ini[8,8] = -1 
    struct[0].Gy_ini[9,6] = -1 
    struct[0].Gy_ini[9,9] = -1 
    struct[0].Gy_ini[10,8] = cos(phi_s) 
    struct[0].Gy_ini[10,9] = sin(phi_s) 
    struct[0].Gy_ini[10,10] = -1 
    struct[0].Gy_ini[11,8] = -sin(phi_s) 
    struct[0].Gy_ini[11,9] = cos(phi_s) 
    struct[0].Gy_ini[11,11] = -1 
    struct[0].Gy_ini[12,12] = -1 
    struct[0].Gy_ini[13,13] = -1 
    struct[0].Gy_ini[13,19] = K_q 
    struct[0].Gy_ini[14,12] = cos(phi_s) 
    struct[0].Gy_ini[14,13] = -sin(phi_s) 
    struct[0].Gy_ini[14,14] = -1 
    struct[0].Gy_ini[15,12] = sin(phi_s) 
    struct[0].Gy_ini[15,13] = cos(phi_s) 
    struct[0].Gy_ini[15,15] = -1 
    struct[0].Gy_ini[16,14] = 1 
    struct[0].Gy_ini[16,17] = -1 
    struct[0].Gy_ini[17,15] = -1 
    struct[0].Gy_ini[17,16] = -1 
    struct[0].Gy_ini[18,1] = 1 
    struct[0].Gy_ini[18,18] = -1 
    struct[0].Gy_ini[18,20] = 1 
    struct[0].Gy_ini[19,19] = -1 
    struct[0].Gy_ini[19,21] = -1 
    struct[0].Gy_ini[20,10] = v_sd 
    struct[0].Gy_ini[20,11] = v_sq 
    struct[0].Gy_ini[20,12] = i_sd 
    struct[0].Gy_ini[20,13] = i_sq 
    struct[0].Gy_ini[20,20] = -1 
    struct[0].Gy_ini[21,10] = v_sq 
    struct[0].Gy_ini[21,11] = -v_sd 
    struct[0].Gy_ini[21,12] = -i_sq 
    struct[0].Gy_ini[21,13] = i_sd 
    struct[0].Gy_ini[21,21] = -1 


def ini_struct(dt,values):

    dt +=     [('x_ini', np.float64, (5,1))]
    values += [np.zeros((5,1))]
    dt +=     [('y_ini', np.float64, (22,1))]
    values += [np.zeros((22,1))]
    dt +=     [('f_ini', np.float64, (5,1))]
    values += [np.zeros((5,1))]
    dt +=     [('g_ini', np.float64, (22,1))]
    values += [np.zeros((22,1))]
    dt +=     [('Fx_ini', np.float64, (5,5))]
    values += [np.zeros((5,5))]
    dt +=     [('Fy_ini', np.float64, (5,22))]
    values += [np.zeros((5,22))]
    dt +=     [('Gx_ini', np.float64, (22,5))]
    values += [np.zeros((22,5))]
    dt +=     [('Gy_ini', np.float64, (22,22))]
    values += [np.zeros((22,22))]


@numba.njit(cache=True) 
def solver(struct): 
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    i = 0 

    Dt = struct[i].Dt 
    N_steps = struct[i].N_steps 
    N_store = struct[i].N_store 
    N_x = 5 
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
    sys = {'t_end': 20.0, 'Dt': 0.01, 'solver': 'forward-euler', 'decimation': 10, 'name': 'vsg_pi_wz16', 'models': [{'params': {'K_p': 0.01, 'K_i': 0.1, 'K_q': 1.0, 'T_q': 1.0, 'Omega_b': 376.99111843077515, 'R_g': 0.01, 'X_g': 0.05, 'V_g': 1.0, 'K_f': 10.0, 'K_s': 1.0, 'H': 5.0}, 'f': ['dphi_s = K_s*Omega_b*(omega_s - 1)', 'dxi_p  = epsilon_p', 'dxi_q  = epsilon_q', 'dphi_g = Omega_b*(omega_g - 1) -1e-5*phi_g', 'domega_g = RoCoFpu - 1e-5*(omega_g - 1)'], 'g': ['omega_s @ -omega_s + K_p*epsilon_p + K_i*xi_p +1', 'p_s_ref @ -p_s_ref + K_f*(omega_s - 1) + p_m', 'p_h  @ p_h + 2*H*RoCoFpu', 'p_m0 @ p_m0 - p_m', 'v_gr @-v_gr + V_g*cos(phi_g)', 'v_gi @-v_gi + V_g*sin(phi_g)', 'i_sr @ - i_sr -(R_g*(v_sr - v_gr) + X_g*(v_si -v_gi))/(X_g**2 + R_g**2)', 'i_si @ - i_si -(R_g*(v_si - v_gi) - X_g*(v_sr -v_gr))/(X_g**2 + R_g**2)', 'i_sd_s@-i_sd_s + i_si', 'i_sq_s@-i_sq_s - i_sr', 'i_sd @-i_sd + cos(phi_s)*i_sd_s + cos(phi_s-pi/2)*i_sq_s', 'i_sq @-i_sq - sin(phi_s)*i_sd_s - sin(phi_s-pi/2)*i_sq_s', 'v_sd @ -v_sd + 0.0', 'v_sq @ -v_sq -1+ K_q*(epsilon_q + xi_q/T_q)', 'v_sd_s @ -v_sd_s + cos(-phi_s)*v_sd + cos(-phi_s-pi/2)*v_sq', 'v_sq_s @ -v_sq_s - sin(-phi_s)*v_sd - sin(-phi_s-pi/2)*v_sq', 'v_si@-v_si + v_sd_s', 'v_sr@-v_sr - v_sq_s', 'epsilon_p@-epsilon_p + p_s_ref + p_s', 'epsilon_q@-epsilon_q + q_s_ref - q_s', 'p_s@-p_s+ i_sd*v_sd + i_sq*v_sq', 'q_s@-q_s+ i_sd*v_sq - i_sq*v_sd'], 'u': {'p_m': 0.8, 'q_s_ref': 0.1, 'RoCoFpu': 0.0}, 'y': ['omega_s', 'p_s_ref', 'p_h', 'p_m0', 'v_gr', 'v_gi', 'i_sr', 'i_si', 'i_sd_s', 'i_sq_s', 'i_sd', 'i_sq', 'v_sd', 'v_sq', 'v_sd_s', 'v_sq_s', 'v_sr', 'v_si', 'epsilon_p', 'epsilon_q', 'p_s', 'q_s'], 'y_ini': ['omega_s', 'p_s_ref', 'p_h', 'p_m0', 'v_gr', 'v_gi', 'i_sr', 'i_si', 'i_sd_s', 'i_sq_s', 'i_sd', 'i_sq', 'v_sd', 'v_sq', 'v_sd_s', 'v_sq_s', 'v_sr', 'v_si', 'epsilon_p', 'epsilon_q', 'p_s', 'q_s'], 'h': ['p_m']}], 'perturbations': [{'type': 'step', 'time': 100.0, 'var': 'p_m', 'final': 1.01}], 'itol': 1e-08, 'imax': 100, 'solvern': 1}
    syst =  vsg_pi_wz16_class()
    T,X = solver(syst.struct)
    from scipy.optimize import fsolve
    x0 = np.ones(syst.N_x+syst.N_y)
    s = fsolve(syst.ini_problem,x0 )
    print(s)