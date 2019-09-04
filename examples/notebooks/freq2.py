import numpy as np
import numba
from pysimu.nummath import interp


class freq2_class: 
    def __init__(self): 

        self.t_end = 20.000000 
        self.Dt = 0.010000 
        self.decimation = 10.000000 
        self.itol = 0.000000 
        self.solvern = 1 
        self.imax = 100 
        self.N_x = 11 
        self.N_y = 13 
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
                ('S_1', np.float64),
                ('B_1', np.float64),
                ('H_1', np.float64),
                ('Droop_1', np.float64),
                ('D_1', np.float64),
                ('T_b_1', np.float64),
                ('T_c_1', np.float64),
                ('K_pgov_1', np.float64),
                ('K_igov_1', np.float64),
                ('K_imw_1', np.float64),
                ('omega_ref_1', np.float64),
                ('S_2', np.float64),
                ('B_2', np.float64),
                ('H_2', np.float64),
                ('Droop_2', np.float64),
                ('D_2', np.float64),
                ('T_b_2', np.float64),
                ('T_c_2', np.float64),
                ('K_pgov_2', np.float64),
                ('K_igov_2', np.float64),
                ('K_imw_2', np.float64),
                ('omega_ref_2', np.float64),
                ('Omega_b', np.float64),
                ('S_b', np.float64),
                ('p_load', np.float64),
                ('K_p_agc', np.float64),
                ('K_i_agc', np.float64),
                    ('N_x', np.int64),
                    ('idx', np.int64),
                    ('f', np.float64, (11,1)),
                    ('x', np.float64, (11,1)),
                    ('x_0', np.float64, (11,1)),
                    ('h', np.float64, (2,1)),
                    ('Fx', np.float64, (11,11)),
                    ('T', np.float64, (self.N_store+1,1)),
                    ('X', np.float64, (self.N_store+1,11)),
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
                1,   # S_1 
                 10.0,   # B_1 
                 5.0,   # H_1 
                 0.05,   # Droop_1 
                 10.0,   # D_1 
                 10.0,   # T_b_1 
                 2.0,   # T_c_1 
                 10.0,   # K_pgov_1 
                 2,   # K_igov_1 
                 0.0,   # K_imw_1 
                 1.0,   # omega_ref_1 
                 1,   # S_2 
                 10.0,   # B_2 
                 5.0,   # H_2 
                 0.05,   # Droop_2 
                 10.0,   # D_2 
                 10.0,   # T_b_2 
                 2.0,   # T_c_2 
                 10.0,   # K_pgov_2 
                 2,   # K_igov_2 
                 0.001,   # K_imw_2 
                 1.0,   # omega_ref_2 
                 314.1592653589793,   # Omega_b 
                 1.0,   # S_b 
                 0.4,   # p_load 
                 0.0,   # K_p_agc 
                 0.0001,   # K_i_agc 
                 11,
                0,
                np.zeros((11,1)),
                np.zeros((11,1)),
                np.zeros((11,1)),
                np.zeros((2,1)),
                                np.zeros((11,11)),
                np.zeros((self.N_store+1,1)),
                np.zeros((self.N_store+1,11)),
                ]  
        ini_struct(dt,values)

        dt +=     [('t', np.float64)]
        values += [0.0]

        dt +=     [('N_y', np.int64)]
        values += [self.N_y]

        dt +=     [('g', np.float64, (13,1))]
        values += [np.zeros((13,1))]
        dt +=     [('y', np.float64, (13,1))]
        values += [np.zeros((13,1))]
        dt +=     [('Fy', np.float64, (11,13))]
        values += [np.zeros((11,13))]
        dt +=     [('Gx', np.float64, (13,11))]
        values += [np.zeros((13,11))]
        dt +=     [('Gy', np.float64, (13,13))]
        values += [np.zeros((13,13))]




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
    S_1 = struct[it].S_1
    B_1 = struct[it].B_1
    H_1 = struct[it].H_1
    Droop_1 = struct[it].Droop_1
    D_1 = struct[it].D_1
    T_b_1 = struct[it].T_b_1
    T_c_1 = struct[it].T_c_1
    K_pgov_1 = struct[it].K_pgov_1
    K_igov_1 = struct[it].K_igov_1
    K_imw_1 = struct[it].K_imw_1
    omega_ref_1 = struct[it].omega_ref_1
    S_2 = struct[it].S_2
    B_2 = struct[it].B_2
    H_2 = struct[it].H_2
    Droop_2 = struct[it].Droop_2
    D_2 = struct[it].D_2
    T_b_2 = struct[it].T_b_2
    T_c_2 = struct[it].T_c_2
    K_pgov_2 = struct[it].K_pgov_2
    K_igov_2 = struct[it].K_igov_2
    K_imw_2 = struct[it].K_imw_2
    omega_ref_2 = struct[it].omega_ref_2
    Omega_b = struct[it].Omega_b
    S_b = struct[it].S_b
    p_load = struct[it].p_load
    K_p_agc = struct[it].K_p_agc
    K_i_agc = struct[it].K_i_agc

    # inputs 

    # states 
    phi_1 = struct[it].x[0,0] 
    omega_1  = struct[it].x[1,0] 
    xi_lc_1  = struct[it].x[2,0] 
    xi_w_1  = struct[it].x[3,0] 
    x_m_1  = struct[it].x[4,0] 
    phi_2 = struct[it].x[5,0] 
    omega_2  = struct[it].x[6,0] 
    xi_lc_2  = struct[it].x[7,0] 
    xi_w_2  = struct[it].x[8,0] 
    x_m_2  = struct[it].x[9,0] 
    xi_agc  = struct[it].x[10,0] 


    # algebraic states 
    p_1 = struct[it].y[0,0] 
    eps_1 = struct[it].y[1,0] 
    valv_1 = struct[it].y[2,0] 
    p_m_1 = struct[it].y[3,0] 
    p_2 = struct[it].y[4,0] 
    eps_2 = struct[it].y[5,0] 
    valv_2 = struct[it].y[6,0] 
    p_m_2 = struct[it].y[7,0] 
    phi = struct[it].y[8,0] 
    omega_coi = struct[it].y[9,0] 
    p_ref = struct[it].y[10,0] 
    p_ref_1 = struct[it].y[11,0] 
    p_ref_2 = struct[it].y[12,0] 


    if mode==2: # derivatives 

        dphi_1=Omega_b*(omega_1-omega_coi) 
        domega_1 = (p_m_1 - p_1 - D_1*(omega_1-omega_coi))/(2*H_1) 
        dxi_lc_1 = K_imw_1 * (p_ref_1 - p_1) 
        dxi_w_1 = eps_1 
        dx_m_1 = -x_m_1/T_b_1 + valv_1 
        dphi_2=Omega_b*(omega_2-omega_coi) 
        domega_2 = (p_m_2 - p_2 - D_2*(omega_2-omega_coi))/(2*H_2) 
        dxi_lc_2 = K_imw_2 * (p_ref_2 - p_2) 
        dxi_w_2 = eps_2 
        dx_m_2 = -x_m_2/T_b_2 + valv_2 
        dxi_agc = 1.0 - omega_coi 

        struct[it].f[0,0] = dphi_1  
        struct[it].f[1,0] = domega_1   
        struct[it].f[2,0] = dxi_lc_1   
        struct[it].f[3,0] = dxi_w_1   
        struct[it].f[4,0] = dx_m_1   
        struct[it].f[5,0] = dphi_2  
        struct[it].f[6,0] = domega_2   
        struct[it].f[7,0] = dxi_lc_2   
        struct[it].f[8,0] = dxi_w_2   
        struct[it].f[9,0] = dx_m_2   
        struct[it].f[10,0] = dxi_agc   

    if mode==3: # algebraic equations 

        struct[it].g[0,0] = -B_1*sin(phi - phi_1) - S_1*p_1/S_b  
        struct[it].g[1,0] = -Droop_1*p_1 - eps_1 - omega_1 + omega_ref_1 + xi_lc_1  
        struct[it].g[2,0] = K_igov_1*xi_w_1 + K_pgov_1*eps_1 - valv_1  
        struct[it].g[3,0] = -p_m_1 + x_m_1*(1/T_b_1 - T_c_1/T_b_1**2) + T_c_1*valv_1/T_b_1  
        struct[it].g[4,0] = -B_2*sin(phi - phi_2) - S_2*p_2/S_b  
        struct[it].g[5,0] = -Droop_2*p_2 - eps_2 - omega_2 + omega_ref_2 + xi_lc_2  
        struct[it].g[6,0] = K_igov_2*xi_w_2 + K_pgov_2*eps_2 - valv_2  
        struct[it].g[7,0] = -p_m_2 + x_m_2*(1/T_b_2 - T_c_2/T_b_2**2) + T_c_2*valv_2/T_b_2  
        struct[it].g[8,0] = S_1*p_1/S_b + S_2*p_2/S_b - p_load  
        struct[it].g[9,0] = -omega_coi + (H_1*omega_1 + H_2*omega_2)/(H_1 + H_2)  
        struct[it].g[10,0] = K_i_agc*xi_agc + K_p_agc*(-omega_coi + 1.0) - p_ref  
        struct[it].g[11,0] = S_1*p_ref/S_b - p_ref_1  
        struct[it].g[12,0] = S_2*p_ref/S_b - p_ref_2  

    if mode==4: # outputs 

        struct[it].h[0,0] = omega_1  
        struct[it].h[1,0] = omega_2  
    

    if mode==10: # Fx 

        struct[it].Fx[0,1] = Omega_b 
        struct[it].Fx[1,1] = -D_1/(2*H_1) 
        struct[it].Fx[4,4] = -1/T_b_1 
        struct[it].Fx[5,6] = Omega_b 
        struct[it].Fx[6,6] = -D_2/(2*H_2) 
        struct[it].Fx[9,9] = -1/T_b_2 
    

    if mode==11: # Fy,Gx,Gy 

        struct[it].Fy[0,9] = -Omega_b 
        struct[it].Fy[1,0] = -1/(2*H_1) 
        struct[it].Fy[1,3] = 1/(2*H_1) 
        struct[it].Fy[1,9] = D_1/(2*H_1) 
        struct[it].Fy[2,0] = -K_imw_1 
        struct[it].Fy[2,11] = K_imw_1 
        struct[it].Fy[3,1] = 1 
        struct[it].Fy[4,2] = 1 
        struct[it].Fy[5,9] = -Omega_b 
        struct[it].Fy[6,4] = -1/(2*H_2) 
        struct[it].Fy[6,7] = 1/(2*H_2) 
        struct[it].Fy[6,9] = D_2/(2*H_2) 
        struct[it].Fy[7,4] = -K_imw_2 
        struct[it].Fy[7,12] = K_imw_2 
        struct[it].Fy[8,5] = 1 
        struct[it].Fy[9,6] = 1 
        struct[it].Fy[10,9] = -1 
    

        struct[it].Gx[0,0] = B_1*cos(phi - phi_1) 
        struct[it].Gx[1,1] = -1 
        struct[it].Gx[1,2] = 1 
        struct[it].Gx[2,3] = K_igov_1 
        struct[it].Gx[3,4] = 1/T_b_1 - T_c_1/T_b_1**2 
        struct[it].Gx[4,5] = B_2*cos(phi - phi_2) 
        struct[it].Gx[5,6] = -1 
        struct[it].Gx[5,7] = 1 
        struct[it].Gx[6,8] = K_igov_2 
        struct[it].Gx[7,9] = 1/T_b_2 - T_c_2/T_b_2**2 
        struct[it].Gx[9,1] = H_1/(H_1 + H_2) 
        struct[it].Gx[9,6] = H_2/(H_1 + H_2) 
        struct[it].Gx[10,10] = K_i_agc 
    

        struct[it].Gy[0,0] = -S_1/S_b 
        struct[it].Gy[0,8] = -B_1*cos(phi - phi_1) 
        struct[it].Gy[1,0] = -Droop_1 
        struct[it].Gy[1,1] = -1 
        struct[it].Gy[2,1] = K_pgov_1 
        struct[it].Gy[2,2] = -1 
        struct[it].Gy[3,2] = T_c_1/T_b_1 
        struct[it].Gy[3,3] = -1 
        struct[it].Gy[4,4] = -S_2/S_b 
        struct[it].Gy[4,8] = -B_2*cos(phi - phi_2) 
        struct[it].Gy[5,4] = -Droop_2 
        struct[it].Gy[5,5] = -1 
        struct[it].Gy[6,5] = K_pgov_2 
        struct[it].Gy[6,6] = -1 
        struct[it].Gy[7,6] = T_c_2/T_b_2 
        struct[it].Gy[7,7] = -1 
        struct[it].Gy[8,0] = S_1/S_b 
        struct[it].Gy[8,4] = S_2/S_b 
        struct[it].Gy[9,9] = -1 
        struct[it].Gy[10,9] = -K_p_agc 
        struct[it].Gy[10,10] = -1 
        struct[it].Gy[11,10] = S_1/S_b 
        struct[it].Gy[11,11] = -1 
        struct[it].Gy[12,10] = S_2/S_b 
        struct[it].Gy[12,12] = -1 


@numba.njit(cache=True)
def initialization(struct):

    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    S_1 = struct[0].S_1 
    B_1 = struct[0].B_1 
    H_1 = struct[0].H_1 
    Droop_1 = struct[0].Droop_1 
    D_1 = struct[0].D_1 
    T_b_1 = struct[0].T_b_1 
    T_c_1 = struct[0].T_c_1 
    K_pgov_1 = struct[0].K_pgov_1 
    K_igov_1 = struct[0].K_igov_1 
    K_imw_1 = struct[0].K_imw_1 
    omega_ref_1 = struct[0].omega_ref_1 
    S_2 = struct[0].S_2 
    B_2 = struct[0].B_2 
    H_2 = struct[0].H_2 
    Droop_2 = struct[0].Droop_2 
    D_2 = struct[0].D_2 
    T_b_2 = struct[0].T_b_2 
    T_c_2 = struct[0].T_c_2 
    K_pgov_2 = struct[0].K_pgov_2 
    K_igov_2 = struct[0].K_igov_2 
    K_imw_2 = struct[0].K_imw_2 
    omega_ref_2 = struct[0].omega_ref_2 
    Omega_b = struct[0].Omega_b 
    S_b = struct[0].S_b 
    p_load = struct[0].p_load 
    K_p_agc = struct[0].K_p_agc 
    K_i_agc = struct[0].K_i_agc 
    phi_1 = struct[0].x_ini[0,0] 
    omega_1 = struct[0].x_ini[1,0] 
    xi_lc_1 = struct[0].x_ini[2,0] 
    xi_w_1 = struct[0].x_ini[3,0] 
    x_m_1 = struct[0].x_ini[4,0] 
    phi_2 = struct[0].x_ini[5,0] 
    omega_2 = struct[0].x_ini[6,0] 
    xi_lc_2 = struct[0].x_ini[7,0] 
    xi_w_2 = struct[0].x_ini[8,0] 
    x_m_2 = struct[0].x_ini[9,0] 
    xi_agc = struct[0].x_ini[10,0] 
    p_1 = struct[0].y_ini[0,0] 
    eps_1 = struct[0].y_ini[1,0] 
    valv_1 = struct[0].y_ini[2,0] 
    p_m_1 = struct[0].y_ini[3,0] 
    p_2 = struct[0].y_ini[4,0] 
    eps_2 = struct[0].y_ini[5,0] 
    valv_2 = struct[0].y_ini[6,0] 
    p_m_2 = struct[0].y_ini[7,0] 
    phi = struct[0].y_ini[8,0] 
    omega_coi = struct[0].y_ini[9,0] 
    p_ref = struct[0].y_ini[10,0] 
    p_ref_1 = struct[0].y_ini[11,0] 
    p_ref_2 = struct[0].y_ini[12,0] 
    struct[0].f_ini[0,0] = Omega_b*(omega_1 - omega_coi) 
    struct[0].f_ini[1,0] = (-D_1*(omega_1 - omega_coi) - p_1 + p_m_1)/(2*H_1) 
    struct[0].f_ini[2,0] = K_imw_1*(-p_1 + p_ref_1) 
    struct[0].f_ini[3,0] = eps_1 
    struct[0].f_ini[4,0] = valv_1 - x_m_1/T_b_1 
    struct[0].f_ini[5,0] = Omega_b*(omega_2 - omega_coi) 
    struct[0].f_ini[6,0] = (-D_2*(omega_2 - omega_coi) - p_2 + p_m_2)/(2*H_2) 
    struct[0].f_ini[7,0] = K_imw_2*(-p_2 + p_ref_2) 
    struct[0].f_ini[8,0] = eps_2 
    struct[0].f_ini[9,0] = valv_2 - x_m_2/T_b_2 
    struct[0].f_ini[10,0] = -omega_coi + 1.0 
    struct[0].g_ini[0,0]  = -B_1*sin(phi - phi_1) - S_1*p_1/S_b  
    struct[0].g_ini[1,0]  = -Droop_1*p_1 - eps_1 - omega_1 + omega_ref_1 + xi_lc_1  
    struct[0].g_ini[2,0]  = K_igov_1*xi_w_1 + K_pgov_1*eps_1 - valv_1  
    struct[0].g_ini[3,0]  = -p_m_1 + x_m_1*(1/T_b_1 - T_c_1/T_b_1**2) + T_c_1*valv_1/T_b_1  
    struct[0].g_ini[4,0]  = -B_2*sin(phi - phi_2) - S_2*p_2/S_b  
    struct[0].g_ini[5,0]  = -Droop_2*p_2 - eps_2 - omega_2 + omega_ref_2 + xi_lc_2  
    struct[0].g_ini[6,0]  = K_igov_2*xi_w_2 + K_pgov_2*eps_2 - valv_2  
    struct[0].g_ini[7,0]  = -p_m_2 + x_m_2*(1/T_b_2 - T_c_2/T_b_2**2) + T_c_2*valv_2/T_b_2  
    struct[0].g_ini[8,0]  = S_1*p_1/S_b + S_2*p_2/S_b - p_load  
    struct[0].g_ini[9,0]  = -omega_coi + (H_1*omega_1 + H_2*omega_2)/(H_1 + H_2)  
    struct[0].g_ini[10,0]  = K_i_agc*xi_agc + K_p_agc*(-omega_coi + 1.0) - p_ref  
    struct[0].g_ini[11,0]  = S_1*p_ref/S_b - p_ref_1  
    struct[0].g_ini[12,0]  = S_2*p_ref/S_b - p_ref_2  
    struct[0].Fx_ini[0,1] = Omega_b 
    struct[0].Fx_ini[1,1] = -D_1/(2*H_1) 
    struct[0].Fx_ini[4,4] = -1/T_b_1 
    struct[0].Fx_ini[5,6] = Omega_b 
    struct[0].Fx_ini[6,6] = -D_2/(2*H_2) 
    struct[0].Fx_ini[9,9] = -1/T_b_2 
    struct[0].Fy_ini[0,9] = -Omega_b 
    struct[0].Fy_ini[1,0] = -1/(2*H_1) 
    struct[0].Fy_ini[1,3] = 1/(2*H_1) 
    struct[0].Fy_ini[1,9] = D_1/(2*H_1) 
    struct[0].Fy_ini[2,0] = -K_imw_1 
    struct[0].Fy_ini[2,11] = K_imw_1 
    struct[0].Fy_ini[3,1] = 1 
    struct[0].Fy_ini[4,2] = 1 
    struct[0].Fy_ini[5,9] = -Omega_b 
    struct[0].Fy_ini[6,4] = -1/(2*H_2) 
    struct[0].Fy_ini[6,7] = 1/(2*H_2) 
    struct[0].Fy_ini[6,9] = D_2/(2*H_2) 
    struct[0].Fy_ini[7,4] = -K_imw_2 
    struct[0].Fy_ini[7,12] = K_imw_2 
    struct[0].Fy_ini[8,5] = 1 
    struct[0].Fy_ini[9,6] = 1 
    struct[0].Fy_ini[10,9] = -1 
    struct[0].Gx_ini[0,0] = B_1*cos(phi - phi_1) 
    struct[0].Gx_ini[1,1] = -1 
    struct[0].Gx_ini[1,2] = 1 
    struct[0].Gx_ini[2,3] = K_igov_1 
    struct[0].Gx_ini[3,4] = 1/T_b_1 - T_c_1/T_b_1**2 
    struct[0].Gx_ini[4,5] = B_2*cos(phi - phi_2) 
    struct[0].Gx_ini[5,6] = -1 
    struct[0].Gx_ini[5,7] = 1 
    struct[0].Gx_ini[6,8] = K_igov_2 
    struct[0].Gx_ini[7,9] = 1/T_b_2 - T_c_2/T_b_2**2 
    struct[0].Gx_ini[9,1] = H_1/(H_1 + H_2) 
    struct[0].Gx_ini[9,6] = H_2/(H_1 + H_2) 
    struct[0].Gx_ini[10,10] = K_i_agc 
    struct[0].Gy_ini[0,0] = -S_1/S_b 
    struct[0].Gy_ini[0,8] = -B_1*cos(phi - phi_1) 
    struct[0].Gy_ini[1,0] = -Droop_1 
    struct[0].Gy_ini[1,1] = -1 
    struct[0].Gy_ini[2,1] = K_pgov_1 
    struct[0].Gy_ini[2,2] = -1 
    struct[0].Gy_ini[3,2] = T_c_1/T_b_1 
    struct[0].Gy_ini[3,3] = -1 
    struct[0].Gy_ini[4,4] = -S_2/S_b 
    struct[0].Gy_ini[4,8] = -B_2*cos(phi - phi_2) 
    struct[0].Gy_ini[5,4] = -Droop_2 
    struct[0].Gy_ini[5,5] = -1 
    struct[0].Gy_ini[6,5] = K_pgov_2 
    struct[0].Gy_ini[6,6] = -1 
    struct[0].Gy_ini[7,6] = T_c_2/T_b_2 
    struct[0].Gy_ini[7,7] = -1 
    struct[0].Gy_ini[8,0] = S_1/S_b 
    struct[0].Gy_ini[8,4] = S_2/S_b 
    struct[0].Gy_ini[9,9] = -1 
    struct[0].Gy_ini[10,9] = -K_p_agc 
    struct[0].Gy_ini[10,10] = -1 
    struct[0].Gy_ini[11,10] = S_1/S_b 
    struct[0].Gy_ini[11,11] = -1 
    struct[0].Gy_ini[12,10] = S_2/S_b 
    struct[0].Gy_ini[12,12] = -1 


def ini_struct(dt,values):

    dt +=     [('x_ini', np.float64, (11,1))]
    values += [np.zeros((11,1))]
    dt +=     [('y_ini', np.float64, (13,1))]
    values += [np.zeros((13,1))]
    dt +=     [('f_ini', np.float64, (11,1))]
    values += [np.zeros((11,1))]
    dt +=     [('g_ini', np.float64, (13,1))]
    values += [np.zeros((13,1))]
    dt +=     [('Fx_ini', np.float64, (11,11))]
    values += [np.zeros((11,11))]
    dt +=     [('Fy_ini', np.float64, (11,13))]
    values += [np.zeros((11,13))]
    dt +=     [('Gx_ini', np.float64, (13,11))]
    values += [np.zeros((13,11))]
    dt +=     [('Gy_ini', np.float64, (13,13))]
    values += [np.zeros((13,13))]


@numba.njit(cache=True) 
def solver(struct): 
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    i = 0 

    Dt = struct[i].Dt 
    N_steps = struct[i].N_steps 
    N_store = struct[i].N_store 
    N_x = 11 
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
    if t>1.000000: struct[0].p_load = 1.000000



if __name__ == "__main__":
    sys = {'t_end': 20.0, 'Dt': 0.01, 'solver': 'forward-euler', 'decimation': 10, 'name': 'freq2', 'models': [{'params': {'S_1': 1, 'B_1': 10.0, 'H_1': 5.0, 'Droop_1': 0.05, 'D_1': 10.0, 'T_b_1': 10.0, 'T_c_1': 2.0, 'K_pgov_1': 10.0, 'K_igov_1': 2, 'K_imw_1': 0.0, 'omega_ref_1': 1.0, 'S_2': 1, 'B_2': 10.0, 'H_2': 5.0, 'Droop_2': 0.05, 'D_2': 10.0, 'T_b_2': 10.0, 'T_c_2': 2.0, 'K_pgov_2': 10.0, 'K_igov_2': 2, 'K_imw_2': 0.001, 'omega_ref_2': 1.0, 'Omega_b': 314.1592653589793, 'S_b': 1.0, 'p_load': 0.4, 'K_p_agc': 0.0, 'K_i_agc': 0.0001}, 'f': ['dphi_1=Omega_b*(omega_1-omega_coi)', 'domega_1 = (p_m_1 - p_1 - D_1*(omega_1-omega_coi))/(2*H_1)', 'dxi_lc_1 = K_imw_1 * (p_ref_1 - p_1)', 'dxi_w_1 = eps_1', 'dx_m_1 = -x_m_1/T_b_1 + valv_1', 'dphi_2=Omega_b*(omega_2-omega_coi)', 'domega_2 = (p_m_2 - p_2 - D_2*(omega_2-omega_coi))/(2*H_2)', 'dxi_lc_2 = K_imw_2 * (p_ref_2 - p_2)', 'dxi_w_2 = eps_2', 'dx_m_2 = -x_m_2/T_b_2 + valv_2', 'dxi_agc = 1.0 - omega_coi'], 'g': ['p_1@ B_1*sin(phi_1 - phi) - p_1*(S_1/S_b)', 'eps_1@-eps_1 + omega_ref_1 - omega_1 + xi_lc_1 - Droop_1*p_1', 'valv_1@-valv_1 + K_pgov_1*eps_1 + K_igov_1*xi_w_1', 'p_m_1@-p_m_1 + (1/T_b_1 - T_c_1/(T_b_1*T_b_1))*x_m_1 + T_c_1/T_b_1*valv_1', 'p_2@B_2*sin(phi_2 - phi) - p_2*(S_2/S_b)', 'eps_2@-eps_2 + omega_ref_2 - omega_2 + xi_lc_2 - Droop_2*p_2', 'valv_2@-valv_2 + K_pgov_2*eps_2 + K_igov_2*xi_w_2', 'p_m_2@-p_m_2 + (1/T_b_2 - T_c_2/(T_b_2*T_b_2))*x_m_2 + T_c_2/T_b_2*valv_2', 'phi@ (p_1*S_1/S_b + p_2*S_2/S_b) - p_load', 'omega_coi@-omega_coi + (H_1*omega_1 + H_2*omega_2)/(H_1 + H_2)', 'p_ref@-p_ref + K_p_agc * (1.0 - omega_coi) + K_i_agc * (xi_agc) ', 'p_ref_1@-p_ref_1 + S_1/S_b*p_ref', 'p_ref_2@-p_ref_2 + S_2/S_b*p_ref'], 'u': {}, 'u_ini': {}, 'y_ini': ['p_1', 'eps_1', 'valv_1', 'p_m_1', 'p_2', 'eps_2', 'valv_2', 'p_m_2', 'phi', 'omega_coi', 'p_ref', 'p_ref_1', 'p_ref_2'], 'h': ['omega_1', 'omega_2']}], 'perturbations': [{'type': 'step', 'time': 1.0, 'var': 'p_load', 'final': 1.0}], 'itol': 1e-08, 'imax': 100, 'solvern': 1}
    syst =  freq2_class()
    T,X = solver(syst.struct)
    from scipy.optimize import fsolve
    x0 = np.ones(syst.N_x+syst.N_y)
    s = fsolve(syst.ini_problem,x0 )
    print(s)