import numpy as np
import numba
from pysimu.nummath import interp


class vsg_w_class: 
    def __init__(self): 

        self.t_end = 20.000000 
        self.Dt = 0.010000 
        self.decimation = 10.000000 
        self.itol = 0.000000 
        self.solvern = 1 
        self.imax = 100 
        self.N_x = 3 
        self.N_y = 21 
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
                ('X', np.float64),
                ('R', np.float64),
                ('K_p', np.float64),
                ('T_pi', np.float64),
                ('K_q', np.float64),
                ('T_q', np.float64),
                ('K_d', np.float64),
                ('T_d', np.float64),
                ('Omega_b', np.float64),
                ('R_g', np.float64),
                ('X_g', np.float64),
                ('phi_g', np.float64),
                ('V_g', np.float64),
                ('p_m', np.float64),
                ('q_s_ref', np.float64),
                    ('N_x', np.int64),
                    ('idx', np.int64),
                    ('f', np.float64, (3,1)),
                    ('x', np.float64, (3,1)),
                    ('x_0', np.float64, (3,1)),
                    ('h', np.float64, (1,1)),
                    ('Fx', np.float64, (3,3)),
                    ('T', np.float64, (self.N_store+1,1)),
                    ('X', np.float64, (self.N_store+1,3)),
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
                 0.3,   # X 
                 0.1,   # R 
                 1.0,   # K_p 
                 10.0,   # T_pi 
                 1.0,   # K_q 
                 1.0,   # T_q 
                 1.0,   # K_d 
                 1.0,   # T_d 
                 376.99111843077515,   # Omega_b 
                 0.01,   # R_g 
                 0.05,   # X_g 
                 0.0,   # phi_g 
                 1.0,   # V_g 
                 0.8,   # p_m 
                 0.1,   # q_s_ref 
                 3,
                0,
                np.zeros((3,1)),
                np.zeros((3,1)),
                np.zeros((3,1)),
                np.zeros((1,1)),
                                np.zeros((3,3)),
                np.zeros((self.N_store+1,1)),
                np.zeros((self.N_store+1,3)),
                ]  
        ini_struct(dt,values)

        dt +=     [('t', np.float64)]
        values += [0.0]

        dt +=     [('N_y', np.int64)]
        values += [self.N_y]

        dt +=     [('g', np.float64, (21,1))]
        values += [np.zeros((21,1))]
        dt +=     [('y', np.float64, (21,1))]
        values += [np.zeros((21,1))]
        dt +=     [('Fy', np.float64, (3,21))]
        values += [np.zeros((3,21))]
        dt +=     [('Gx', np.float64, (21,3))]
        values += [np.zeros((21,3))]
        dt +=     [('Gy', np.float64, (21,21))]
        values += [np.zeros((21,21))]




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
    X = struct[it].X
    R = struct[it].R
    K_p = struct[it].K_p
    T_pi = struct[it].T_pi
    K_q = struct[it].K_q
    T_q = struct[it].T_q
    K_d = struct[it].K_d
    T_d = struct[it].T_d
    Omega_b = struct[it].Omega_b
    R_g = struct[it].R_g
    X_g = struct[it].X_g
    phi_g = struct[it].phi_g
    V_g = struct[it].V_g

    # inputs 
    p_m = struct[it].p_m
    q_s_ref = struct[it].q_s_ref

    # states 
    delta  = struct[it].x[0,0] 
    omega_t  = struct[it].x[1,0] 
    xi_q  = struct[it].x[2,0] 


    # algebraic states 
    v_gr = struct[it].y[0,0] 
    v_gi = struct[it].y[1,0] 
    i_sr = struct[it].y[2,0] 
    i_si = struct[it].y[3,0] 
    i_sd_s = struct[it].y[4,0] 
    i_sq_s = struct[it].y[5,0] 
    i_sd = struct[it].y[6,0] 
    i_sq = struct[it].y[7,0] 
    v_sd = struct[it].y[8,0] 
    v_sq = struct[it].y[9,0] 
    v_sd_s = struct[it].y[10,0] 
    v_sq_s = struct[it].y[11,0] 
    v_si = struct[it].y[12,0] 
    v_sr = struct[it].y[13,0] 
    epsilon_q = struct[it].y[14,0] 
    e = struct[it].y[15,0] 
    v_td = struct[it].y[16,0] 
    v_tq = struct[it].y[17,0] 
    p_e = struct[it].y[18,0] 
    p_s = struct[it].y[19,0] 
    q_s = struct[it].y[20,0] 


    if mode==2: # derivatives 

        ddelta = Omega_b*(omega_t-1) 
        domega_t = 1/(2*H)*(p_m - p_e - (omega_t-1)*D) 
        dxi_q = q_s 

        struct[it].f[0,0] = ddelta   
        struct[it].f[1,0] = domega_t   
        struct[it].f[2,0] = dxi_q   

    if mode==3: # algebraic equations 

        struct[it].g[0,0] = V_g*cos(phi_g) - v_gr  
        struct[it].g[1,0] = V_g*sin(phi_g) - v_gi  
        struct[it].g[2,0] = -i_sr - (R_g*(-v_gr + v_sr) + X_g*(-v_gr + v_sr))/(R_g**2 + X_g**2)  
        struct[it].g[3,0] = -i_si - (R_g*(-v_gi + v_si) - X_g*(-v_gi + v_si))/(R_g**2 + X_g**2)  
        struct[it].g[4,0] = -i_sd_s + i_si  
        struct[it].g[5,0] = -i_sq_s - i_sr  
        struct[it].g[6,0] = -i_sd + i_sd_s*cos(delta) + i_sq_s*sin(delta)  
        struct[it].g[7,0] = -i_sd_s*sin(delta) - i_sq + i_sq_s*cos(delta)  
        struct[it].g[8,0] = -R*i_sd + X*i_sq + v_sd - v_td  
        struct[it].g[9,0] = -R*i_sq - X*i_sd + v_sq - v_tq  
        struct[it].g[10,0] = v_sd*cos(delta) - v_sd_s - v_sq*sin(delta)  
        struct[it].g[11,0] = v_sd*sin(delta) + v_sq*cos(delta) - v_sq_s  
        struct[it].g[12,0] = -v_sd_s + v_si  
        struct[it].g[13,0] = -v_sq_s - v_sr  
        struct[it].g[14,0] = -epsilon_q - q_s + q_s_ref  
        struct[it].g[15,0] = K_q*(epsilon_q + xi_q/T_q) - e - 1  
        struct[it].g[16,0] = -v_td  
        struct[it].g[17,0] = e - v_td  
        struct[it].g[18,0] = i_sd*v_td + i_sq*v_tq - p_e  
        struct[it].g[19,0] = i_sd*v_sd + i_sq*v_sq - p_s  
        struct[it].g[20,0] = i_sd*v_sq - i_sq*v_sd - q_s  

    if mode==4: # outputs 

        struct[it].h[0,0] = omega_t  
    

    if mode==10: # Fx 

        struct[it].Fx[0,1] = Omega_b 
        struct[it].Fx[1,1] = -D/(2*H) 
    

    if mode==11: # Fy,Gx,Gy 

        struct[it].Fy[1,18] = -1/(2*H) 
        struct[it].Fy[2,20] = 1 
    

        struct[it].Gx[6,0] = -i_sd_s*sin(delta) + i_sq_s*cos(delta) 
        struct[it].Gx[7,0] = -i_sd_s*cos(delta) - i_sq_s*sin(delta) 
        struct[it].Gx[10,0] = -v_sd*sin(delta) - v_sq*cos(delta) 
        struct[it].Gx[11,0] = v_sd*cos(delta) - v_sq*sin(delta) 
        struct[it].Gx[15,2] = K_q/T_q 
    

        struct[it].Gy[0,0] = -1 
        struct[it].Gy[1,1] = -1 
        struct[it].Gy[2,0] = -(-R_g - X_g)/(R_g**2 + X_g**2) 
        struct[it].Gy[2,2] = -1 
        struct[it].Gy[2,13] = -(R_g + X_g)/(R_g**2 + X_g**2) 
        struct[it].Gy[3,1] = -(-R_g + X_g)/(R_g**2 + X_g**2) 
        struct[it].Gy[3,3] = -1 
        struct[it].Gy[3,12] = -(R_g - X_g)/(R_g**2 + X_g**2) 
        struct[it].Gy[4,3] = 1 
        struct[it].Gy[4,4] = -1 
        struct[it].Gy[5,2] = -1 
        struct[it].Gy[5,5] = -1 
        struct[it].Gy[6,4] = cos(delta) 
        struct[it].Gy[6,5] = sin(delta) 
        struct[it].Gy[6,6] = -1 
        struct[it].Gy[7,4] = -sin(delta) 
        struct[it].Gy[7,5] = cos(delta) 
        struct[it].Gy[7,7] = -1 
        struct[it].Gy[8,6] = -R 
        struct[it].Gy[8,7] = X 
        struct[it].Gy[8,8] = 1 
        struct[it].Gy[8,16] = -1 
        struct[it].Gy[9,6] = -X 
        struct[it].Gy[9,7] = -R 
        struct[it].Gy[9,9] = 1 
        struct[it].Gy[9,17] = -1 
        struct[it].Gy[10,8] = cos(delta) 
        struct[it].Gy[10,9] = -sin(delta) 
        struct[it].Gy[10,10] = -1 
        struct[it].Gy[11,8] = sin(delta) 
        struct[it].Gy[11,9] = cos(delta) 
        struct[it].Gy[11,11] = -1 
        struct[it].Gy[12,10] = -1 
        struct[it].Gy[12,12] = 1 
        struct[it].Gy[13,11] = -1 
        struct[it].Gy[13,13] = -1 
        struct[it].Gy[14,14] = -1 
        struct[it].Gy[14,20] = -1 
        struct[it].Gy[15,14] = K_q 
        struct[it].Gy[15,15] = -1 
        struct[it].Gy[16,16] = -1 
        struct[it].Gy[17,15] = 1 
        struct[it].Gy[17,16] = -1 
        struct[it].Gy[18,6] = v_td 
        struct[it].Gy[18,7] = v_tq 
        struct[it].Gy[18,16] = i_sd 
        struct[it].Gy[18,17] = i_sq 
        struct[it].Gy[18,18] = -1 
        struct[it].Gy[19,6] = v_sd 
        struct[it].Gy[19,7] = v_sq 
        struct[it].Gy[19,8] = i_sd 
        struct[it].Gy[19,9] = i_sq 
        struct[it].Gy[19,19] = -1 
        struct[it].Gy[20,6] = v_sq 
        struct[it].Gy[20,7] = -v_sd 
        struct[it].Gy[20,8] = -i_sq 
        struct[it].Gy[20,9] = i_sd 
        struct[it].Gy[20,20] = -1 
