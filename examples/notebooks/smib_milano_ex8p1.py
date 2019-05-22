import numpy as np
import numba
from pysimu.nummath import interp


class smib_milano_ex8p1_class: 
    def __init__(self): 

        self.t_end = 20.000000 
        self.Dt = 0.001000 
        self.decimation = 10.000000 
        self.itol = 0.000000 
        self.solvern = 1 
        self.imax = 100 
        self.N_x = 4 
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
                ('D', np.float64),
                ('Omega_b', np.float64),
                ('v_0', np.float64),
                ('theta_0', np.float64),
                ('p_m', np.float64),
                ('v_f', np.float64),
                    ('N_x', np.int64),
                    ('idx', np.int64),
                    ('f', np.float64, (4,1)),
                    ('x', np.float64, (4,1)),
                    ('x_0', np.float64, (4,1)),
                    ('h', np.float64, (1,1)),
                    ('Fx', np.float64, (4,4)),
                    ('T', np.float64, (self.N_store+1,1)),
                    ('X', np.float64, (self.N_store+1,4)),
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
                 0.05,   # X_l 
                 3.5,   # H 
                 1.0,   # D 
                 314.1592653589793,   # Omega_b 
                 0.9008,   # v_0 
                 0.0,   # theta_0 
                 0.2,   # p_m 
                 1.2,   # v_f 
                 4,
                0,
                np.zeros((4,1)),
                np.zeros((4,1)),
                np.zeros((4,1)),
                np.zeros((1,1)),
                                np.zeros((4,4)),
                np.zeros((self.N_store+1,1)),
                np.zeros((self.N_store+1,4)),
                ]  
        ini_struct(dt,values)

        dt +=     [('t', np.float64)]
        values += [0.0]

        dt +=     [('N_y', np.int64)]
        values += [self.N_y]

        dt +=     [('g', np.float64, (9,1))]
        values += [np.zeros((9,1))]
        dt +=     [('y', np.float64, (9,1))]
        values += [np.zeros((9,1))]
        dt +=     [('Fy', np.float64, (4,9))]
        values += [np.zeros((4,9))]
        dt +=     [('Gx', np.float64, (9,4))]
        values += [np.zeros((9,4))]
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
    D = struct[it].D
    Omega_b = struct[it].Omega_b
    v_0 = struct[it].v_0
    theta_0 = struct[it].theta_0

    # inputs 
    p_m = struct[it].p_m
    v_f = struct[it].v_f

    # states 
    delta  = struct[it].x[0,0] 
    omega  = struct[it].x[1,0] 
    e1q  = struct[it].x[2,0] 
    e1d  = struct[it].x[3,0] 


    # algebraic states 
    v_1 = struct[it].y[0,0] 
    theta_1 = struct[it].y[1,0] 
    v_d = struct[it].y[2,0] 
    v_q = struct[it].y[3,0] 
    i_d = struct[it].y[4,0] 
    i_q = struct[it].y[5,0] 
    p_e = struct[it].y[6,0] 
    P_t = struct[it].y[7,0] 
    Q_t = struct[it].y[8,0] 


    if mode==2: # derivatives 

        ddelta = Omega_b*(omega - 1) 
        domega = 1/(2*H)*(p_m - p_e - D*(omega - 1)) 
        de1q = 1/T1d0*(-e1q - (X_d - X1d)*i_d + v_f) 
        de1d = 1/T1q0*(-e1d + (X_q - X1q)*i_q) 

        struct[it].f[0,0] = ddelta   
        struct[it].f[1,0] = domega   
        struct[it].f[2,0] = de1q   
        struct[it].f[3,0] = de1d   

    if mode==3: # algebraic equations 

        struct[it].g[0,0] = P_t + v_0*v_1*sin(theta_0 - theta_1)/X_l  
        struct[it].g[1,0] = Q_t + v_0*v_1*cos(theta_0 - theta_1)/X_l - v_1**2/X_l  
        struct[it].g[2,0] = v_1*sin(delta - theta_1) - v_d  
        struct[it].g[3,0] = v_1*cos(delta - theta_1) - v_q  
        struct[it].g[4,0] = R_a*i_q + X1d*i_d - e1q + v_q  
        struct[it].g[5,0] = R_a*i_d - X1q*i_q - e1d + v_d  
        struct[it].g[6,0] = i_d*(R_a*i_d + v_d) + i_q*(R_a*i_q + v_q) - p_e  
        struct[it].g[7,0] = -P_t + i_d*v_d + i_q*v_q  
        struct[it].g[8,0] = -Q_t + i_d*v_q - i_q*v_d  

    if mode==4: # outputs 

        struct[it].h[0,0] = omega  
    

    if mode==10: # Fx 

        struct[it].Fx[0,1] = Omega_b 
        struct[it].Fx[1,1] = -D/(2*H) 
        struct[it].Fx[2,2] = -1/T1d0 
        struct[it].Fx[3,3] = -1/T1q0 
    

    if mode==11: # Fy,Gx,Gy 

        struct[it].Fy[1,6] = -1/(2*H) 
        struct[it].Fy[2,4] = (X1d - X_d)/T1d0 
        struct[it].Fy[3,5] = (-X1q + X_q)/T1q0 
    

        struct[it].Gx[2,0] = v_1*cos(delta - theta_1) 
        struct[it].Gx[3,0] = -v_1*sin(delta - theta_1) 
        struct[it].Gx[4,2] = -1 
        struct[it].Gx[5,3] = -1 
    

        struct[it].Gy[0,0] = v_0*sin(theta_0 - theta_1)/X_l 
        struct[it].Gy[0,1] = -v_0*v_1*cos(theta_0 - theta_1)/X_l 
        struct[it].Gy[0,7] = 1 
        struct[it].Gy[1,0] = v_0*cos(theta_0 - theta_1)/X_l - 2*v_1/X_l 
        struct[it].Gy[1,1] = v_0*v_1*sin(theta_0 - theta_1)/X_l 
        struct[it].Gy[1,8] = 1 
        struct[it].Gy[2,0] = sin(delta - theta_1) 
        struct[it].Gy[2,1] = -v_1*cos(delta - theta_1) 
        struct[it].Gy[2,2] = -1 
        struct[it].Gy[3,0] = cos(delta - theta_1) 
        struct[it].Gy[3,1] = v_1*sin(delta - theta_1) 
        struct[it].Gy[3,3] = -1 
        struct[it].Gy[4,3] = 1 
        struct[it].Gy[4,4] = X1d 
        struct[it].Gy[4,5] = R_a 
        struct[it].Gy[5,2] = 1 
        struct[it].Gy[5,4] = R_a 
        struct[it].Gy[5,5] = -X1q 
        struct[it].Gy[6,2] = i_d 
        struct[it].Gy[6,3] = i_q 
        struct[it].Gy[6,4] = 2*R_a*i_d + v_d 
        struct[it].Gy[6,5] = 2*R_a*i_q + v_q 
        struct[it].Gy[6,6] = -1 
        struct[it].Gy[7,2] = i_d 
        struct[it].Gy[7,3] = i_q 
        struct[it].Gy[7,4] = v_d 
        struct[it].Gy[7,5] = v_q 
        struct[it].Gy[7,7] = -1 
        struct[it].Gy[8,2] = -i_q 
        struct[it].Gy[8,3] = i_d 
        struct[it].Gy[8,4] = v_q 
        struct[it].Gy[8,5] = -v_d 
        struct[it].Gy[8,8] = -1 


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
    D = struct[0].D 
    Omega_b = struct[0].Omega_b 
    v_0 = struct[0].v_0 
    theta_0 = struct[0].theta_0 
    p_m = struct[0].p_m 
    v_f = struct[0].v_f 
    delta = struct[0].x_ini[0,0] 
    omega = struct[0].x_ini[1,0] 
    e1q = struct[0].x_ini[2,0] 
    e1d = struct[0].x_ini[3,0] 
    theta_1 = struct[0].y_ini[0,0] 
    v_1 = struct[0].y_ini[1,0] 
    v_d = struct[0].y_ini[2,0] 
    v_q = struct[0].y_ini[3,0] 
    i_d = struct[0].y_ini[4,0] 
    i_q = struct[0].y_ini[5,0] 
    p_e = struct[0].y_ini[6,0] 
    P_t = struct[0].y_ini[7,0] 
    Q_t = struct[0].y_ini[8,0] 
    struct[0].f_ini[0,0] = Omega_b*(omega - 1) 
    struct[0].f_ini[1,0] = (-D*(omega - 1) - p_e + p_m)/(2*H) 
    struct[0].f_ini[2,0] = (-e1q - i_d*(-X1d + X_d) + v_f)/T1d0 
    struct[0].f_ini[3,0] = (-e1d + i_q*(-X1q + X_q))/T1q0 
    struct[0].g_ini[0,0]  = P_t + v_0*v_1*sin(theta_0 - theta_1)/X_l  
    struct[0].g_ini[1,0]  = Q_t + v_0*v_1*cos(theta_0 - theta_1)/X_l - v_1**2/X_l  
    struct[0].g_ini[2,0]  = v_1*sin(delta - theta_1) - v_d  
    struct[0].g_ini[3,0]  = v_1*cos(delta - theta_1) - v_q  
    struct[0].g_ini[4,0]  = R_a*i_q + X1d*i_d - e1q + v_q  
    struct[0].g_ini[5,0]  = R_a*i_d - X1q*i_q - e1d + v_d  
    struct[0].g_ini[6,0]  = i_d*(R_a*i_d + v_d) + i_q*(R_a*i_q + v_q) - p_e  
    struct[0].g_ini[7,0]  = -P_t + i_d*v_d + i_q*v_q  
    struct[0].g_ini[8,0]  = -Q_t + i_d*v_q - i_q*v_d  
    struct[0].Fx_ini[0,1] = Omega_b 
    struct[0].Fx_ini[1,1] = -D/(2*H) 
    struct[0].Fx_ini[2,2] = -1/T1d0 
    struct[0].Fx_ini[3,3] = -1/T1q0 
    struct[0].Fy_ini[1,6] = -1/(2*H) 
    struct[0].Fy_ini[2,4] = (X1d - X_d)/T1d0 
    struct[0].Fy_ini[3,5] = (-X1q + X_q)/T1q0 
    struct[0].Gx_ini[2,0] = v_1*cos(delta - theta_1) 
    struct[0].Gx_ini[3,0] = -v_1*sin(delta - theta_1) 
    struct[0].Gx_ini[4,2] = -1 
    struct[0].Gx_ini[5,3] = -1 
    struct[0].Gy_ini[0,0] = -v_0*v_1*cos(theta_0 - theta_1)/X_l 
    struct[0].Gy_ini[0,1] = v_0*sin(theta_0 - theta_1)/X_l 
    struct[0].Gy_ini[0,7] = 1 
    struct[0].Gy_ini[1,0] = v_0*v_1*sin(theta_0 - theta_1)/X_l 
    struct[0].Gy_ini[1,1] = v_0*cos(theta_0 - theta_1)/X_l - 2*v_1/X_l 
    struct[0].Gy_ini[1,8] = 1 
    struct[0].Gy_ini[2,0] = -v_1*cos(delta - theta_1) 
    struct[0].Gy_ini[2,1] = sin(delta - theta_1) 
    struct[0].Gy_ini[2,2] = -1 
    struct[0].Gy_ini[3,0] = v_1*sin(delta - theta_1) 
    struct[0].Gy_ini[3,1] = cos(delta - theta_1) 
    struct[0].Gy_ini[3,3] = -1 
    struct[0].Gy_ini[4,3] = 1 
    struct[0].Gy_ini[4,4] = X1d 
    struct[0].Gy_ini[4,5] = R_a 
    struct[0].Gy_ini[5,2] = 1 
    struct[0].Gy_ini[5,4] = R_a 
    struct[0].Gy_ini[5,5] = -X1q 
    struct[0].Gy_ini[6,2] = i_d 
    struct[0].Gy_ini[6,3] = i_q 
    struct[0].Gy_ini[6,4] = 2*R_a*i_d + v_d 
    struct[0].Gy_ini[6,5] = 2*R_a*i_q + v_q 
    struct[0].Gy_ini[6,6] = -1 
    struct[0].Gy_ini[7,2] = i_d 
    struct[0].Gy_ini[7,3] = i_q 
    struct[0].Gy_ini[7,4] = v_d 
    struct[0].Gy_ini[7,5] = v_q 
    struct[0].Gy_ini[7,7] = -1 
    struct[0].Gy_ini[8,2] = -i_q 
    struct[0].Gy_ini[8,3] = i_d 
    struct[0].Gy_ini[8,4] = v_q 
    struct[0].Gy_ini[8,5] = -v_d 
    struct[0].Gy_ini[8,8] = -1 


def ini_struct(dt,values):

    dt +=     [('x_ini', np.float64, (4,1))]
    values += [np.zeros((4,1))]
    dt +=     [('y_ini', np.float64, (9,1))]
    values += [np.zeros((9,1))]
    dt +=     [('f_ini', np.float64, (4,1))]
    values += [np.zeros((4,1))]
    dt +=     [('g_ini', np.float64, (9,1))]
    values += [np.zeros((9,1))]
    dt +=     [('Fx_ini', np.float64, (4,4))]
    values += [np.zeros((4,4))]
    dt +=     [('Fy_ini', np.float64, (4,9))]
    values += [np.zeros((4,9))]
    dt +=     [('Gx_ini', np.float64, (9,4))]
    values += [np.zeros((9,4))]
    dt +=     [('Gy_ini', np.float64, (9,9))]
    values += [np.zeros((9,9))]


@numba.njit(cache=True) 
def solver(struct): 
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    i = 0 

    Dt = struct[i].Dt 
    N_steps = struct[i].N_steps 
    N_store = struct[i].N_store 
    N_x = 4 
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
    if t>1.000000: struct[0].p_m = 1.010000



if __name__ == "__main__":
    sys = {'t_end': 20.0, 'Dt': 0.001, 'solver': 'forward-euler', 'decimation': 10, 'name': 'smib_milano_ex8p1', 'models': [{'params': {'X_d': 1.81, 'X1d': 0.3, 'T1d0': 8.0, 'X_q': 1.76, 'X1q': 0.65, 'T1q0': 1.0, 'R_a': 0.003, 'X_l': 0.05, 'H': 3.5, 'D': 1.0, 'Omega_b': 314.1592653589793, 'v_0': 0.9008, 'theta_0': 0.0}, 'f': ['ddelta = Omega_b*(omega - 1)', 'domega = 1/(2*H)*(p_m - p_e - D*(omega - 1))', 'de1q = 1/T1d0*(-e1q - (X_d - X1d)*i_d + v_f)', 'de1d = 1/T1q0*(-e1d + (X_q - X1q)*i_q)'], 'g': ['v_1@ P_t - (v_1*v_0*sin(theta_1 - theta_0))/X_l ', 'theta_1@ Q_t + (v_1*v_0*cos(theta_1 - theta_0))/X_l - v_1**2/X_l', 'v_d@ v_1*sin(delta - theta_1) - v_d', 'v_q@ v_1*cos(delta - theta_1) - v_q', 'i_d@ v_q + R_a*i_q + X1d*i_d - e1q', 'i_q@ v_d + R_a*i_d - X1q*i_q - e1d', 'p_e@ i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q) - p_e', 'P_t@ i_d*v_d + i_q*v_q - P_t', 'Q_t@ i_d*v_q - i_q*v_d - Q_t'], 'u': {'p_m': 0.2, 'v_f': 1.2}, 'u_ini': {}, 'y_ini': ['theta_1', 'v_1', 'v_d', 'v_q', 'i_d', 'i_q', 'p_e', 'P_t', 'Q_t'], 'h': ['omega']}], 'perturbations': [{'type': 'step', 'time': 1.0, 'var': 'p_m', 'final': 1.01}], 'itol': 1e-08, 'imax': 100, 'solvern': 1}
    syst =  smib_milano_ex8p1_class()
    T,X = solver(syst.struct)
    from scipy.optimize import fsolve
    x0 = np.ones(syst.N_x+syst.N_y)
    s = fsolve(syst.ini_problem,x0 )
    print(s)