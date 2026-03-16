# Description of the PDE being solved

#%% Demo for using an object
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as spi
import scipy.sparse.linalg as spm
import numpy.linalg as numlin
from scipy.sparse import coo_array, bmat
from scipy.optimize import root, anderson
from scipy.sparse.linalg import spsolve, eigs

import finite_volume.finite_volume as fv

# Option 1: Load user input
# user_input = read('some_file.txt')

# Option 2: input is in the demo script
#Positive and negative parts of a real number
def pos(a):
    return np.maximum(a,0)
def neg(a):
    x = - np.minimum(a,0)
    return x

EPS = 1e-12
MAX_RHO = 1e6

def safe_pow(x, p):
    """
    Safely raise x to the power gamma.
    Ensures x is at least EPS to avoid NaN/inf from negative or zero guesses.
    """
    x = np.clip(x, EPS, MAX_RHO)
    return np.exp(p * np.log(x))

def v_scpr(a, b, c, d, gm, dx):
    """
    Scaled pressure part of the velocity correction (safe version).
    """
    # Ensure positive values for sqrt
    sqrt_arg = 0.25 * (a + b) * (c + d)
    sqrt_val = np.sqrt(max(sqrt_arg, EPS))
    
    # Safe numerator
    num = safe_pow(a, gm) - safe_pow(b, gm)
    
    # Safe denominator
    denom = dx * sqrt_val
    denom = max(denom, EPS)
    
    v = num / denom
    
    # Final safety check
    if not math.isfinite(v):
        v = 0.0  # fallback
    return v

def v_cor(w, r1, r2, r3, r4, R, L, d, gm, dx):
    """
    Corrected velocity after eliminating w^{n+1} safely for nonlinear solvers.
    """
    # Compute v_scpr safely
    v1 = v_scpr(r1, r2, r3, r4, gm, dx)
    
    # Safe denominator for the second term
    denom = dx * 0.5 * (r1 + r2)
    #denom = max(denom, EPS)
    
    # Safe numerator for the second term
    term2_num = safe_pow(R, gm) - safe_pow(L, gm)
    
    # Avoid inf - inf or very small differences
    if np.abs(term2_num) < EPS:
        term2_num = 0.0
    
    term2 = term2_num / denom
    
    # Compute final corrected velocity
    v = w + d * (v1 - term2)
    
    # Final safety check
    if not np.isfinite(v):
        v = 0.0  # fallback
    
    return v


tf = 2.0
kappa = 1.0
nu = 0.1
gamma = 2.0
rho_initial_condition = fv.initial_condition.gaussian_rho
u_initial_condition = fv.initial_condition.constant_u
case = fv.computational_case(a = -3.0, b = 3.0, Tf = 0.5, N = 100, dt = 0.0001, ng = 1)
"-------initialization of the scheme--------------"
a = case.a
b = case.b
N = case.N
nghost = case.ng
l = b - a #length of the domain
cell_size = l/N #uniform cell size
dt = case.dt
lda = dt/cell_size
lda2 = dt/(cell_size * cell_size)
c = (kappa * nu * lda)/cell_size
tstep = case.Tf/case.dt
N_tstep = math.floor(tstep)

c1 = lda * (1.0/4.0)
c2 = nu * (1.0 - kappa) * lda2
c3 = kappa * nu * lda2
d = kappa * (nu ** 2) * (1.0 - kappa) * lda2


#Discretize the initial density by taking cell averages on PRIMAL CELLS
x_dual = np.array([(a + i * cell_size) for i in range(0, N+1)])#edges of N uniform subintervals of (a,b)/edges of the primal cells including bdary a,b
x_prim = np.array([(x_dual[i] + 0.5 * cell_size) for i in range(0, N)])#primal cell centres
#rho_init = np.array([spi.quad(lambda x: (1.0/cell_size) * rho_initial_condition(x), prim_edge[i], prim_edge[i+1])[0] for i in range(0,N)])
rho_init = np.array([rho_initial_condition(x_prim[i]) for i in range(0,N)]) #midpoint rule
#Discretize the initial velocity by taking cell averages on DUAL CELLS
x_dual_int = np.array([(x_dual[i+1]) for i in range(0, N-1)]) #dual cell centres/internal edges/primal cell edges lying inside (a,b) hence excluding a,b
#u_0 = np.array([spi.quad(lambda x: (1.0/cell_size) * u_initial_condition(x), x_prim[i], x_prim[i+1])[0] for i in range(0,N-1)])
u_0 =  np.array([u_initial_condition(x_dual[i]) for i in range(0,N+1)]) #midpoint rule, contains value on the boundary
#Compute the initial DRIFT VELOCITY) on DUAL CELLS
v_init = np.empty(len(rho_init)+1, dtype=rho_init.dtype)
v_init[1:-1] = (rho_init[1:] - rho_init[:-1])/(cell_size * 0.5 * (rho_init[0] + rho_init[-1]))          # normal differences
v_init[0] = (rho_init[0] - rho_init[-1])/(cell_size * 0.5 * (rho_init[0] + rho_init[-1]))                # left wrap
v_init[-1] = (rho_init[0] - rho_init[-1])/(cell_size * 0.5 * (rho_init[0] + rho_init[-1]))               # right wrap to close periodicity
#Compute the discrete EFFECTIVE VELOCITY on DUAL CELLS
w_0 = u_0 - kappa * nu * v_init
#----------------------------------plot discretized initial data--------------------------------------------------
#x = np.linspace(0, 1, num=int(1e2))
#rho0 = initial_condition(x)[1]
f, ax = plt.subplots(layout="constrained")
ax.plot(x_prim, rho_init, label=r"$\rho^-1$")
ax.plot(x_dual, u_0, label=r"$u_0$")
#ax.plot(x_dual, v_init, label=r"$\partial_x \ln(\rho)$")
#ax.plot(x_dual, w_0, label=r"$w_0$")
ax.set_xlabel("x")
ax.set_title("Initial condition")
ax.legend()
#-------------------------------------------------------------------------------------------------------------------
# """-----------------------Update steps---------------------"""
# """Compute rho^0 (PRIMAL CELLS): solve a linear system"""
f_up = fv.convective_flux.flx_upwind
#-------------Entries of the sparse (M-)matrix A corresponding to the update for rho^0------------------------------
p_linsolv = fv.solver_assembly.primal_linsolv_periodic
A = p_linsolv(w_0, lda, c, neg, pos)
#--------------------------------------------------------------------------------------------------------------------
#------------Solving for rho^0 from the corresponding linear problem--------------------------------------------------
rho_0 = spm.spsolve(A, rho_init)

# ax.plot(x_prim, rho_0, label=r"$\rho^0$, lin_solve")
# ax.legend()

d_linsolv = fv.solver_assembly.dual_linsolv
d_linsolv_dif = fv.solver_assembly.dual_linsolv_dif
build_mtx = fv.solver_assembly.build_matrix
# #------------------------------------------------------------------------------------------------------------------
L1_tot = np.sum(rho_0)
print(L1_tot)
#------------------------
"""Time-looping begins"""
#------------------------
num_steps = 10000
for n in range(num_steps):
    #Compute dual average of the discrete mass on the DUAL CELLS
    # rho_init_d = np.array([(0.5 * (rho_init[i+1]+rho_init[i])) for i in range(0,N-1)])
    rho_init_d = np.empty(len(rho_init)+1, dtype=rho_init.dtype)
    rho_init_d[1:-1] = 0.5 * (rho_init[1:] + rho_init[:-1])          
    rho_init_d[0] = 0.5 * (rho_init[0] + rho_init[-1])   # left wrap
    rho_init_d[-1] = 0.5 * (rho_init[0] + rho_init[-1])  # right wrap to close periodicity
    
    rho_0_d = np.empty(len(rho_0)+1, dtype=rho_0.dtype)
    rho_0_d[1:-1] = 0.5 * (rho_0[1:] + rho_0[:-1])          
    rho_0_d[0] = 0.5 * (rho_0[0] + rho_0[-1])   # left wrap
    rho_0_d[-1] = 0.5 * (rho_0[0] + rho_0[-1])  # right wrap to close periodicity
    

    #Pressure scaling step: compute the scaled pressure gradient on the DUAL CELLS
    """Pressure gradient from the previous step"""
    pr_grad = np.empty(len(rho_0)+1, dtype=rho_0.dtype)
    pr_grad[1:-1] = (safe_pow(rho_0[1:], gamma) - safe_pow(rho_0[:-1], gamma))/cell_size       
    pr_grad[0] = (safe_pow(rho_0[0], gamma) - safe_pow(rho_0[-1], gamma))/cell_size  # left wrap
    pr_grad[-1] = (safe_pow(rho_0[0], gamma) - safe_pow(rho_0[-1], gamma))/cell_size  # right wrap to close periodicity
    """Pressure scaling"""
    sc_pr_grad = np.sqrt(rho_0_d / rho_init_d) * pr_grad #the scaled pressure

    # ax.plot(x_dual, rho_init_d, label=r"$\rho^{-1}$ on edges")
    # ax.plot(x_dual, rho_0_d, label=r"$\rho^0$ on edges")
    # #ax.plot(x_dual, sc_pr_grad, label=r"scaled presssure")
    # ax.legend()
  

    #Prediction step: solve a linear system to get the intermediate effective vel. and the drift vel.
    #------------------------------------------------------------------------------------------------
    f_up = fv.convective_flux.flx_upwind
    per_bd = fv.boundary_condition.per_bd
    rho_0_per = per_bd(rho_0, nghost) #populating the ghost cells to compute fluxes on the external edges
    #Effective velocity part of the numerical flux on the interfaces including external edges
    f_ev = np.array([(f_up(rho_0_per[i], rho_0_per[i+1],w_0[i])) for i in range(0,N+1)])
    #Drift velocity part of the numerical flux on the interfaces including external edges
    f_dv = np.array([(rho_0[i+1] - rho_0[i])/cell_size for i in range(0,N-1)])
    f_dv = np.empty(len(rho_0)+1, dtype=rho_0.dtype)
    f_dv[1:-1] = (rho_0[1] - rho_0[:-1])/cell_size           
    f_dv[0] = (rho_0[0] - rho_0[-1])/cell_size # left wrap
    f_dv[-1] = (rho_0[0] - rho_0[-1])/cell_size  # right wrap to close periodicity
    
    #Flux = F_ev - kappa * nu * F_dv
    flx = f_ev - kappa * nu * f_dv

    """Matrix blocks corresponding to the linear system for solving tilde{w} and v"""
    W1 = d_linsolv(flx, rho_0, c1, c2) #tilde{w} part of tilde{w} eqn
    V1 = d_linsolv_dif(rho_0, d) #v part of tilde{w} eqn
    V2 = d_linsolv(flx, rho_0, c1, c3) #v part of v eqn
    W2 = d_linsolv_dif(rho_0, lda2) #tilde{w} part of w eqn

    M = build_mtx(W1,V1, W2, V2)
    M = M.tocsc()
    """Compute the intermediate effective velocity and the drift velocity"""
    rhs_tw = rho_init_d * w_0 - dt * sc_pr_grad #rhs of the w equation
    rhs_v = rho_init_d * v_init #rhs of the v equation
    rhs_dual = np.concatenate((rhs_tw, rhs_v)) #build the vector on right hand side
    twv = spsolve(M, rhs_dual) #vector (tw, v)
    #twv -= twv.mean()
    tw, v = twv[:len(twv)//2], twv[len(twv)//2:]
    
    #ax.plot(x_dual, tw, label=r"$\tilde{w}$")
    # ax.plot(x_dual, v, label=r"$v$")   
    # ax.legend()

   #Correction step: solving implicit non-linear problem for \rho and subsequently correcting w
    #-------------------------------------------------------------------------------------------
    """Description of the non-linear problem emerging from eleminating w^{n+1} in the correction steps"""
    def F(r):
        r = np.maximum(r, 1e-12)   # positivity safeguard

        f = np.zeros_like(r)
        N_d = N + 1 #number of cell interfaces including the boundary
        for i in range(N):
            ip = (i + 1) % N
            im = (i - 1) % N

            iR = i + 1
            iL = i

            dtlap = (r[ip] - 2.0 * r[i] + r[im]) * lda2

            flx_r = f_up(r[i], r[ip],
                      v_cor(tw[iR], rho_0[ip], rho_0[i], rho_init[ip], rho_init[i], r[ip], r[i], dt, gamma, cell_size))
            flx_l = f_up(r[im], r[i],
                      v_cor(tw[iL], rho_0[i], rho_0[im], rho_init[i], rho_init[im], r[i], r[im], dt, gamma, cell_size))
            f[i] = lda * (flx_r - flx_l) - kappa * nu * dtlap  #- rho_0[i]

        return f

    
    rho = rho_0.copy()
    max_iter = 100
    #Picard iteration for solving the non-linear problem for \rho^{n+1}
    for k in range(max_iter):

        r = F(rho)        # uses implicit flux evaluation
        rho_new = rho_0 - r
        r1 = (1.0 - 0.3) * rho + 0.3 * rho_new
        if np.linalg.norm(rho_new - r1) < 1e-12:
            break

        rho = rho_new
    def G(r):
         return r - rho_0 + F(r)
    
    rho = anderson(G, rho, 2, 0.9, maxiter=100, f_tol=1e-12)
    #rho -= np.mean(rho) - np.mean(rho_0)
    rho_per = per_bd(rho, nghost)
    rho_init_per = per_bd(rho_init, nghost)
    """w^{n+1} correction"""
    w = np.array([v_cor(tw[i], rho_0_per[i+1], rho_0_per[i], rho_init_per[i+1], 
                        rho_init_per[i], rho_per[i+1], rho_per[i], dt, gamma, cell_size) for i in range(0,N+1)])
    rho_init = rho_0.copy()
    rho_0 = rho.copy()
    w_0 = w.copy()
    v_init = v.copy()
    print("step:", n)

tv = np.empty(len(rho_init)+1, dtype=rho_init.dtype)
tv[1:-1] = (rho_init[1:] - rho_init[:-1])/(cell_size * 0.5 * (rho_init[0] + rho_init[-1]))          # normal differences
tv[0] = (rho_init[0] - rho_init[-1])/(cell_size * 0.5 * (rho_init[0] + rho_init[-1]))                # left wrap
tv[-1] = (rho_init[0] - rho_init[-1])/(cell_size * 0.5 * (rho_init[0] + rho_init[-1]))               # right wrap to close periodicity
ax.plot(x_prim, rho_0, label=r"$\rho$, T_final")
ax.plot(x_dual, w_0, label=r"$w$, T_final")
#ax.plot(x_dual, v_init, label=r"$v$, T_final")  
u = w_0 + kappa * nu * v_init
ax.plot(x_dual, u, label=r"$u$, T_final")
ax.plot(x_dual, v_init, label=r"$v$, T_final")
ax.plot(x_dual, tv, label=r"$\tilde{v}$, T_final")
ax.legend()
L1_tot_final = np.sum(rho_0)
error_tot = L1_tot - L1_tot_final
print(np.abs(error_tot)) 
print(L1_tot_final)
error_v = v_init - tv
norm_error_v = math.sqrt(cell_size) * np.abs(error_v)
print("error_v:", norm_error_v)
T_f = num_steps * dt
print("Final T:", T_f)


#%%
