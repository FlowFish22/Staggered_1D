# Example of what finite_volume could look like
#%%
import math
import numpy as np
import scipy.integrate as spi
from scipy.sparse import diags, csr_matrix, coo_matrix, coo_array, bmat


class initial_condition:
    """Library of initial conditions."""

    @staticmethod
    def disp_Riemann_rho(x):
        rho0 = 1.5 - 0.5 * np.tanh(x/0.1)
        #u0 = 1.5 - 0.5 * np.tanh(x/0.1) #np.zeros_like(x)
        return (rho0)
    def disp_Riemann_u(x):
        u0 = np.ones_like(x)
        return u0
    """smooth test case"""
    def sine_wave_rho(x):
        rho0 = 1.0 + np.exp(-(x -0.5) * (x - 0.5))
        #u0 = 1.0 + np.cos(x) #np.zeros_like(x)
        return (rho0)
    def sine_wave_u(x):
        u0 = np.cos(x)
        return (u0)
    """1D Gaussian"""
    def gaussian_rho(x):
        z = 2*x
        return np.where(np.abs(z) < 1, 1.0 + np.exp(-1/(1 - z**2)), 1.0)
    def constant_u(x):
        u0 = 0.1 + np.zeros_like(x)
        return (u0)


class computational_case:
    """Describes a computational case.
    Contains data defining the domain and discretization parameters"""

    def __init__(self, a = 0.0, b = 1.0, Tf = 1.0, N = 100, dt=1e-4, ng = 1):
        """Constructor for computational_case class.

        Parameters
        ----------
        N : int,
            number of control volumes.
        Tf: float,
            final time.
        a, b: float a<b, the interval (a,b) is the domain.
        
        dt: float, time-step size

        ng: int, 
                  number of ghost cells at each side. 

        """
        self.a = a
        self.b = b
        self.Tf = Tf
        self.N = N
        self.dt = dt
        self.ng = ng

class boundary_condition:
    """To implemennt boundary conditions"""
    
    @staticmethod
    def per_bd(dis, n_ghost):
        d = np.zeros(len(dis) + 2 * n_ghost) #one layers of ghost cell on each side
        d[-1] += dis[0] #ghost cell with the right boudary take value from the first domain cell on the left
        d[0] += dis[-1] #ghost cell with the left boundary take value from the last domain cell on the right
        d[1:-1] += dis
        return d
    
class convective_flux:
    """To implement different discrete flux"""

    @staticmethod
    def flx_upwind(r1, r2, u):
        """UPWIND flux"""
        #overflow safe clipping
        r1 = np.clip(r1, 1e-12, 1e6) 
        r2 = np.clip(r2, 1e-12, 1e6)
        #---------------------------
        u_pos = 0.5 * (np.fabs(u) + u)
        u_neg = 0.5 * (np.fabs(u) - u)
        flx = r1 * u_pos - r2 * u_neg
        return flx
class solver_assembly:
    """To assemble sparse matrices for explicit and implicit solvers"""

    def primal_linsolv_periodic(a, k, c, f, g):
        """
        Build NxN periodic 3-point stencil matrix for solving unknowns on primal cells:
    
        center[i] = 1 + k*(a[i] + a[i+1]) + c
        left[i]   = -k*a[i] - c
        right[i]  = -k*a[i+1] - c
        
        with periodic wrap:
            a[N] = a[0]

        a[i] corresponds to the interface between the cells i, and i+1
        """
        Nc = len(a) #Internal and external dges
        N = Nc - 1 #Cells

        i = np.arange(N)

        # periodic neighbors (nodes)
        ip = (i + 1) % N
        im = (i - 1) % N

        # periodic cell indices
        iR = (i + 1) % Nc
        iL = i % Nc
        
        # each row has exactly 3 entries (allocated in COO structure)
        rows = np.repeat(np.arange(N), 3)

        cols = np.empty(3*N, dtype=int)
        data = np.empty(3*N)

        cols[0::3] = im
        cols[1::3] = i
        cols[2::3] = ip

        # left entries
        data[0::3] = (
            -k * g(a[iL]) - c
        )

        # center entries
        data[1::3] = (
            1.0 + k * (f(a[iL]) + g(a[iR])) + 2.0 * c
            
        )

        # right entries
        data[2::3] = (
            -k * f(a[iR]) - c
            
         )
        return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    
    def dual_linsolv(flx, rh, c1, c2):
        """
        Build NxN periodic 3-point stencil matrix for solving unknowns on dual cells
        with periodic wrap:
    
        """
        N = len(flx) #len(rh) = N - 1
        N_c = N - 1

        i = np.arange(N)
        ip = (i + 1) % N
        im = (i - 1) % N

        iR = i % N_c
        iL = (i - 1) % N_c

        rows = np.repeat(i, 3)

        cols = np.empty(3*N, dtype=int)
        cols[0::3] = im
        cols[1::3] = i
        cols[2::3] = ip

        data = np.empty(3*N)

        #left
        data[0::3] = (
            - c1 * (flx[i] + flx[im]) - c2 * rh[iL]
        )

        #center
        data[1::3] = (
            0.5 * (rh[iL] + rh[iR]) + c1 * (flx[ip] - flx[im]) + c2 * (rh[iL] + rh[iR])
        )

        #right
        data[2::3] = (
            c1 * (flx[ip] + flx[i]) - c2 * rh[iR]
        )
        return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    def dual_linsolv_dif(rh, c):
        """
        Build NxN periodic 3-point stencil matrix for solving unknowns on dual cells in diffusion terms
        with periodic wrap:
    
        """
        N = len(rh) + 1
        N_c = len(rh)
        
        i = np.arange(N)
        ip = (i + 1) % N
        im = (i - 1) % N

        iR = i % N_c
        iL = (i - 1) % N_c

        rows = np.repeat(i, 3)

        cols = np.empty(3*N, dtype=int)
        cols[0::3] = ip
        cols[1::3] = i
        cols[2::3] = im

        data = np.empty(3*N)

        #left
        data[0::3] = (
             c * rh[iL]
        )

        #center
        data[1::3] = (
           - c * (rh[iL] + rh[iR])
        )

        #right
        data[2::3] = (
            c * rh[iR]
        )
        
        return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    
    def build_matrix(a, b, c, d):
        """Build matrix from sparse coo blocks"""
        return bmat([[a, b], 
                     [c,d]])
# %%
