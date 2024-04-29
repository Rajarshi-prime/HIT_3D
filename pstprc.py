import numpy as np 
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from mpi4py import MPI
from time import time
import pathlib
import os

curr_path = pathlib.Path(__file__).parent

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
## ---------------------------------------
## --------------- Params ----------------
nu = 1/200
N = 128
## ---------------------------------------


## ---------- rest of the parameters --------------
loadPath = pathlib.Path(f"/home/rajarshi.chattopadhyay/python/3D-DNS/data/zero-start/")
savePath = loadPath
if rank == 0: 
    try:
        savePath.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
comm.Barrier()
paramfile = (loadPath/f"params.txt")



num_slabs = 128
lp = 1

Ns = num_slabs//num_process
Nslab = N// num_slabs

if rank ==0: print(f"# slabs and Ns of the sim : {num_slabs},{Ns} ")
## -------------------------------------------------------


## -------------Defining the grid ---------------
PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
Np = N//num_process
sx = slice(rank*Np ,  (rank+1)*Np)
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X[sx], Y, Z, indexing='ij')

Kx = Ky = Kz = fftfreq(N,  1./N)*TWO_PI/L
Kz = Kz[:Nf]
dkx, dky,dkz = Kx[1]- Kx[0], Ky[1]  - Ky[0], Kz[1]- Kz[0]
kx,  ky,  kz = np.meshgrid(Kx,  Ky[sx],  Kz,  indexing = 'ij')
## -----------------------------------------------



## --------- kx and ky for differentiation --------------------------

kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()
kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()

## ------------------------------------------------------------------

## ------------Useful Operators-------------------
dealias = (abs(kx)<N//3)*(abs(ky)<N//3)*(abs(kz)<N//3)

lap = -(kx**2 + ky**2 + kz**2 )
k = (-lap)**0.5
invlap = 1.0/np.where(lap == 0, np.inf,  lap)

# Hyperviscous operator
vis = nu*(-1*k**2)**(lp) ## This is in Fourier Space
## -------------------------------------------------


## -------------Empty arrays ----------------------------------------

u  = np.empty((3, Np, N, N), dtype= np.float64)
u1 = np.empty((3, Np, N, N), dtype= np.float64)
p = u[0].copy()

uk = np.empty((3, N, Np, Nf), dtype= np.complex128)
u1k = np.empty((3, N, Np, Nf), dtype= np.complex128)
pk = uk[0].copy()


rhsuk = np.empty_like(pk)
rhsvk = rhsuk.copy()
rhswk = rhsuk.copy()

rhsu = np.empty_like(p)
rhsv = rhsu.copy()
rhsw = rhsu.copy()

rhsu1 = np.empty_like(p)
rhsu2 = np.empty_like(p)
rhsu3 = np.empty_like(p)

div = p.copy()



arr_temp_k = np.empty((N, Np, N),dtype= np.float64)
arr_temp_fr = np.empty((Np, N, Nf), dtype= np.complex128)      
arr_temp_ifr = np.empty((N, Np, Nf), dtype= np.complex128)      
arr_mpi = np.empty((num_process,  Np,  Np, Nf), dtype= np.complex128)
arr_mpi_r = np.empty((num_process,  Np,  Np, N), dtype= np.float64)


ek = np.empty_like(pk,dtype = np.float64)
PIk = np.empty_like(ek,dtype = np.float64)
## ------------------------------------------------------------------


## ------------------------------------------------------------------
##                      Functions to use
## ------------------------------------------------------------------

def create_dir(path):
    path = pathlib.Path(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass


## --------------------- FFT + diff fns ---------------------------
def rfft_mpi(u, fu):
    arr_temp_fr[:] = rfft2(u,  axes=(1, 2))
    arr_mpi[:] = np.swapaxes(np.reshape(arr_temp_fr, (Np,  num_process,  Np, Nf)), 0, 1)
    comm.Alltoall([arr_mpi,  MPI.DOUBLE_COMPLEX], [fu,  MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis = 0)
    return fu

def irfft_mpi(fu, u):
    arr_temp_ifr[:] = ifft(fu,  axis = 0)
    comm.Alltoall([arr_temp_ifr,  MPI.DOUBLE_COMPLEX], [arr_mpi, MPI.DOUBLE_COMPLEX])
    arr_temp_fr[:] = np.reshape(np.swapaxes(arr_mpi,  0, 1), (Np,  N,  Nf))
    u[:] = irfft2(arr_temp_fr, (N, N), axes = (1, 2))
    return u    
    

def diff_x(u,  u_x):
    arr_mpi_r[:] = np.swapaxes(np.reshape(u, (Np,  num_process,  Np,  N)), 0, 1)
    comm.Alltoall([arr_mpi_r,  MPI.DOUBLE], [arr_temp_k,  MPI.DOUBLE])
    arr_temp_k[:] = irfft(1j * kx_diff*rfft(arr_temp_k,  axis = 0), N,  axis=0)
    comm.Alltoall([arr_temp_k,  MPI.DOUBLE], [arr_mpi_r,  MPI.DOUBLE])
    u_x[:] = np.reshape(np.swapaxes(arr_mpi_r,  0, 1), (Np,  N, N))
    return u_x

def diff_y(u, u_y):
    u_y[:] = irfft(1j*ky_diff*rfft(u, axis= 1), N, axis= 1)
    return u_y
    
def diff_z(u, u_z):
    u_z[:] = irfft(1j*kz_diff*rfft(u, axis= 2), N, axis= 2)
    return u_z

    
## -----------------------------------------------------------------

## --------------------- Other functions ---------------------------
def load_data(Ns,Nslab,u):
    for j in range(Ns):
        field = np.load(loadPath/f"time_{t[i]:.1f}/Fields_{rank*Ns+j}.npz")
        u[0,j*Nslab:(j+1)*(Nslab),...] = field["u"]
        u[1,j*Nslab:(j+1)*(Nslab),...] = field["v"]
        u[2,j*Nslab:(j+1)*(Nslab),...] = field["w"]
    return u

def energy_flux(uk,ek,PIk):
    """Calculate the 3D E_k from the given u,v,w,b and store return in ek

    Args:
        u (3*nd array): Velocity
        ek (nd array): Energy
        PI_k (nd array): Flux
    """
    
    
    u1k[:] = np.conjugate(RHS(uk,u1k))/(N**6/TWO_PI**3)**0.5
    uk[:] = uk/(N**6/TWO_PI**3)**0.5
    
    #! Insert proper normalization for zero mode and -1 mode        
    uk[...,0] = 2**(-0.5)*uk[...,0]
    u1k[...,0] = 2**(-0.5)*u1k[...,0]
    # uk[...,N//2] = 2**(-0.5)*uk[...,N//2]
    # u1k[...,N//2] = 2**(-0.5)*u1k[...,N//2]
    
    
    ek[:] = 0.5*dkx*dky*dkz*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*dealias
    PIk[:] = np.real(uk[0]*u1k[0]+uk[1]*u1k[1]+ uk[2]*u1k[2])*dkx*dky*dkz*dealias
    
    return ek, PIk


## ------------------ RHS for Boussinesq -----------------
def RHS(uk, uk_t):
    ## The RHS terms of u, v and w excluding the pressure and the hypervisocsity term 
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    
    rhsu[:] = -(u[0]*diff_x(u[0], rhsu1) + u[1]*diff_y(u[0], rhsu2) + u[2]*diff_z(u[0], rhsu3)) 
    
    rhsv[:] = -(u[0]*diff_x(u[1], rhsu1) + u[1]*diff_y(u[1], rhsu2) + u[2]*diff_z(u[1], rhsu3)) 
    
    rhsw[:] = -(u[0]*diff_x(u[2], rhsu1) + u[1]*diff_y(u[2], rhsu2) + u[2]*diff_z(u[2], rhsu3)) 
    
    rhsuk[:]  = rfft_mpi(rhsu, rhsuk)
    rhsvk[:]  = rfft_mpi(rhsv, rhsvk)
    rhswk[:]  = rfft_mpi(rhsw, rhswk)
    
    ## The pressure term
    pk[:] = 1j*invlap  * (kx*rhsuk + ky*rhsvk + kz*rhswk)
    
    

    ## The RHS term with the pressure   
    uk_t[0] = rhsuk - 1j*kx*pk
    uk_t[1] = rhsvk - 1j*ky*pk
    uk_t[2] = rhswk - 1j*kz*pk
    
    rhsu[:] = irfft_mpi(uk_t[0], rhsu)
    rhsv[:] = irfft_mpi(uk_t[1], rhsv)
    rhsw[:] = irfft_mpi(uk_t[2], rhsw)
        
    return uk_t
## -------------------------------------------------------


## ------------------ binning function ------------------- ## 
def e3d_to_e1d(x): #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
    return np.bincount(np.round(k).ravel().astype(int),weights=x.ravel()) 

## ------------------------------------------------------- ##

## ----------------- Loading the data ------------------------------
interval = 1.
t = list(np.arange(0,24.05,1.) ) +  list(np.arange(24.9,50.1,1.0))
e_arr = np.empty_like(t)
ek_arr = np.empty((len(t),Nf))
PIk_arr = np.empty((len(t),Nf))
## -------------------------------------------------------------------
if rank ==0:
    fluxes = np.zeros(len(t))
    energies = np.zeros(len(t))
## -------------------------------------------------------------------
for i in range(len(t)):
    if rank ==0: print(f"Time : {t[i]:.1f}")
    u[:] = load_data(Ns,Nslab,u)
    div[:] = diff_x(u[0],u1[0]) + diff_y(u[1],u1[1]) + diff_z(u[2],u1[2])
    if rank ==0 : print(f"Rank {rank} has divergence {np.sum(div)}")
    # if rank ==0 : print(f'np.max(np.abs(u)) : {np.max(np.abs(u))}')
    # if rank ==0 : print(f'Loading Done')
    # if rank ==0 : print(f'u_w.shape : {u_w.shape}')
    
    uk[0,:] = rfft_mpi(u[0],uk[0])
    uk[1,:] = rfft_mpi(u[1],uk[1])
    uk[2,:] = rfft_mpi(u[2],uk[2])
    
    
    
    e_arr[i] = comm.allreduce(np.sum(0.5*(u[0]**2 + u[1]**2 + u[2]**2)*dx*dy*dz),op = MPI.SUM)
    
    ek[:],PIk[:] = energy_flux(uk,ek,PIk)
    comm.Barrier()
    # print(f"Rank {rank} has k starting from {int(np.min(np.round(k)))}")
    ek_arr[i,:] = comm.allreduce(np.pad(e3d_to_e1d(ek),(int(np.min(np.round(k))),0),'constant', constant_values=(0,0))[:Nf],op = MPI.SUM)
    PIk_arr[i,:] = comm.allreduce(np.pad(e3d_to_e1d(PIk),(int(np.min(np.round(k))),0),'constant', constant_values=(0,0))[:Nf],op = MPI.SUM)
    PIk_arr[i,:] = np.cumsum(PIk_arr[i,::-1])[::-1]
    
    ## --------------- Plotting the spectra --------------- ##
    if rank ==0: 
        import matplotlib.pyplot as plt
        import matplotlib as mpl 
        mpl.rc('text', usetex = True)
        savePlot = pathlib.Path(f"/home/rajarshi.chattopadhyay/python/3D-DNS/Plots/zero-start/")
        create_dir(savePlot)
        
        plt.figure(figsize = (16,12))
        plt.xticks(fontsize = 35)
        plt.yticks(fontsize = 35)
        plt.plot(range(Nf),ek_arr[i,:],lw = 4,color = "#001219",label = fr'$t = {t[i]}$')
        plt.plot(range(Nf),1e-2*np.arange(Nf)**(-5/3),'--',lw = 2,color = "#001219",label = fr'$t = {t[i]}$')
        plt.xlabel(r"$k$",fontsize =50 )
        plt.ylabel(r"$E(k)$",fontsize =50,rotation = 0 )
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(1e-14,plt.ylim()[1])
        
        plt.savefig(savePlot/fr"Ek_time_{t[i]}.png")#,transparent = True)
        plt.close()
        
        plt.figure(figsize = (16,12))
        plt.xticks(fontsize = 35)
        plt.yticks(fontsize = 35)
        plt.plot(range(Nf),PIk_arr[i,:],lw = 4,color = "#001219",label = fr'$t = {t[i]}$')
        plt.xlabel(r"$k$",fontsize =50 )
        plt.ylabel(r"$\Pi (k)$",fontsize =50,rotation = 0 )
        # plt.yscale('log')
        # plt.xscale('log')
        plt.savefig(savePlot/fr"PIk_time_{t[i]}.png")#,transparent = True)
        plt.close()
        

    ## ---------------------------------------------------- ##   
    
    
    
    # # ------------------ Saving the data ----------------------------------------
    # if rank == 0: 
    #     create_dir(savePath/f"Ek/time_{t[i]:.1f}")
    #     create_dir(savePath/f"PIk/time_{t[i]:.1f}")
    # comm.Barrier()
    
    # np.save(savePath/f"Ek/time_{t[i]:.1f}/e_{rank}",  ek)
    # np.save(savePath/f"PIk/time_{t[i]:.1f}/PI_{rank}",  PIk)
    # comm.Barrier()  


## ------------------ Saving the data ------------- ## 
if rank == 0: 
    create_dir(savePath/f"Ek")
    create_dir(savePath/f"PIk")
    np.save(savePath/f"Ek/ek1ds",  ek_arr)
    np.save(savePath/f"PIk/PIk1ds",  PIk_arr)
comm.Barrier()
    
## ------------------------------------------------- ##

## ------------ Plotting the timeseries --------------- ## 
# if rank ==0 :
#     import matplotlib.pyplot as plt
#     import matplotlib as mpl 
#     mpl.rc('text', usetex = True)

#     plt.figure(figsize = (16,12))
#     plt.xticks(fontsize = 35)
#     plt.yticks(fontsize = 35)
#     # plt.plot(times,lw = 4,eratio,color = "#fb8500",label = 'Ratio')
#     plt.plot(t,e_arr,lw = 4,color = "#001219",label = 'Net energy')
#     # plt.plot(times,lw = 4,ekvtot_arr,color = "#fb8500",label = 'Balanced energy')
#     # plt.plot(times,lw = 4,1 - ektide/ektide[0],color = "#ffb703",label = r'$k_z = 1$ Wave')
#     plt.xlabel(r"$t$",fontsize =50 )
#     plt.ylabel(r"$E$",fontsize =50,rotation = 0 )
#     # plt.ylim(1e-6,plt.ylim()[1])
#     # plt.title(fr"Energy timeseries",fontsize = 40)
#     # plt.tight_layout()
#     plt.grid()
#     plt.legend(fontsize =50,fancybox = True, framealpha = 0.3)
#     savePlot = curr_path/f"Plots/nu_{nu}_N_{N}/Ro_{ro:.1f}/forcedTide-both"
#     try : savePlot.mkdir(parents=True, exist_ok=False)
#     except FileExistsError : pass
#     plt.savefig(savePlot/fr"energyTimeseries.png")#,transparent = True)
#     plt.close()
## ---------------------------------------------------- ##

   
    
"""
time nohup mpirun -n 128 python -u pv--energy.py > errors-outputs/pv--energy.out &
"""
    
    
