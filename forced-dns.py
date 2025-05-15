
import numpy as np 
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from mpi4py import MPI
from time import time
import pathlib,datetime, h5py,os,sys
curr_path = pathlib.Path(__file__).parent
forcestart = False

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
isforcing = True
viscosity_integrator = "implicit"
# viscosity_integrator = "explicit"
# viscosity_integrator = "exponential"
if viscosity_integrator == "explicit": isexplicit = 1.
else : isexplicit = 0.
## ---------------------------------------

## ------------- Time steps --------------
T = 30
dt =  5e-4
dt_save = 1.0
st = round(dt_save/dt)
## ---------------------------------------

## -------------Defining the grid ---------------
PI = np.pi
TWO_PI = 2*PI
N = 1024
Nf = N//2 + 1
Np = N//num_process
sx = slice(rank*Np ,  (rank+1)*Np)
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X[sx], Y, Z, indexing='ij')

Kx = Ky = Kz = fftfreq(N,  1./N)*TWO_PI/L
Kz = Kz[:Nf]
Kz[-1] = -Kz[-1]

kx,  ky,  kz = np.meshgrid(Kx,  Ky[sx],  Kz,  indexing = 'ij')
## -----------------------------------------------




## --------- kx and ky for differentiation ---------    
kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()
kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()

if rank ==0 : print(kx_diff.shape, ky_diff.shape, kz_diff.shape)

## -------------------------------------------------

## ----------- Parameters ----------
nu = 2e-3  # Hyperviscosity
lp = 1 # Hyperviscosity power
einit = 0.8*(TWO_PI)**3 # Initial velocity amplitude
shell_no = 1 # Fixing the energy of this shell with a particular profile
nshells = 1 # Number of consecutive shells to be forced


#----  Kolmogorov length scale - \eta \epsilon etc...---------

k_eta = 0.9*N//3
f0    = k_eta**4.0*nu**3.0



param = dict()
param["nu"] = nu
param["hyperviscous"] = lp
param["Initial p amplitude"] = einit
param["Gridsize"] = N
param["Processes"] = num_process
param["Final_time"] = T
param["time_step"] = dt
param["interval of saving indices"] = st

## ---------------------------------

# savePath = pathlib.Path(f"/home/rajarshi.chattopadhyay/python/3D-DNS/data/samriddhi-tests-euler-spherical-dealias-final/N_{N}")

if nu!= 0: savePath = pathlib.Path(f"./data/forced_{isforcing}/N_{N}_Re_{1/nu:.1f}")
else: savePath = pathlib.Path(f"./data/forced_{isforcing}/N_{N}_Re_inf")

if rank == 0:
    print(savePath)
    try: savePath.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass

## ------------Useful Operators-------------------

lap = -(kx**2 + ky**2 + kz**2 )
k = (-lap)**0.5
kint = np.clip(np.round(k,0).astype(int),None,N//2)
# kh = (kx**2 + ky**2)**0.5
dealias = kint<=N/3 #! Spherical dealiasing
# dealias = (abs(kx)<N//3)*(abs(ky)<N//3)*(abs(kz)<N//3)
invlap = dealias/np.where(lap == 0, np.inf,  lap)

# Hyperviscous operator
vis = nu*(k)**(2*lp) ## This is in Fourier Space

# Forcing length scales
# cond = (kh>0.5)*(kh<2.5)
normalize = np.where((kz== 0) + (kz == N//2) , 1/(N**6/TWO_PI**3),2/(N**6/TWO_PI**3))
shells = np.arange(-0.5,Nf, 1.)
shells[0] = 0.

## -------------------------------------------------


## -------------zeros arrays -----------------------
u  = np.zeros((3, Np, N, N), dtype= np.float64)
omg= np.zeros((3, Np, N, N), dtype= np.float64)
p = u[0].copy()


uk = np.zeros((3, N, Np, Nf), dtype= np.complex128)
pk = uk[0].copy()
bk = pk.copy()
ek = np.zeros_like(pk, dtype = np.float64)
Pik = np.zeros_like(pk, dtype = np.float64)
ek_arr = np.zeros(Nf)
Pik_arr = np.zeros(Nf)
factor = np.zeros(Nf)
factor3d = np.zeros_like(pk,dtype= np.float64)
uknew = np.zeros_like(uk)


# theta2 = np.zeros((2,N, N, Np), dtype= np.complex128)
# theta1 = np.zeros((2,N, N, Np), dtype= np.complex128)
# exptheta1w = np.zeros_like(pk)
# exptheta2w = np.zeros_like(pk)

fk = np.zeros_like(uk)
ff = np.zeros_like(u)

rhsuk = np.zeros_like(pk)
rhsvk = rhsuk.copy()
rhswk = rhsuk.copy()

rhsu = np.zeros_like(p)
rhsv = rhsu.copy()
rhsw = rhsu.copy()

rhsu1 = np.zeros_like(p)
rhsu2 = np.zeros_like(p)
rhsu3 = np.zeros_like(p)


k1u = np.zeros((3, N, Np, Nf), dtype = np.complex128)
k2u = np.zeros((3, N, Np, Nf), dtype = np.complex128)
k3u = np.zeros((3, N, Np, Nf), dtype = np.complex128)
k4u = np.zeros((3, N, Np, Nf), dtype = np.complex128)

arr_temp_k = np.zeros((N, Np, N),dtype= np.float64)
arr_temp_fr = np.zeros((Np, N, Nf), dtype= np.complex128)      
arr_temp_ifr = np.zeros((N, Np, Nf), dtype= np.complex128)      
arr_mpi = np.zeros((num_process,  Np,  Np, Nf), dtype= np.complex128)
arr_mpi_r = np.zeros((num_process,  Np,  Np, N), dtype= np.float64)


## -----------------------------------------------------


## ------FFT + iFFT + derivative functions------- 
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
    arr_mpi_r[:] = np.moveaxis(np.reshape(u, (Np,  num_process,  Np,  N)),[0,1], [1,0])
    comm.Alltoall([arr_mpi_r,  MPI.DOUBLE], [arr_temp_k,  MPI.DOUBLE])
    arr_temp_k[:] = irfft(1j * kx_diff*rfft(arr_temp_k,  axis = 0), N,  axis=0)
    comm.Alltoall([arr_temp_k,  MPI.DOUBLE], [arr_mpi_r,  MPI.DOUBLE])
    u_x[:] = np.reshape(np.moveaxis(arr_mpi_r,  [0,1], [1,0]), (Np,  N, N))
    return u_x

def diff_y(u, u_y):
    u_y[:] = irfft(1j*ky_diff*rfft(u, axis= 1), N, axis= 1)
    return u_y
    
def diff_z(u, u_z):
    u_z[:] = irfft(1j*kz_diff*rfft(u, axis= 2), N, axis= 2)
    return u_z

def e3d_to_e1d(x): #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
    return np.histogram(k.ravel(),bins = shells,weights=x.ravel())[0] 

    
def forcing(uk,fk):
    """
    Calculates the net dissipation of the flow and injects that amount into larges scales of the horizontal flow
    """
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*normalize #! This is the 3D ek array
    
    ek_arr[:] = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek array.
    ek_arr[:] = np.where(np.abs(ek_arr)< 1e-10,np.inf, ek_arr)
    
    """Change forcing starts here"""
    ## Const Power Input
    # factor[:] = 0.
    # factor[shell_no] = f0/(2*ek_arr[shell_no])
    # factor3d[:] = factor[kint]
    
    
    # Constant shell energy
    factor[:] = np.tanh(np.where(np.abs(ek_arr0) < 1e-10, 0, (ek_arr0/ek_arr)**0.5 - 1)) #! The factors for each shell is calculated
    factor3d[:] = factor[kint]
    
    fk[0] = factor3d*uk[0]*dealias
    fk[1] = factor3d*uk[1]*dealias
    fk[2] = factor3d*uk[2]*dealias

    """Change forcing ends here here"""
    
    pk[:] = invlap  * (kx*fk[0] + ky*fk[1] + kz*fk[2])*dealias
    
    fk[0] = fk[0] + kx*pk
    fk[1] = fk[1] + ky*pk
    fk[2] = fk[2] + kz*pk
    
    
    return fk*isforcing*dealias
    
     



def RHS(uk, uk_t):
    ## The RHS terms of u, v and w excluding the forcing and the hypervisocsity term 
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    
    omg[0] = irfft_mpi(1j*(ky*uk[2] - kz*uk[1]),omg[0])
    omg[1] = irfft_mpi(1j*(kz*uk[0] - kx*uk[2]),omg[1])
    omg[2] = irfft_mpi(1j*(kx*uk[1] - ky*uk[0]),omg[2])
    
    rhsu[:] = (omg[2]*u[1] - omg[1]*u[2])
    rhsv[:] = (omg[0]*u[2] - omg[2]*u[0])
    rhsw[:] = (omg[1]*u[0] - omg[0]*u[1]) 
    
    
    rhsuk[:]  = rfft_mpi(rhsu, rhsuk)*dealias 
    rhsvk[:]  = rfft_mpi(rhsv, rhsvk)*dealias 
    rhswk[:]  = rfft_mpi(rhsw, rhswk)*dealias 
    
    ## The pressure term
    pk[:] = 1j*invlap  * (kx*rhsuk + ky*rhsvk + kz*rhswk)
    
    

    ## The RHS term with the pressure   
    uk_t[0] = rhsuk - 1j*kx*pk + nu*lap*uk[0]*isexplicit 
    uk_t[1] = rhsvk - 1j*ky*pk + nu*lap*uk[1]*isexplicit 
    uk_t[2] = rhswk - 1j*kz*pk + nu*lap*uk[2]*isexplicit
    

        
    return uk_t




## -----------------------------------------------------------


## ---------------- Saving data + energy + Showing total energy ---------------------

def save(i,uk):
    
    # div = diff_x(u[0], rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw)
    # if rank == 0: print(f"Rank {rank} has divergence {np.sum(np.abs(div))}")
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*normalize #! This is the 3D ek array
    ek_arr[:] = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek array.
    
    k1u[:] = RHS(uk, k1u)
    Pik[:] = np.real(np.conjugate(uk[0])*k1u[0]+np.conjugate(uk[1])*k1u[1]+ np.conjugate(uk[2])*k1u[2])*dealias*normalize
    Pik_arr[:] = comm.allreduce(e3d_to_e1d(Pik),op = MPI.SUM)
    Pik_arr[:] = np.cumsum(Pik_arr[::-1])[::-1]
    
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    # ----------- ----------------------------
    #                 Saving the data (field)
    # ----------- ----------------------------
    new_dir = savePath/f"time_{t[i]:.1f}"
    try: new_dir.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass
    
    comm.Barrier()
    np.savez_compressed(f"{new_dir}/Fields_{rank}",u = u[0],v = u[1],w = u[2])
    np.savez_compressed(f"{new_dir}/Energy_spectrum",ek = ek_arr)
    np.savez_compressed(f"{new_dir}/Flux_spectrum",Pik = Pik_arr)
    
    comm.Barrier()
    
    # ----------- ----------------------------
    #          Calculating and printing
    # ----------- ----------------------------
    eng1 = comm.allreduce(np.sum(0.5*(u[0]**2 + u[1]**2 + u[2]**2)*dx*dy*dz), op = MPI.SUM)
    eng2 = np.sum(ek_arr)
    divmax = comm.allreduce(np.max(np.abs(diff_x(u[0],  rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw))),op = MPI.MAX)
    #! Needs to be changed 
    # # dissp = -nu*comm.allreduce(np.sum((kc**(2*lp)*(np.abs(uk[0])**2 + np.abs(uk[1])**2) +sin_to_cos( ks**(2*lp)*(np.abs(uk[2])**2/alph**2 + np.abs(bk)**2)))), op = MPI.SUM)
    if rank == 0:
        print( "#----------------------------","\n",f"Energy at time {t[i]} is : {eng1}, {eng2}","\n","#----------------------------")
        print(f"Maximum divergence {divmax}")
        # print( "#----------------------------","\n",f"Total dissipation at time {t[i]} is : {dissp}","\n","#----------------------------")
        # print( "#----------------------------","\n",f" Rms field value at time {t[i]} is : {sqfld}","\n","#----------------------------")
    return "Done!"    
## -------------------------------------------------    
    
## ------------- Evolving the system ----------------- 
def evolve_and_save(t,  u): 
    global begin
    h = t[1] - t[0]
    
    if viscosity_integrator == "implicit": hypervisc= dealias*(1. + h*vis)**(-1)
    else: hypervisc = 1.
    
    if  viscosity_integrator == "exponential": 
        semi_G =  np.exp(-nu*(k**(2*lp))*h)
        semi_G_half =  semi_G**0.5
    else: semi_G = semi_G_half = 1.
    
    t3  = time()
    for i in range(t.size-1):
        # if rank == 0:  print(f"step {i} in time {time() - t3}", end= '\r')
        ## ------------- saving the data -------------------- ##
        if i % st ==0 :
            save(i,uk)
        begin = True   
        ## -------------------------------------------------- ##
        t3 = time()
        
        fk[:] = forcing(uknew,fk)
        
        k1u[:] = RHS(uk, k1u)
        # k2u[:] = RHS(uk + h*k1u ,k2u) #! Only for RK2
        k2u[:] = RHS(semi_G_half*(uk + h/2.*k1u) ,k2u)
        k3u[:] = RHS(semi_G_half*uk + h/2.*k2u, k3u)
        k4u[:] = RHS(semi_G*uk + semi_G_half*h*k3u, k4u)
        
        # uknew[:] = uk + h/2.0* ( k1u + k2u )
        uknew[:] = (semi_G*uk + h/6.0* ( semi_G*k1u + 2*semi_G_half*(k2u + k3u) + k4u)  + h*fk)*hypervisc
        # uknew[:] = (uknew + h*fk)
        
        # comm.Barrier()
        uk[:] = uknew
  
        # ------------------------------------- #
        
        
        # divmax = np.max(np.abs(diff_x(unew[0], p11) + diff_y(unew[1], p12) + diff_z_sin(unew[2], p13)))
        # if rank > 0: 
        #     comm.send(divmax,dest= 0, tag = 112)
        # else:
        #     for j in range(1, num_process):
        #         divmax = max(divmax, comm.recv(source = j, tag = 112))
        #     print(f"max of div  after forcing: {divmax}")
            
        # comm.Barrier()
        
 
        ## -------------------------------------------------------
        if uknew.max() > 100*N**3 : 
            print("Threshold exceeded at time", t[i+1], "Code about to be terminated")
            comm.Abort()
        
        
        comm.Barrier()
        
    ## ---------- Saving the final data ------------
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*normalize #! This is the 3D ek array
    ek_arr[:] = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek array.
    save(i+1, uk)
    ## ---------------------------------------------

    

## --------------- Initializing ---------------------


"""Structure 
If there exists a folder with the parameter names and has time folders in it. 
Load the parameters from parameters.txt
If the parameters match the current code parameters enter the last time folder.
Finally load the data.
If not start from scratch."""



#! Modify the loading process!

if not forcestart:
    ## ------------------------- Beginning from existing data -------------------------
    if rank ==0 : print("Found existing simulation! Using last saved data.")
    """Loading the data from the last time  """    
    paths = sorted([x for x in (savePath).iterdir() if "time_" in str(x)], key=os.path.getmtime)
    """The folder is paths[-1]"""
    paths = paths[-1]
    if rank ==0 : print(f"Loading data from {paths}")
    # # tinit = float(str(paths).split("time_")[-1])
    tinit = 22.0
    
    # load_num_slabs = 512
    load_num_slabs = len([x for x in (paths).iterdir() if "Fields" in str(x)])
    data_per_rank = N//load_num_slabs
    rank_data = range(rank*Np,(rank + 1)*Np) # The rank contains these slices 
    slab_old = np.inf
    for lidx,j in enumerate(rank_data):
        slab = j//data_per_rank
        idx = j%data_per_rank
        # print(f"Rank {rank} is loading slab {slab} and idx {idx}")
        if slab_old != slab:  Field = np.load(savePath/f"time_{tinit:.1f}/Fields_{slab}.npz")
        slab_old = slab
        u[0,lidx] = Field['u'][idx]
        u[1,lidx] = Field['v'][idx]
        u[2,lidx] = Field['w'][idx]
    del paths
    comm.Barrier()
    if rank ==0: print("Data loaded successfully")
    
    uk[0] = rfft_mpi(u[0], uk[0])
    uk[1] = rfft_mpi(u[1], uk[1])
    uk[2] = rfft_mpi(u[2], uk[2])
    

if forcestart:
    ## ---------------------- Beginning from start ----------------------------------

    kinit = 31 # Wavenumber of maximum non-zero initial pressure mode.    
    thu = np.random.uniform(0, TWO_PI,  k.shape)
    thv = np.random.uniform(0, TWO_PI,  k.shape)
    thw = np.random.uniform(0, TWO_PI,  k.shape)

    # eprofile = 1/np.where(kint ==0, np.inf,kint**(2.0))/normalize
    eprofile = kint**2*np.exp(-kint**2/2)/normalize
    
    
    amp = (eprofile/np.where(kint == 0, np.inf, kint**2))**0.5
    
    uk[0] = amp*np.exp(1j*thu)*(kint**2<kinit**2)*(kint>0)*dealias
    uk[1] = amp*np.exp(1j*thv)*(kint**2<kinit**2)*(kint>0)*dealias
    uk[2] = amp*np.exp(1j*thw)*(kint**2<kinit**2)*(kint>0)*dealias
    
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    
    uk[0] = rfft_mpi(u[0],uk[0])
    uk[1] = rfft_mpi(u[1],uk[1])
    uk[2] = rfft_mpi(u[2],uk[2])
    
    trm = (kx*uk[0]  + ky*uk[1] + kz*uk[2])
    uk[0] = uk[0] + invlap*kx*trm
    uk[1] = uk[1] + invlap*ky*trm
    uk[2] = uk[2] + invlap*kz*trm
    
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*normalize #! This is the 3D ek array
    ek_arr0 = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek a
    # if rank ==0: print(ek_arr0, np.sum(ek_arr0))
    
    uk[0] = uk[0] *(einit/ek_arr0[shell_no])**0.5
    uk[1] = uk[1] *(einit/ek_arr0[shell_no])**0.5
    uk[2] = uk[2] *(einit/ek_arr0[shell_no])**0.5
    
    
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    
    tinit = 0.


ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)*normalize #! This is the 3D ek array
ek_arr0 = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek a
if rank ==0: print(ek_arr0, np.sum(ek_arr0))

ek_arr0[0:shell_no] = 0.
ek_arr0[shell_no+ nshells:] = 0.


divmax = comm.allreduce(np.max(np.abs( diff_x(u[0],  rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw))),op = MPI.MAX)
if rank ==0 : print(f" max divergence {divmax}")

#----------------- The initial energy ------------------
e0 = comm.allreduce(0.5*dx*dy*dz*np.sum(u[0]**2 + u[1]**2 + (u[2]**2)),op = MPI.SUM)
if rank ==0 : print(f"Initial Physical space energy: {np.sum(e0)}")
#-------------------------------------------------------

# --------------------------------------------------

## ----- executing the code -------------------------
t = np.arange(tinit,T+ 0.5*dt, dt)
# print(len(t))
t1 = time()
evolve_and_save(t,u)
t2 = time() - t1 
# --------------------------------------------------
if rank ==0: print(t2)
## --------- saving the calculation time -----------
if rank ==0: 
    with open(savePath/f"calcTime.txt","a") as f:
        f.write(str({f"time taken to run from {tinit} to {T} is": t2}))
## --------------------------------------------------
