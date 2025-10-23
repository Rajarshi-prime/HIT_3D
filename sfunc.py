import numpy as np
from scipy.fft import fftn,ifftn,fftshift
import matplotlib.pyplot as plt
from numba import njit, prange, jit
from time import time
import sys,pathlib,os #, concurrent.futures


N = 1024
sysidx = float(sys.argv[-1])
lent = 11
Ti = 1.0
Tf = 5.0
p = 1.0*(sysidx//lent) + 1.0
t = 0.4*float(sysidx%lent)+ Ti
nu = 0.
isforcing = False

""" If the number of vectors for a given r is more than 100000, we will keep randomly chosen 100000 of them"""
maxrhats = 100000
maxcenters = 100000


PI = np.pi
TWO_PI = 2*PI

X = Y =Z = np.linspace(0,TWO_PI,N,endpoint=False)
dx = dy = dz = X[1] - X[0]
x,y,z = np.meshgrid(X,Y,Z, indexing='ij')
idxs = np.arange(N)
vec = ((np.array(np.meshgrid(idxs,idxs,idxs,indexing='ij'))).astype(int) - N//2).reshape((3,N**3))
norms = np.linalg.norm(vec,axis=0)


centres = np.random.randint(0,N**3,(maxcenters))
newvecs = np.zeros((3,0))
cond = np.zeros((N**3,),dtype=bool)
cidx = np.zeros((maxrhats,),dtype=int)
for r in range(N//2+1):
    cond[:] = (norms > r - 0.5)*(norms < r + 0.5)
    if cond.sum() > maxrhats: 
        idx = np.argwhere(cond).ravel()
        cidx = np.random.choice(idx, maxrhats, replace=False)
        newvecs = np.append(newvecs, vec[:,cidx],axis=1)
    else:
        idx = np.argwhere(cond).ravel()
        newvecs = np.append(newvecs, vec[:,idx],axis=1)

del vec,norms
vec = newvecs
del newvecs
""" Now, we can use this new vecs """
Nvec = vec.shape[1]
norms = np.linalg.norm(vec,axis=0)
print(f"Number of vectors = {Nvec}")

# g = np.zeros((N,N,N))
# kx,ky,kz = np.random.randint(1,N//3,(3,))
# u = np.array([np.sin(3*x)*np.cos(5*y)*np.sin(12*z),np.cos(kx*x)*np.sin(ky*y)*np.sin(kz*z),np.cos(kx*x)*np.cos(ky*y)*np.cos(kz*z)])
# u = np.random.random((3,N,N,N))
# u = u - u.mean()
loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/3D-DNS/data_new_new/forced_{isforcing}/N_{N}_Re_inf/time_{t:.1f}")
num_slabs = len([x for x in (loadPath).iterdir() if "Fields" in str(x)])
u = np.zeros((3,N,N,N))
Ns = N//num_slabs
for i in range(num_slabs):
    Field = np.load(loadPath/ f"Fields_{i}.npz")
    u[0,i*Ns:(i+1)*Ns,:,:] = Field['u']
    u[1,i*Ns:(i+1)*Ns,:,:] = Field['v']
    u[2,i*Ns:(i+1)*Ns,:,:] = Field['w']
print(f"Data loaded for time {t}")
# diffu = np.zeros((3,N,N,N))
# cond = np.zeros(N**3,dtype=bool)
diff = np.zeros(N//2+1)
shells = np.arange(-0.5,N//2+1,1)
shells[0] = 0.
print(np.histogram(norms.ravel(),bins = shells)[0])
# print(shells.size)
# rhat = vec/np.where(norms ==0 , np.inf,norms)
# vec[:] = N/2
# If vec is correct, g seems to be correct
@njit(parallel=True)
def sp(u,vec,p,N,Nvec,centres):
    diffu = np.zeros((3,len(centres)))
    g = np.zeros((Nvec))
    # for i in prange(N**3):
    for ind1 in prange(Nvec):
        # ind1 = int(i%N)
        # ind2 = int((i//N)%N)
        # ind3 = int((i//N)//N)
        shift = vec[:, ind1]
        norm = (shift[0]**2 + shift[1]**2 + shift[2]**2)**0.5
        rcap = shift/np.where(norm ==0 , np.inf,norm)
        # for j in prange(N**3):
        for ii in prange(len(centres)):
            ii1 = int(centres[ii]%N)
            ii2 = int((centres[ii]//N)%N)
            ii3 = int((centres[ii]//N)//N)
            diffu[:,ii] = u[:,int((ii1 - shift[0])%N),int((ii2 - shift[1])%N),int((ii3 - shift[2])%N)] - u[:,ii1,ii2,ii3]
            g[ind1]  +=  (np.abs(diffu[0,ii]*rcap[0] + diffu[1,ii]*rcap[1] + diffu[1,ii]*rcap[2])**p )
            
    return g/(len(centres))

def e3d_to_1d(x):
    return np.histogram(norms.ravel(),bins = shells,weights= x.ravel())[0]/np.histogram(norms.ravel(),bins = shells)[0]

sp(np.random.random((3,4,4,4)),np.random.random((3,64)),p,4,64,np.random.randint(0,4**3,(16)))
print("Initialization done")
t1 = time()
g = sp(u,vec,p,N,Nvec,centres)
t2 = time()
print(f"Time taken = {t2 - t1:.2f} seconds")
diff[:] = e3d_to_1d(g)
savePath = pathlib.Path(f'/mnt/pfs/rajarshi.chattopadhyay/codes/3D-DNS/pstprc_data/time_{t:.1f}/N_{N}/')
savePath.mkdir(parents=True, exist_ok=True)
np.savez_compressed(savePath/f"struct_funct_{p}.npz",diff = diff)

