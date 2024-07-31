import numpy as np
from scipy.fft import fftn,ifftn,fftshift
import matplotlib.pyplot as plt
# from numba import njit, prange, jit
from time import time
import sys, concurrent.futures

N = 128
PI = np.pi
TWO_PI = 2*PI
p = float(sys.argv[-1])

X = Y =Z = np.linspace(0,TWO_PI,N,endpoint=False)
x,y,z = np.meshgrid(X,Y,Z, indexing='ij')
idxs = np.arange(N)
vec = (np.array(np.meshgrid(idxs,idxs,idxs,indexing='ij'))).astype(int) - N//2
norms = np.linalg.norm(vec,axis=0)
g = np.zeros((N,N,N))
kx,ky,kz = np.random.randint(1,N//3,(3,))
# u = np.array([np.sin(3*x)*np.cos(5*y)*np.sin(12*z),np.cos(kx*x)*np.sin(ky*y)*np.sin(kz*z),np.cos(kx*x)*np.cos(ky*y)*np.cos(kz*z)])
# u = np.random.random((3,N,N,N))
# u = u - u.mean()
u = np.zeros((3,N,N,N))
num_slab = 32
Ns = N//num_slab
t = 30.
nu = 1/200
isforcing = True
for i in range(num_slab):
    Field = np.load(f"/mnt/pfs/rajarshi.chattopadhyay/3D-DNS/data/forced_{isforcing}/N_{N}_Re_{1/nu:.1f}/time_{t:.1f}/Fields_{i}.npz")
    u[0,i*Ns:(i+1)*Ns,:,:] = Field['u']
    u[1,i*Ns:(i+1)*Ns,:,:] = Field['v']
    u[2,i*Ns:(i+1)*Ns,:,:] = Field['w']
print(f"Data loaded for time {t}")
diffu = np.zeros((3,N,N,N))
cond = np.zeros(N**3,dtype=bool)
diff = np.zeros(N//2+1)
shells = np.arange(-0.5,N//2+1,1)
shells[0] = 0.
# print(shells.size)
rhat = vec/np.where(norms ==0 , np.inf,norms)
# vec[:] = N/2
# If vec is correct, g seems to be correct
# @njit(parallel=True)

def work(x):
    # if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers = 32) as executor: 
        scalc = executor.map(sp, [pos for pos in x.T])
    return np.array(list(scalc)).reshape((N,N,N))
            
def sp(shift,u = u,p = p):
    # global diffu
    # for i in range(N):
    #     for j in range(N):
    #         for k in range(N):
        # shift = vec[:,i,j,k]
    norm = np.linalg.norm(shift)
    rcap = shift/np.where(norm ==0 , np.inf,norm)
    # print(shift.shape, diffu.shape)
    diffu[:] = np.roll(u,shift,axis=(1,2,3)) - u
    # u1 = np.roll(u,shift,axis=(1,2,3))
    diff = np.abs((diffu[0]*rcap[0] + diffu[1]*rcap[1] + diffu[2]*rcap[2])**p).sum()/N**3  #- (2* (u[0]*rcap[0] + u[1]*rcap[1] + u[2]*rcap[2])**p).sum()/N**3
        #! Just for testing the shift
        # diffu[0] = np.roll(u,shift,axis=(0,1,2)) - u
        # g[i,j,k] = (diffu[0]**p - 2*u**p).sum()
    return diff

def e3d_to_1d(x):
    return np.histogram(norms.ravel(),bins = shells,weights= x.ravel())[0]/np.histogram(norms.ravel(),bins = shells)[0]

# sp(np.zeros((2,2,2)),2,np.zeros((2,2,2)),np.zeros((3,2,2,2)),np.zeros((3,2,2,2)),2)
# print("Initialization done")
t1 = time()
if __name__ == '__main__': g[:] = work(vec.reshape((3,N**3)))
t2 = time()
print(f"Time taken = {t2 - t1:.2f} seconds")
diff[:] = e3d_to_1d(g)
np.save(f'struct-func-{p}.npy',diff)
plt.plot(np.arange(N//2+1),diff,'.-')
plt.xlabel(r'$r$')
plt.ylabel(fr'$S_{p:.0f}(r)$',rotation = 0,labelpad = 20)
# plt.xscale('log')
plt.savefig(f'sturct-func-{p}.png')
# plt.yscale('log')
# g[:] = sp(f[0],p,g,vec,diffu,N)
# fk = fftn(f[0])
# efk = np.abs(fk)**2
# ef = ifftn(efk).real
# print(np.max(np.abs(fftshift(g,axes = (0,1,2)) + 2*ef)),np.max(np.abs(2*ef)))
