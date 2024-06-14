import numpy as np
from scipy.fft import fftn,ifftn,fftshift
import matplotlib.pyplot as plt
from numba import njit, prange, jit

N = 30
PI = np.pi
TWO_PI = 2*PI
p = 2.0

X = Y =Z = np.linspace(0,TWO_PI,N,endpoint=False)
x,y,z = np.meshgrid(X,Y,Z, indexing='ij')
idxs = np.arange(N)
vec = (np.array(np.meshgrid(idxs,idxs,idxs,indexing='ij'))).astype(int) - N//2
norms = np.linalg.norm(vec,axis=0)
g = np.zeros((N,N,N))
kx,ky,kz = np.random.randint(1,N//3,(3,))
# f = np.array([np.sin(3*x)*np.cos(5*y)*np.sin(12*z),np.cos(kx*x)*np.sin(ky*y)*np.sin(kz*z),np.cos(kx*x)*np.cos(ky*y)*np.cos(kz*z)])
f = np.random.random((3,N,N,N))
f = f - f.mean()
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
def sp(u,p,g,vec,diffu,N):
    for i in prange(N):
        for j in prange(N):
            for k in prange(N):
                shift = vec[:,i,j,k]
                rcap = rhat[:,i,j,k]
                diffu[:] = np.roll(u,shift,axis=(1,2,3)) - u
                u1 = np.roll(u,shift,axis=(1,2,3))
                g[i,j,k] = ((diffu[0]*rcap[0] + diffu[1]*rcap[1] + diffu[2]*rcap[2])**p).sum()/N**3  - ((u1[0]*rcap[0] + u1[1]*rcap[1] + u1[2]*rcap[2])**p + (u[0]*rcap[0] + u[1]*rcap[1] + u[2]*rcap[2])**p).sum()/N**3
                #! Just for testing the shift
                # diffu[0] = np.roll(u,shift,axis=(0,1,2)) - u
                # g[i,j,k] = (diffu[0]**p - 2*u**p).sum()
    return g

def e3d_to_1d(x):
    return np.histogram(norms.ravel(),bins = shells,weights= x.ravel())[0]/np.histogram(norms.ravel(),bins = shells)[0]

# sp(np.zeros((2,2,2)),2,np.zeros((2,2,2)),np.zeros((3,2,2,2)),np.zeros((3,2,2,2)),2)
# print("Initialization done")
g[:] = sp(f,p,g,vec,diffu,N)
diff[:] = e3d_to_1d(g)
plt.plot(np.arange(N//2+1),diff,'.-')
plt.xlabel('r')
plt.ylabel('S(r)')
plt.savefig(f'velocity-increment_p_{p}.png')
plt.xscale('log')
# plt.yscale('log')
# g[:] = sp(f[0],p,g,vec,diffu,N)
# fk = fftn(f[0])
# efk = np.abs(fk)**2
# ef = ifftn(efk).real
# print(np.max(np.abs(fftshift(g,axes = (0,1,2)) + 2*ef)),np.max(np.abs(2*ef)))
