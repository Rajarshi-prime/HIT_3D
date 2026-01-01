#%%
import numpy as np
from scipy.fft import fft,ifft, rfft, irfft, rfftn, irfftn,fftfreq,fftshift
import matplotlib.pyplot as plt

#%%
N = 128

PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
Np = N
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X, Y, Z, indexing='ij')
#%%
Kx = Ky = fftfreq(N,  1./N)*TWO_PI/L
Kz = np.abs(Ky[:Nf])
#%%
kvec = np.meshgrid(Kx,  Ky,  Kz,  indexing = 'ij')
kx, ky,kz = kvec[0],kvec[1],kvec[2]
lap = -1.0*(kx**2 + ky**2 + kz**2 )
k = (-lap)**0.5
kint = np.clip(np.round(k,0).astype(int),None,N//2)
#%%
"""Testing procedure
1. Simple product whre dealiasing is not necessary. 
2. Case where phase shifted keeps the full function, but the  N//3 does not. 
3. Where both fails but we have explicit convoluton.
"""
m, n  = 24,19
f = np.cos(m*x)*np.sin(2*z)
g = np.cos(n*x)

dealias = kint<=2**0.5*N/3 #! Phase-shift dealiasing
phase_k = np.exp(1j*(kx + ky + kz)*(L/(2*N)))

#%%
"""Calcualte the nonlinear term u times omega and u.grad(b) pseudospectrally using phase-shifted dealiasing ref. Canuto et al. 2006,2007, and Patterson and Orszag 1971"""


fk = rfftn(f,axes = (-3,-2,-1))
gk = rfftn(g,axes = (-3,-2,-1))   

prod = f*g
prodk = rfftn(prod,axes = (-3,-2,-1))*dealias/2.

f[:] = irfftn(fk*dealias*phase_k,(N,N,N),axes = (-3,-2,-1))
g[:] = irfftn(gk*dealias*phase_k,(N,N,N),axes = (-3,-2,-1))

prod[:] = f*g

prodk += rfftn(prod,axes = (-3,-2,-1))*(1/phase_k)*dealias/2. 


prod[:] = irfftn(prodk,(N,N,N),axes = (-3,-2,-1))
#%%
np.abs(prod - np.cos(m*x)*np.cos(n*x)).max()
# %%
f = np.cos(m*x)*np.sin(2*z)
g = np.cos(n*x)

dealias1 = kint<=N//3 #! N//3 dealiasing
prod1 = f*g
prodk1 = rfftn(prod1,axes = (-3,-2,-1))*dealias1
prod1[:] = irfftn(prodk1,(N,N,N),axes = (-3,-2,-1))
# %%
np.abs(prod1 - np.cos(m*x)*np.cos(n*x)).max()
# %%

plt.plot(prod[:,0,0])
plt.plot(prod1[:,0,0])
plt.plot((np.cos(m*x)*np.cos(n*x))[:,0,0])

#%%
normalize = 1/N**6 #! kz == 0 need not be separated out because the convolution misses that as well. 
kxint = np.round(kx).astype(np.int32)
kyint = np.round(ky).astype(np.int32)
kzint = np.round(kz).astype(np.int32)
# %%
convolutionk = ((2**0.5*N//3> abs(n-m))*((kxint + n - m == 0)*1/4 + (kxint  == n - m)*1/4) + (2**0.5*N//3> abs(n+m))*((kxint + n + m==0)*1/4 + (kxint  - n - m == 0)*1/4))*(kyint==0)*((kzint==2)*(-0.5j) + (kzint == -2)*0.5j)/normalize**0.5
convolution = irfftn(convolutionk,(N,N,N),axes = (-3,-2,-1))
# %%
"""For the N//3 dealising"""



convolutionk1 = ((N//3> abs(n-m))*((kxint + n - m == 0)*1/4 + (kxint  == n - m)*1/4) + (N//3> abs(n+m))*((kxint + n + m==0)*1/4 + (kxint  - n - m == 0)*1/4))*(kyint==0)*((kzint==2)*(-0.5j) + (kzint == -2)*0.5j)/normalize**0.5
convolution1 = irfftn(convolutionk1,(N,N,N),axes = (-3,-2,-1))
# %%
plt.plot(fftshift(Kx),fftshift(np.real(convolutionk1[:,0,2])),'s')
plt.plot(fftshift(Kx),fftshift(np.real((prodk1)[:,0,2])),'^')
plt.xlim(-N//3,N//3)

#%%
plt.plot(fftshift(Kx),fftshift(np.imag(convolutionk[:,0,2]))/N**3,'*')
plt.plot(fftshift(Kx),fftshift(np.imag((prodk)[:,0,2]))/N**3,'x')
#  # %%
# %%
2**0.5*360//3
# %%
np.round(Kz).astype(np.int32)  == -n + m
# %%

def convolve(fk,gk):

    f[:] = irfftn(fk*dealias,(N,N,N),axes = (-3,-2,-1))
    g[:] = irfftn(gk*dealias,(N,N,N),axes = (-3,-2,-1))

    prod = f*g
    prodk = rfftn(prod,axes = (-3,-2,-1))*dealias/2.

    f[:] = irfftn(fk*dealias*phase_k,(N,N,N),axes = (-3,-2,-1))
    g[:] = irfftn(gk*dealias*phase_k,(N,N,N),axes = (-3,-2,-1))

    prod[:] = f*g
    prodk += rfftn(prod,axes = (-3,-2,-1))*(1/phase_k)*dealias/2. 


    prod[:] = irfftn(prodk,(N,N,N),axes = (-3,-2,-1))