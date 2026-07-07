# API Reference

## Core Functions

### FFT Operations (MPI-parallel)

#### `rfft_mpi(u, fu)`
Real-valued FFT with MPI decomposition.

**Args:**
- `u`: physical space field (float64, shape: [Np, N, N])
- `fu`: Fourier space output (complex128, shape: [N, Np, Nf])

**Returns:** Fourier coefficients in Hermitian format

**Note:** Handles MPI communication internally. Decomposes over y-direction.

#### `irfft_mpi(fu, u)`
Inverse real FFT with MPI.

**Args:**
- `fu`: Fourier coefficients (complex128)
- `u`: physical space output (float64)

**Returns:** Real-valued physical space field

---

### Spatial Derivatives

#### `diff_x(u, u_x)`, `diff_y(u, u_y)`, `diff_z(u, u_z)`
Compute ∂u/∂x, ∂u/∂y, ∂u/∂z using spectral differentiation.

**Method:** FFT → multiply by ikα → iFFT

**Args:**
- `u`: field in physical space
- `u_x`: output array for derivative

**Returns:** Spatial derivative

**Example:**
```python
# Compute ∂(u[0])/∂x
du_dx = np.zeros_like(u[0])
diff_x(u[0], du_dx)
```

---

### Spectral Analysis

#### `e3d_to_e1d(x)`
Bin 3D field into 1D radial energy spectrum.

**Args:**
- `x`: 3D field (shape: [Np, N, N])

**Returns:** 1D spectrum binned in wavenumber shells (shape: [Nf])

**Physics:** Sums energy over spherical shells in k-space. Result: E(k).

**Example:**
```python
ek_1d = e3d_to_e1d(ek_3d)  # 3D → 1D energy spectrum
```

---

## Physics Functions

### `forcing(uk, fk)`
Inject energy at specified wavenumber shells.

**Args:**
- `uk`: velocity in Fourier space (3 components)
- `fk`: output forcing term

**Mechanism:**
1. Calculates energy E(k) in each shell
2. Scales velocity to maintain constant power input f0
3. Enforces divergence-free condition (∇·u = 0)

**Parameters:**
- `f0`: power input density
- `shell_no`: which shells to force (e.g., [1, 2])

**Returns:** Forcing term in Fourier space

**Physics:** Maintains steady state by balancing dissipation.

---

### `RHS(uk, uk_t, visc=1, forc=1)`
Calculate right-hand side of Navier-Stokes equation.

**Args:**
- `uk`: velocity Fourier coefficients (3 components)
- `uk_t`: output time derivative
- `visc`: viscosity multiplier (0 = no dissipation)
- `forc`: forcing multiplier (0 = no forcing)

**Physics:** Computes
```
∂u/∂t = -(u·∇)u - ∇p + ν∇²u + force
```

**Method:** 
1. Inverse transform to physical space
2. Compute vorticity ω = ∇ × u
3. Nonlinear term: (ω × u)
4. Pressure: p from incompressibility constraint
5. Viscous term: ν·∇²u (hyperviscous if lp > 1)
6. Transform back to Fourier space

**Returns:** Time derivative ∂uk/∂t

**Note:** Two variants: standard and phase-shifted (higher accuracy, more expensive).

---

## I/O Functions

### `save(i, uk)`
Save simulation snapshot to .npz files.

**Saves:**
- `Fields_k_{rank}.npz`: Fourier coefficients (û, v̂, ŵ)
- `Energy_spectrum.npz`: E(k)
- `Flux_spectrum.npz`: Energy flux $\Pi(k)$

**Also computes and prints:**
- Total kinetic energy (in physical and spectral space)
- Maximum divergence (should be $< 10^{-10}$)

**Example output:**
```
Energy at time 5.0 is : 157.3, 157.2
Maximum divergence 1e-12
```

---

### `save_hdf5(i, uk)`
Save to HDF5 format (parallel, efficient).

**Saves same data as `save()` but in single HDF5 file. Requires parallel hdf5. No compression.**

**Advantages:**
- Faster I/O at high resolution
- Self-documenting metadata

**Metadata stored:**
- `nu`: viscosity
- `Power input`: forcing magnitude
- `eta`: Kolmogorov length scale
- `N`: grid resolution
- `forcing`: description of forcing method

---

### `load_npz(paths,uk)`
Load simulation from .npz snapshot.

**Args:**
- `paths`: directory containing Fields_*.npz files
- `uk`: output array

**Returns:** Loaded velocity field

**Handles:**
- MPI distribution (each rank loads its slice)
- Truncated (dealias) format

---

### `load_hdf5(paths, u)`
Load from HDF5 file with MPI I/O.

**Args:**
- `paths`: directory with Fields.hdf5
- `u`: output velocity array

**Returns:** Velocity field

---

## Main Loop

### `evolve_and_save(t, u)`
Time-step the system from t[0] to t[-1].

**Args:**
- `t`: time array (from tinit to T)
- `u`: initial velocity field

**Algorithm:**
1. Setup viscous integration (implicit/exponential/explicit)
2. For each time step:
   - Compute RHS (4-step RK4)
   - Apply hyperviscosity implicitly
   - Enforce reality condition (real-valued u)
   - Enforce divergence-free (∇·u = 0)
   - Save if i % st == 0
3. Return final state

**Time integration:** 4th-order Runge-Kutta with semi-implicit viscosity

**Viscosity schemes:**
- `"implicit"`: (1 + h·ν·k²ˡᵖ)⁻¹ (stable, recommended)
- `"exponential"`: exp(-h·ν·k²ˡᵖ) (exact for linear part)
- `"explicit"`: explicit Runge-Kutta (unstable at low Re)

---

## Configuration Parameters

### Grid/Domain
- `N`: resolution per direction (64, 128, 256...)
- `Np`: local grid points per MPI rank (N / num_process)
- `dx, dy, dz`: grid spacing
- `L`: box size (2π)

### Time Stepping
- `dt`: time step (auto-scales: 0.256/N)
- `T`: total simulation time
- `st`: save interval in steps (dt_save/dt)

### Physics
- `nu0`: base viscosity (scales with resolution)
- `lp`: hyperviscosity power (default: 1 = regular Laplacian)
- 'm' : Desired $k_{\text{max}}\eta$ in the steady state. 
- `nu`: actual viscosity (computed from nu0 and m).
- `Re = 1/nu`: Reynolds number

### Forcing
- `shell_no`: which k-shells to force (e.g., [1, 2])
- `f0`: power input
- `nshells`: number of shells being forced
- `isforcing`: boolean switch (1 = on, 0 = off)

### Dealiasing
- `dealias`: boolean mask.
- `phase_shifted`: use Patterson-Orszag dealiasing.

---

## Global Arrays

### Velocity & Derivatives
- `u`: [3, Np, N, N] - physical space velocity
- `uk`: [3, N, Np, Nf] - Fourier velocity
- `omg`: [3, Np, N, N] - physical space vorticity ω = ∇×u

### RHS Terms
- `rhsuk, rhsvk, rhswk`: RHS of u, v, w equations

### Spectral Operators
- `lap`: Laplacian in k-space = -(kx² + ky² + kz²)
- `invlap`: inverse Laplacian (with dealias)
- `k`: magnitude |k|
- `vis`: hyperviscous operator ν·k^(2lp)

### Energy/Flux
- `ek`: 3D energy field (half the |u|²)
- `ek_arr`: 1D energy spectrum E(k)
- `Pik_arr`: 1D energy flux P(k)

---

## Typical Workflow

```python
# 1. Configure (edit top of forced-dns.py)
N = 128
nu = 0.001
T = 50

# 2. Run
mpirun -n 4 python forced-dns.py

# 3. Analyze
import numpy as np
data = np.load("data/.../time_10.0/Energy_spectrum.npz")
E_k = data['ek']
```
