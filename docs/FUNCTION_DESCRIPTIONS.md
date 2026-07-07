# Function Descriptions
# Not verified yet!
## Transform Functions

### `rfft_mpi(u, fu)`

**Purpose:** Convert real-valued velocity field from physical space to Fourier space using real FFT with MPI parallelization.

**Signature:**
```python
def rfft_mpi(u, fu):
    # u: [Np, N, N]  physical space field
    # fu: [N, Np, Nf]  Fourier output (Hermitian format)
    # returns: fu (modified in-place)
```

**Computational steps:**
1. 2D real FFT on y-z plane: `rfft2(u, axes=(1,2))` → [Np, N, Nf]
2. Reshape and transpose for MPI communication
3. AlltoAll MPI exchange (each rank sends/receives)
4. 1D FFT in x direction: `fft(fu, axis=0)`
5. Result: fu[kx, ky, kz] (Hermitian redundancy exploited)

**Why MPI:** At high resolution (N>256), data exceeds single CPU memory. MPI distributes u[y-slice] across ranks.

**Complexity:** O(N³ log N) with parallel I/O.

---

### `irfft_mpi(fu, u)`

**Purpose:** Inverse of rfft_mpi. Convert Fourier coefficients back to physical space.

**Signature:**
```python
def irfft_mpi(fu, u):
    # fu: [N, Np, Nf]  Fourier input
    # u: [Np, N, N]  physical space output
    # returns: u (modified in-place)
```

**Computational steps:**
1. 1D inverse FFT in x: `ifft(fu, axis=0)` → [N, Np, Nf]
2. AlltoAll MPI exchange
3. Reshape for 2D real inverse FFT
4. 2D real inverse FFT on y-z: `irfft2(..., axes=(1,2))` → [Np, N, N]
5. Normalization: divide by N³ (built into scipy)

**Note:** Applies dealias mask automatically in some variants.

---

## Derivative Functions

### `diff_x(u, u_x)`, `diff_y(u, u_y)`, `diff_z(u, u_z)`

**Purpose:** Compute spatial partial derivatives ∂u/∂x, ∂u/∂y, ∂u/∂z using spectral method.

**Signature (for diff_x):**
```python
def diff_x(u, u_x):
    # u: [Np, N, N]  physical space field
    # u_x: [Np, N, N]  output ∂u/∂x
    # returns: u_x
```

**Method:**
1. **diff_x:** Transpose to make x-direction local, then FFT
   - Most expensive (MPI communication required)
   - Uses `kx_diff` wavenumber array
   
2. **diff_y:** FFT along y, multiply by i·ky, iFFT
   - No MPI communication (y is local)
   - Uses `ky_diff` array
   
3. **diff_z:** Same as diff_y but for z-direction
   - Uses `kz_diff` array

**Key trick:** Spectral differentiation = multiply by ikα in Fourier space.

```
∂u/∂x = iFFT(ikx × FFT(u))
```

**Accuracy:** Exact up to machine precision (unlike finite differences).

---

## Analysis Functions

### `e3d_to_e1d(x)`

**Purpose:** Radially bin a 3D energy/spectrum field into 1D spherical shell spectrum.

**Signature:**
```python
def e3d_to_e1d(x):
    # x: [Np, N, N]  3D field in Fourier space
    # returns: [Nf]  1D spectrum
```

**Algorithm:**
1. Flatten 3D k-space: k = sqrt(kx² + ky² + kz²)
2. Round k to nearest integer shell
3. Use `np.histogram(k, bins=shells, weights=x)` to sum energy per shell
4. Result: E[n] = sum of x over all k-points with |k| ≈ n

**Physics:** Converts 3D spectrum E(kx, ky, kz) to isotropic 1D spectrum E(k).

**Example:**
```python
ek_3d = 0.5 * (np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2)  # 3D energy density
ek_1d = e3d_to_e1d(ek_3d)                        # 1D energy spectrum
print(ek_1d)  # [E0, E1, E2, ..., E_Nf]
```

**Normalization:** Already includes physical space norm factor.

---

## Physics Functions

### `forcing(uk, fk)`

**Purpose:** Calculate forcing term to maintain steady-state turbulence.

**Signature:**
```python
def forcing(uk, fk):
    # uk: [3, N, Np, Nf]  velocity Fourier coefficients
    # fk: [3, N, Np, Nf]  output forcing term
    # returns: fk
```

**Algorithm:**

**Step 1: Compute shell energy**
```python
ek = 0.5 * (|uk[0]|² + |uk[1]|² + |uk[2]|²)        # 3D energy
ek_arr = e3d_to_e1d(ek)  (with MPI allreduce)       # 1D spectrum
```

**Step 2: Scale velocity in force shells**
```python
for shell in shell_no:
    if ek_arr[shell] > threshold:
        factor[shell] = f0 / (2 * ek_arr[shell])  # Constant power input
    else:
        factor[shell] = 0
```

This maintains: Power = ε = ν × integral(k² E(k) dk)

**Step 3: Apply to velocity**
```python
fk[α] = factor[kint] × uk[α] × dealias
```

**Step 4: Enforce divergence-free**
```python
p = invlap × (kx fk[0] + ky fk[1] + kz fk[2])     # Solve Poisson
fk[α] += i kα p                                    # Correct forcing
```

**Result:** fk is divergence-free and injects energy only in specified shells.

**Parameters:**
- `f0`: power per shell = nu0³ × (2π)³ / nshells
- `shell_no`: e.g., [1] forces only k≈1
- `isforcing`: boolean switch (returns 0 if False)

---

### `RHS(uk, uk_t, visc=1, forc=1)`

**Purpose:** Calculate time derivative ∂uk/∂t from Navier-Stokes equation.

**Signature:**
```python
def RHS(uk, uk_t, visc=1, forc=1):
    # uk: [3, N, Np, Nf]  velocity Fourier coefficients
    # uk_t: [3, N, Np, Nf]  output: ∂uk/∂t
    # visc: float  viscosity multiplier (0 = no dissipation)
    # forc: float  forcing multiplier (0 = no forcing)
    # returns: uk_t
```

**Algorithm:**

**Step 1: Compute forcing**
```python
fk = forcing(uk, fk) * forc
```

**Step 2: Transform to physical space**
```python
u[α] = irfft_mpi(uk[α])  # Inverse FFT
```

**Step 3: Compute vorticity**
```python
omg[0] = irfft_mpi(i(ky uk[2] - kz uk[1]))  # ωx = ∂w/∂y - ∂v/∂z
omg[1] = irfft_mpi(i(kz uk[0] - kx uk[2]))  # ωy = ∂u/∂z - ∂w/∂x
omg[2] = irfft_mpi(i(kx uk[1] - ky uk[0]))  # ωz = ∂v/∂x - ∂u/∂y
```

**Step 4: Nonlinear term (Lamb vector form)**
```python
# (ω × u) in physical space:
rhs[0] = omg[2] u[1] - omg[1] u[2]  # = (ω × u)_x
rhs[1] = omg[0] u[2] - omg[2] u[0]  # = (ω × u)_y
rhs[2] = omg[1] u[0] - omg[0] u[1]  # = (ω × u)_z
```

This avoids computing ∇(u²) directly (more stable).

**Step 5: Transform back to Fourier**
```python
rhsuk = rfft_mpi(rhs[0]) + fk[0]
# (repeat for v, w)
```

**Step 6: Pressure projection**
Incompressibility ∇·u = 0 requires:
```python
p = invlap × (kx rhsuk + ky rhsvk + kz rhswk)  # Poisson equation
```

Then subtract ∇p:
```python
rhsuk -= i kx p
rhsvk -= i ky p
rhswk -= i kz p
```

**Step 7: Viscous dissipation**
```python
uk_t[0] = rhsuk - nu × k^(2lp) × uk[0] × visc
# (repeat for v, w)
```

where k^(2lp) is hyperviscosity operator.

**Dealiasing:** All operations apply dealias mask to remove aliasing errors. Furthermore, the non-linear terms are computed again for the phase-shifted dealiasing.

**Return:** ∂uk/∂t ready for time-stepping.

---

## I/O Functions

### `save(i, uk)` and `save_hdf5(i, uk)`

**Purpose:** Save flow fields, energy spectrum, and energy flux at snapshot.

**Signature:**
```python
def save(i, uk):
    # i: int  time step index
    # uk: [3, N, Np, Nf]  velocity Fourier coefficients
    # Saves to ./data/forced_{isforcing}/N_{N}_Re_{1/nu}/time_{t[i]}/
```

**Saves (save):**
- `Fields_k_{rank}.npz`: uk, vk, wk (Fourier coefficients)
- `Energy_spectrum.npz`: ek (energy density per k-shell)
- `Flux_spectrum.npz`: Pik (energy flux per k-shell)

**Saves (save_hdf5):**
- Single file `Fields.hdf5` with datasets: uk, vk, wk, Energy_Spectrum, Flux_Spectrum
- Attributes: nu, Power input, eta, t_eta, forcing method, N

**Intermediate calculations:**
```python
# Energy
ek = 0.5 * |uk|²
ek_arr = radially_bin(ek)  # 1D spectrum

# Energy transfer rate
k1u = RHS(uk, ..., visc=0, forc=0)  # Time deriv without dissipation
Pik = real(conj(uk) · k1u)           # Work done by nonlinear term
Pik_arr = cumsum(Pik_arr[::-1])[::-1]  # Convert to flux
```

**Output example:**
```
Energy at time 5.0 is : 157.3, 157.2
Maximum divergence 0.0012
```

Two energy numbers: (1) physical space integral, (2) sum of E(k).

**Prints:**
- Kinetic energy (two methods should match)
- Maximum ∇·u (should be < 0.01)

---

### `load_npz(paths, u/uk)` and `load_hdf5(paths, u)`

**Purpose:** Restore velocity from saved snapshot (for continuing runs).

**Signature:**
```python
def load_npz(paths, uk):  # phase_shifted variant
    # paths: Path to directory with Fields_*.npz
    # uk: [3, N, Np, Nf]  output velocity
    # Loads truncated Fourier coefficients
```

**Algorithm:**
1. Count .npz files to determine how many slabs were saved
2. Each rank loads its own y-slices from corresponding .npz
3. Reconstructs uk from Fields_k_{slab}.npz

**For phase_shifted:** Loads truncated (dealias) format.
**For standard:** Loads full complex representation.

**Note:** Performs no transforms; directly restores Fourier coefficients.

---

## Main Evolution

### `evolve_and_save(t, u)`

**Purpose:** Time-step Navier-Stokes from t[0] to t[-1], saving snapshots.

**Signature:**
```python
def evolve_and_save(t, u):
    # t: array [tinit, tinit+dt, ..., T]  time array
    # u: [3, Np, N, N]  initial velocity (physical space)
    # Evolves in-place, saves to disk
```

**Algorithm:**

**Initialization:**
```python
h = t[1] - t[0]  # time step dt
# Setup viscous integrators:
if implicit:
    hypervisc = (1 + h·ν·k^(2lp))⁻¹      # Applied at end of RK4
elif exponential:
    semi_G = exp(-h·ν·k^(2lp))            # Exact viscous evolution
else:
    explicit RK4 only
```

**Main time loop:**
```python
for i in range(len(t)-1):
    if i % st == 0:  # Save every st steps
        save(i, uk)
    
    # 4-step Runge-Kutta with semi-implicit viscosity:
    k1 = RHS(uk)
    k2 = RHS(semi_G_half·(uk + h/2·k1))
    k3 = RHS(semi_G_half·uk + h/2·k2)
    k4 = RHS(semi_G·uk + semi_G_half·h·k3)
    
    # Combine steps:
    uknew = semi_G·uk + h/6·(semi_G·k1 + 2·semi_G_half·(k2+k3) + k4)
    uknew *= hypervisc  # Apply implicit viscosity
    
    # Enforce constraints:
    u = irfft_mpi(uknew)           # Physical space
    uk = rfft_mpi(u)               # Back to Fourier
    p = invlap·(∇·uk)              # Poisson solve
    uk += ∇p                       # Correct velocity
    
    # Safety check:
    if max(uk) > 100·N³:
        print("Instability detected!")
        abort()
```

**Stability:** Time step dt = 0.256/N scales so CFL ≈ constant.

**Constraints enforced each step:**
1. Reality condition: u ∈ ℝ (no spurious imaginary parts)
2. Divergence-free: ∇·u = 0 (enforced by Poisson solve)

**Final save:** After loop, save final state.

---

## Initialization

**Cold start:**
```python
# Generate random phases for k < kinit
thu, thv, thw = random uniform [0, 2π]

# Specify energy profile
eprofile = k² exp(-k²/2)  # Gaussian in wavenumber

# Create velocity with this energy
uk = amplitude × exp(i·phase) × (k < kinit)
```

Then project to divergence-free:
```python
u = irfft(uk)
uk = rfft(u)
p = invlap·(∇·uk)
uk += ∇p
```

Finally, scale to target energy:
```python
uk *= sqrt(einit / current_energy)
```

**Restart:**
```python
uk = load_npz(last_time_folder)
```

---

## Summary Table

| Function | Input | Output | Cost |
|----------|-------|--------|------|
| rfft_mpi | u [Np,N,N] | uk [N,Np,Nf] | O(N³ logN) MPI |
| irfft_mpi | uk | u | O(N³ logN) MPI |
| diff_x/y/z | u | ∂u/∂x | O(N³) FFT + MPI |
| e3d_to_e1d | 3D field | 1D spectrum | O(N³) histogram |
| forcing | uk | fk | O(N³) with allreduce |
| RHS | uk | ∂uk/∂t | O(N³) FFT-based |
| save | i, uk | .npz files | I/O bound |
| evolve_and_save | t, u | final uk | 4 RHS calls/step |
