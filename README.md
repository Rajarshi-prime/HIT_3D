
# HIT_3D: 3D Homogeneous Isotropic Turbulence Solver

* README generated using CLAUDE and checked by the owner. 

A high-performance pseudo-spectral direct numerical simulation (DNS) code for studying homogeneous isotropic turbulence and particle-laden flows in 3D. Written in Python with MPI parallelization.

## Overview

This code solves the incompressible Navier-Stokes equations in a triply periodic domain using Fourier spectral methods. It's designed to simulate turbulent flows with and without particles, capturing dynamics across a wide range of scales.


## What It Does

- Solves the Navier-Stokes equations: ∂v/∂t + (v·∇)v = -∇p + ν∇²v in a 3D periodic box.
- The viscosity can be modified to hyperviscosity by changing the power of the Laplacian.
- Uses Fourier basis functions for spatial discretization.
- Applies 2/3 dealiasing or $2\sqrt{2}/3$ phase-shifted dealiasing (recommended) to prevent aliasing errors in nonlinear terms.
- Supports both forced turbulence (steady state) and decaying turbulence.
- Autocomputes (hyper) viscosity based on the desired $k_{\text{max}}\eta$.
- Computes spectral statistics and structure functions.

## Requirements

- Python 3.6+
- NumPy
- SciPy
- mpi4py (for MPI parallelization)
- FFTW or pyFFTW (for fast Fourier transforms)
- h5py (for data storage) (optional).

## Installation

```bash
# Clone the repository
git clone https://github.com/Rajarshi-prime/HIT_3D.git
cd HIT_3D

# Install dependencies
pip install numpy scipy mpi4py h5py

# For parallel FFT support
pip install pyfftw

```

## Usage

### Basic Forced Turbulence Simulation

```python
# Run a forced DNS simulation
python forced-dns.py

# Configure parameters in the script:
# - Resolution (N): Grid points per direction
# - Desired m = k_max\eta in the stationary state
# - Time parameters (dt, total time)
# - set isforcing true or false
```


### Analysis and Visualization

```python
# Open the analysis notebook
jupyter notebook plots.ipynb

# This includes:
# - Energy spectra E(k).
# - Spectral energy flux.
# - Kinetic energy evolution.
# - Velocity structure functions S_p(r).
```

## Key Files

| File | Purpose |
|------|---------|
| `forced-dns.py` | Main solver for forced turbulence simulations |
| `plots.ipynb` | Jupyter notebook for visualization and analysis |
| `sfunc.py` | Structure function calculations |


## Output Data

Simulations produce:
- Flow fields: u(x,y,z,t) stored in npz (default) or HDF5 format.
- Energy spectra and flux at different times.

## Performance Notes

- MPI scaling is good up to several hundred processors
- Memory scales as O(N³) where N is grid resolution

## References

This code builds on the spectralDNS framework:
- [spectralDNS on GitHub](https://github.com/spectralDNS/spectralDNS)


## Contributing

This is an active research project. For questions, modifications, or bug reports, open an issue on GitHub.

## Author

Rajarshi Chattopadhyay
International Centre for Theoretical Sciences (ICTS), Bengaluru, India
[https://github.com/Rajarshi-prime](https://github.com/Rajarshi-prime)

