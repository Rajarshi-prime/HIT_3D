# Quickstart

## Run Your First Simulation

### Single Process 

```bash
python forced-dns.py
```

Output: Data in `./data/forced_True/N_64_Re_125.0/`

### With MPI (4 processes)

```bash
mpirun -n 4 python forced-dns.py
```

## What Happens

1. Code generates random turbulent flow at low wavenumbers
2. Simulates 20 time units (configurable)
3. Saves flow fields, energy spectra, flux every 1.0 time unit
4. Prints energy and divergence stats

## Customize Parameters

Edit top of `forced-dns.py`:

```python
N = 128          # Grid resolution (64, 128, 256...)
dt = 0.256/N     # Time step (auto-scales)
T = 20           # Total time
nu0 = 0.8        # Base viscosity
shell_no = np.arange(1,2)  # Force shell 1 (can be [1,2,3])
m = 1,2,3.5 # k_max\eta value
```

## Output Files

```
data/forced_True/N_64_Re_125.0/
├── time_0.0/
│   ├── Fields_k_0.npz        # Fourier coefficients
│   ├── Energy_spectrum.npz   # E(k) at this time
│   └── Flux_spectrum.npz     # Transfer cascade
├── time_1.0/
└── ...
```

## Analyze Results

```python
import numpy as np
import matplotlib.pyplot as plt

# Load energy spectrum at time 1.0
data = np.load("data/forced_True/N_64_Re_125.0/time_1.0/Energy_spectrum.npz")
E_k = data['ek']

plt.loglog(E_k)
plt.xlabel('k')
plt.ylabel('E(k)')
plt.show()
```

## Next Steps

- Read `TUTORIAL.md` for physics background
- Check `API_REFERENCE.md` for function details
