# Installation Guide

## System Requirements

- Python 3.6+
- MPI compiler (OpenMPI or MPICH)
- C/C++ compiler (gcc, clang, or icc)

## Step 1: Clone Repository

```bash
git clone https://github.com/Rajarshi-prime/HIT_3D.git
cd HIT_3D
```

## Step 2: Install Dependencies

### Basic Dependencies

```bash
pip install numpy scipy h5py
```

### MPI Support

```bash
pip install mpi4py
```

### FFT (Critical)

```bash
# Install FFTW libraries first (system-dependent)
# Ubuntu/Debian
sudo apt-get install libfftw3-dev

# macOS with Homebrew
brew install fftw

# Then install Python wrapper
pip install pyfftw
```

### All at Once

```bash
pip install numpy scipy h5py mpi4py pyfftw
```

## Step 3: Verify Installation

```bash
python -c "import numpy, scipy, h5py, mpi4py, pyfftw; print('All imports successful')"
```

## Step 4: Run Test

```bash
# Single process (CPU only)
python forced-dns.py

# With MPI (4 processes)
mpirun -n 4 python forced-dns.py
```

## Troubleshooting

**ImportError: No module named 'fftw'**
- FFTW C library not installed. Return to scipy fft. It is well optimized. 

**MPI errors**
- Verify MPI installation: `mpirun --version`
- Reinstall mpi4py: `pip install --force-reinstall mpi4py`

**Memory issues at high resolution**
- Reduce grid size N in code
- Use more MPI processes
