# Firn Transient

A 1D transient firn compaction model with nondimensional scaling, implemented in `main.py`.

## Quick Start

Clone the repository and enter it:

```bash
git clone https://github.com/ldeo-glaciology/firn_transient
cd firn_transient
```

Using `uv`:

```bash
uv sync
```

If you do not have `uv`, use `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install .
```

In a notebook cell (for example from `notebooks/`), run:

```python
import sys
sys.path.insert(0, "..")

from main import FirnModel

sim = FirnModel()
sim.setup()
out = sim.run()
out
```

To change parameters, pass keyword arguments to `setup(...)`:

```python
sim = FirnModel()
sim.setup(
    b0_mpy=0.2,          # dimensional accumulation rate
    beta=1.3,            # nondimensional accumulation-rate parameter
    T_s_dim=258.15,      # surface temperature [K]
    z0=120,              # initial column height [m]
    dz=0.01,             # nondimensional grid spacing factor
    simDuration=6,       # nondimensional simulation duration
    interp_on_reg_z=True,
    print_messages=False,
)
out = sim.run()
out
```

## Run Tests

Using `uv`:

```bash
uv sync --extra test
.venv/bin/pytest -v
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install ".[test]"
pytest -v
```

## Notebook

Exploratory notebook for transient response to accumulation-rate changes:

- `notebooks/transient_accumulation_xarray_sweep.ipynb`

This notebook was created by Codex for exploratory purposes.

## Reference

The model is based on:

Kingslake, J., Skarbek, R., Case, E., and McCarthy, C. (2022). *Grain-size evolution controls the accumulation dependence of modelled firn thickness*. **The Cryosphere**, 16, 3413-3430. [https://doi.org/10.5194/tc-16-3413-2022](https://doi.org/10.5194/tc-16-3413-2022)

## CI

GitHub Actions workflow runs tests:

- `.github/workflows/tests.yml`
