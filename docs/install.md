# Installation

```bash
pip install moment-sos
```

Requirements: Python ≥ 3.9, NumPy, SciPy, PyTorch ≥ 2.0, NetworkX. Optional: `cvxpy` and `clarabel` (or MOSEK) for the
interior-point reference solver `moment_sos.solve_dual_ipm`, `matplotlib` for the examples.

From a source checkout:

```bash
git clone https://github.com/TongCHEN779/moment-sos.git
cd moment-sos
pip install -e ".[test]"
pytest
```

:::{note}
Computations are in double precision on the CPU. `device="cuda"` is accepted by the solvers but has **not been
tested**; the dense eigendecomposition is the bottleneck and its speed on GPUs depends on your hardware.
:::

## Command line and Julia

The package installs the `moment-sos` command (see {doc}`guide/cli_json`). The Julia package `MomentSOS.jl` calls it;
see {doc}`julia`.
