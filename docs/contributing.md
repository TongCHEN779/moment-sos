# Contributing

```bash
git clone https://github.com/TongCHEN779/moment-sos.git && cd moment-sos
pip install -e ".[test]"
pytest                          # software tests (add -k "not docs" to skip the documentation examples)
```

- **Tests.** Every behavior change needs a test in `tests/`. Cross-validation against the independent NumPy reference is
  in `tests/test_library.py`; the interior-point comparison needs `cvxpy` and `clarabel` (or MOSEK).
- **Docs.** Pages are Markdown (MyST) in `docs/`. Python code blocks in the docs are executed by
  `tests/test_docs_examples.py`, so examples must run. Build with
  `pip install -r docs/requirements.txt && sphinx-build -W docs docs/_build/html`.
- **Honesty about numbers.** Any performance claim in the docs must come from a file in `research/experiments/results/` and be
  listed in `research/REPRODUCE.md`.
- **Julia package.** `julia/MomentSOS`: `julia --project=julia/MomentSOS -e 'using Pkg; Pkg.test()'` (needs the
  `moment-sos` Python package on `PYTHONPATH` or installed).
