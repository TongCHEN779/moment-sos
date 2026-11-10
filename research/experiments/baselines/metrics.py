"""Accuracy-vs-time bookkeeping shared by driver.py and collect.py."""
import numpy as np

TOLS = (1e-1, 1e-2, 1e-3)


def strict_err(row, ref):
    """row = [t, value, infeas_inf]: max(|value - ref|, max constraint violation)."""
    return max(abs(row[1] - ref), row[2])


def time_to(hist, ref, tol, strict=True):
    """First time at which |value-ref| <= tol (and, if strict, max violation <= tol).  None if never."""
    for t, v, inf in hist:
        if abs(v - ref) <= tol and (not strict or inf <= tol):
            return t
    return None


def best_final_err(hist, ref):
    return min(strict_err(r, ref) for r in hist) if hist else np.inf


def times(hist, ref, strict=True):
    return {str(tol): time_to(hist, ref, tol, strict) for tol in TOLS}
