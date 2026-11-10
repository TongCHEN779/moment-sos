"""Helper (not a test): exact moment-relaxation value of a moment_sos Problem through the independent MOSEK model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from msos.core import POP
from msos import reference

def exact_relaxation(prob, order, trace_cuts=True):
    if prob.bool_mask.any() and not prob.boolean:
        raise ValueError("mixed Boolean not supported by the reference model")
    pop = POP(prob.n, prob.f.terms, [g.terms for g in prob.ineq], [h.terms for h in prob.eq], boolean=prob.boolean)
    return reference.moment_relaxation(pop, order, trace_cuts=trace_cuts)[0]
