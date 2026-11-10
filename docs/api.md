# API reference

## Modeling

```{eval-rst}
.. autofunction:: moment_sos.variables
.. autoclass:: moment_sos.Poly
   :members: constant, variable, degree, substitute_affine
.. autoclass:: moment_sos.Constraint
.. autoclass:: moment_sos.Problem
   :members: to_original, to_internal, value, violation
```

## Solving

```{eval-rst}
.. autofunction:: moment_sos.solve
.. autofunction:: moment_sos.solve_alm
.. autofunction:: moment_sos.min_order
.. autoclass:: moment_sos.Result
   :members: bound, bound_kind, round, rigorous, summary
.. autodata:: moment_sos.DEFAULT_ETAS
.. autodata:: moment_sos.SHORT_ETAS
```

## Adaptive lifting (triangle walk)

```{eval-rst}
.. autofunction:: moment_sos.walk
.. autoclass:: moment_sos.WalkResult
.. autofunction:: moment_sos.select_triples
.. autofunction:: moment_sos.triangle_violations
```

## Relaxation internals

```{eval-rst}
.. autoclass:: moment_sos.Relaxation
.. autofunction:: moment_sos.correlative_sparsity_cliques
.. autoclass:: moment_sos.Oracle
.. autofunction:: moment_sos.lift_theta
```

## Verification and references

```{eval-rst}
.. autofunction:: moment_sos.rigorous_bound
.. autofunction:: moment_sos.verified_lmin_lower_bound
.. autofunction:: moment_sos.solve_dual_ipm
```

## Serialization

```{eval-rst}
.. autofunction:: moment_sos.problem_to_dict
.. autofunction:: moment_sos.problem_from_dict
.. autofunction:: moment_sos.load_problem
.. autofunction:: moment_sos.save_problem
.. autofunction:: moment_sos.result_to_dict
```

## Neural-network encodings

```{eval-rst}
.. automodule:: moment_sos.nn
   :members: relu_verification, verification_exact_milp, verification_pgd, lipschitz_problem, lipschitz_exact_pattern, lipsdp_neuron, ibp, forward
```
