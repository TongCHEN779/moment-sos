# Theory in brief

This page states the results the software relies on. Proofs and the derivations are in the accompanying manuscript
(`research/moment-sos.tex` in the repository).

## Problem and relaxation

$$f^\star=\max_{x\in K} f(x),\qquad K=\{x:\ g_i(x)\ge0,\ h_j(x)=0\}\cap[-1,1]^n .$$

The order-$r$ Lasserre relaxation $\rho_r$ maximizes $L_y(f)$ over moment vectors $y$ with $y_0=1$,
$M(y)\succeq0$, $M^{\rm loc}_i(y)\succeq0$ and the equality moment constraints; $f^\star\le\rho_r$. With correlative
sparsity there is one moment block per clique. Adding the **trace cuts**
$\operatorname{tr}M_k(y)\le s_k$ and $\operatorname{tr}M_i^{\rm loc}(y)\le s_iB_i$ (valid on the box, with $B_i\ge\sup g_i$)
gives $\rho_r^{\rm tr}$ with $f^\star\le\rho_r^{\rm tr}\le\rho_r$; for spin variables the trace is identically $s_k$
and $\rho_r^{\rm tr}=\rho_r$.

## Dual points with free Gram matrices

A **dual point** $\theta$ is a scalar $\sigma$, symmetric matrices $Q_{0,k},Q_i$ (not required to be PSD) and equality
multipliers $p_j$ such that

$$\sigma-f=\sum_k v_k^\top Q_{0,k}v_k+\sum_i (z_i^\top Q_i z_i)\,g_i+\sum_j p_j h_j$$

as polynomials (modulo $x_i^2=1$ for spins). The package parametrizes this set by an unconstrained vector — one value
per Gram slot, with a zero-sum projection that enforces the identity — so **every parameter value is a dual point**.

## The certified bound

$$U(\theta)=\sigma+\sum_k s_k\,\Phi_k(Q_{0,k})+\sum_i s_iB_i\,[-\lambda_{\min}(Q_i)]_+,$$

with $\Phi_k=-\lambda_{\min}$ for spin cliques (no positive part) and $[-\lambda_{\min}]_+$ otherwise.

:::{admonition} Theorem (certificate)
For every dual point $\theta$ and every $x\in K$, $f(x)\le U(\theta)$.
:::

*Idea.* Shifting each $Q$ by a multiple of the identity makes it PSD; on the box $\|v_k(x)\|^2\le s_k$ (equal to
$s_k$ for spins) and $0\le g_i\le B_i$, so the shift costs at most the added terms.

:::{admonition} Theorem (no duality gap)
$\inf_\theta U(\theta)=\rho_r^{\rm tr}$ and the primal problem attains its maximum. For spin problems this equals the
Lasserre value $\rho_r$.
:::

So minimizing $U$ over the unconstrained $\theta$ — an eigenvalue optimization problem — recovers the relaxation value,
and any intermediate iterate is still a valid bound. This is what the solvers do.

## Solvers

- **Smoothing.** Replace $-\lambda_{\min}$ by $\eta\log\operatorname{tr}\exp(-Q/\eta)$; the gradient is the softmax
  density $P=\exp(-Q/\eta)/\operatorname{tr}\exp(-Q/\eta)$ and the backward pass of the eigen-oracle is analytic (no
  eigenvector derivatives, so repeated eigenvalues are harmless). The smoothed value overestimates $U$ by at most
  $\eta\sum_b w_b\log s_b$.
- **Augmented Lagrangian** (boundary-point type) for $\min\ \sigma+\sum w_bt_b$ s.t. $Q_b(\theta)+t_bI\succeq0$.

## Beyond the box

Weights other than the uniform shift (diagonal shifts, `solve_alm(shift="diag")`) give the diagonal-cut relaxation,
which is valid but only differs from the uniform one for non-spin blocks.
