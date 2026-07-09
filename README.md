# Solving Large Sparse Linear Systems: CG and PCG Methods



> **Individual Project** — Numerical Methods of Linear Algebra for Sparse Matrices  
> Southern Federal University, Department of Mathematical Modeling, 2026  
> **Authors:** Goriola-Obafemi Babatunde Sukanmi · Ovolabani Ayobami  
> **Supervisor:** Anna Nasedkina

---

## Overview

This project implements and analyses two Krylov subspace iterative methods for solving large sparse symmetric positive definite (SPD) linear systems of the form **Ax = b**:

- **Method No. 20 — Conjugate Gradient (CG):** The theoretically optimal iterative method for SPD systems. Derived from two fundamental conditions: residual orthogonality and A-conjugacy of search directions.
- **Method No. 30 — Preconditioned Conjugate Gradient (PCG):** Extends CG with an Incomplete Cholesky preconditioner that reduces the effective condition number κ(M⁻¹A) ≪ κ(A), dramatically accelerating convergence for ill-conditioned systems.

Both methods belong to the family of Krylov subspace methods based on **Lanczos orthogonalisation** and are implemented from scratch in MATLAB, tested across multiple linear systems, and compared directly against each other and against MATLAB's built-in `pcg()` function.

---

## Why CG and PCG?

Large sparse systems arise from PDE discretisation in structural mechanics, heat transfer, and fluid dynamics. Direct methods (LU, Gaussian elimination) destroy sparsity through fill-in and require O(n²) memory — infeasible for n = 10⁶. Iterative methods exploit sparsity by performing only matrix-vector products at O(nnz) cost per iteration.

CG is the **uniquely optimal** Krylov method for SPD systems:
- Minimises ‖e_k‖_A over K_k(A, r₀) at every step
- O(n) memory — stores only 4 vectors: x, r, p, Ap
- Convergence in at most n steps (finite termination)
- Rate depends on √κ(A) — far superior to stationary methods

PCG is its natural extension — one extra step per iteration unlocks the full benefit of preconditioning.

---

## Mathematical Background

### The CG Algorithm

Given A (SPD), b, initial guess x₀, tolerance ε:

```
r₀ ← b − Ax₀ ;   p₀ ← r₀ ;   k ← 0

while ‖r_k‖/‖b‖ > ε do:
    α_k ← (r_k, r_k) / (Ap_k, p_k)       ← optimal step length
    x_{k+1} ← x_k + α_k · p_k             ← update solution
    r_{k+1} ← r_k − α_k · A · p_k         ← update residual
    β_k ← (r_{k+1}, r_{k+1}) / (r_k, r_k) ← direction scalar
    p_{k+1} ← r_{k+1} + β_k · p_k         ← new A-conjugate direction
    k ← k + 1
end while
```

**Why is it called Conjugate Gradient?**
- *Gradient*: search directions are derived from the residual r = −∇φ(x), the negative gradient of the energy functional φ(x) = ½xᵀAx − xᵀb
- *Conjugate*: directions satisfy (Ap_i, p_j) = 0 for i ≠ j — A-orthogonal, making each step non-redundant

### Convergence Bound (Saad, 2003, Theorem 6.29)

```
‖e_k‖_A  ≤  2 · ((√κ − 1) / (√κ + 1))^k · ‖e₀‖_A
```

where κ(A) = λ_max / λ_min is the condition number.

### The PCG Algorithm

PCG modifies CG with one extra step — applying the preconditioner M⁻¹:

```
r₀ ← b − Ax₀ ;  z₀ ← M⁻¹r₀ ;  p₀ ← z₀ ;  k ← 0

while ‖r_k‖/‖b‖ > ε do:
    α_k ← (r_k, z_k) / (Ap_k, p_k)
    x_{k+1} ← x_k + α_k · p_k
    r_{k+1} ← r_k − α_k · A · p_k
    z_{k+1} ← M⁻¹ r_{k+1}              ← APPLY PRECONDITIONER (only change from CG)
    β_k ← (r_{k+1}, z_{k+1}) / (r_k, z_k)
    p_{k+1} ← z_{k+1} + β_k · p_k
    k ← k + 1
end while
```

In MATLAB: `z = L_tilde' \ (L_tilde \ r)` — two sparse triangular solves.

---

## Sparsity Patterns

The sparsity pattern of A reflects the underlying mesh connectivity and directly determines κ(A) and convergence speed.

| 1D Poisson Matrix (n=500, nnz=1498) | gr30×30 Matrix Market (n=900, nnz=7744) |
|:---:|:---:|
| ![1D Poisson](figures/spy_poisson.png) | ![gr30x30](figures/spy_gr30.png) |
| Tridiagonal structure — 1D connectivity | Five-band structure — 2D connectivity |
| IC factor is near-exact → PCG converges in 1 step | Wider bandwidth → larger κ(A) → more iterations |

---

## Numerical Results

All experiments: MATLAB R2024b, tolerance ε = 10⁻⁸, initial guess x₀ = 0.

### Small System (n = 5)

| Method | Iterations | ‖r‖₂ | Relative Residual | Abs. Error | Converged |
|--------|-----------|-------|------------------|------------|-----------|
| CG | 5 | 5.96e-16 | 5.40e-17 | 3.35e-16 | ✅ Yes |
| PCG (IC) | 1 | 2.82e-15 | 2.56e-16 | 3.36e-16 | ✅ Yes |

CG converged in exactly n=5 iterations — confirming the **finite termination property**. PCG converged in 1 iteration — the IC factor is near-exact for small dense matrices giving κ(M⁻¹A) ≈ 1.

### Scaling Tests

| n | Iterations | ‖r‖₂ | Relative Residual | CPU (s) | Converged |
|---|-----------|-------|------------------|---------|-----------|
| 100 | 7 | 5.72e-05 | 4.05e-09 | 0.0078 | ✅ Yes |
| 1,000 | 7 | 2.47e-02 | 6.29e-09 | 0.0027 | ✅ Yes |
| 3,000 | 7 | 1.11e-01 | 1.80e-09 | 0.0187 | ✅ Yes |

**Key analytical insight:** Iteration count stayed constant at 7 across three orders of magnitude in n. This confirms CG is **spectrally aware, not size-aware** — convergence depends on κ(A), not n directly.

### Matrix Market — gr30×30 (n = 900)

| Method | Iterations | ‖r‖₂ | Relative Residual | Abs. Error | CPU (s) | Converged |
|--------|-----------|-------|------------------|------------|---------|-----------|
| CG | 41 | 2.38e-07 | 7.14e-09 | 5.10e-08 | 0.0071 | ✅ Yes |
| PCG (IC) | **22** | 2.24e-07 | 6.72e-09 | 4.85e-07 | 0.0089 | ✅ Yes |

**PCG reduced iterations by 46%** (41 → 22). The marginal CPU overhead at n=900 is explained by the two triangular solves per PCG iteration — at n ≥ 10⁵ the iteration savings dominate and PCG wins decisively in total time.



---

## Repository Structure

```
CG-PCG-Sparse-Linear-Systems/
│
├── README.md                          ← This file
├── code/
│   ├── myCG.m                         ← CG implementation (MATLAB)
│   └── myPCG.m                        ← PCG implementation (MATLAB)
├── report/
│   └── CG_PCG_Report.pdf              ← Full academic report
├── presentation/
│   └── CG_PCG_Slides.pptx             ← 20-slide presentation
├── figures/
│   ├── spy_poisson.png                ← 1D Poisson sparsity pattern
│   ├── spy_gr30.png                   ← gr30×30 sparsity pattern
│   ├── convergence_plot.png           ← CG convergence vs κ(A)
│   └── cg_vs_pcg.png                  ← CG vs PCG comparison
└── data/
    └── gr_30_30.mtx                   ← Matrix Market test matrix
```

---

## How to Run

### Requirements
- MATLAB R2020b or later
- `mmread.m` function (available from [NIST Matrix Market](https://math.nist.gov/MatrixMarket/))

### Run CG

```matlab
% Open myCG.m in MATLAB as a Live Script
% Run section by section with Ctrl+Enter

% Or call the function directly:
n = 100;
R = rand(n);
A = R'*R + n*eye(n);       % SPD matrix
x_exact = rand(n,1);
b = A * x_exact;

[x, r, r_norm, rel_res, k] = CG_solver(A, b, zeros(n,1), 1e-8, 5000);
fprintf('CG converged in %d iterations, rel. residual = %.2e\n', k, rel_res)
```

### Run PCG

```matlab
% Compute IC preconditioner
M_lower = ichol(sparse(A));

[x, r, r_norm, rel_res, k] = PCG_solver(A, b, zeros(n,1), 1e-8, 5000, M_lower);
fprintf('PCG converged in %d iterations, rel. residual = %.2e\n', k, rel_res)
```

### Load Matrix Market matrix

```matlab
addpath('path/to/mmread')
A = mmread('data/gr_30_30.mtx');
whos A
```

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **46% iteration reduction** | PCG (22 iters) vs CG (41 iters) on gr30×30 |
| **Spectral awareness** | CG iteration count determined by κ(A), not matrix size n |
| **Perfect preconditioning** | PCG converges in 1 iteration on tridiagonal Poisson — IC factor is exact |
| **CPU paradox** | PCG marginally slower at n=900 due to triangular solve overhead — reverses at large n |
| **Verified** | Both implementations match MATLAB pcg() exactly in all tests |

---

## References

1. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM.
2. Hestenes, M.R. and Stiefel, E. (1952). Methods of conjugate gradients for solving linear systems. *Journal of Research of the National Bureau of Standards*, 49(6), 409–436.
3. Nasedkina, A. (2026). Lecture Notes: Numerical Methods of Linear Algebra for Sparse Matrices (Lectures 8, 10, 14, 15). Southern Federal University.
4. NIST Matrix Market. Available at: https://math.nist.gov/MatrixMarket

---

## Academic Context

This project was completed as an individual project for the course **Numerical Methods of Linear Algebra for Sparse Matrices** at Southern Federal University (SFedU), Department of Mathematical Modeling, MSD Group, 2026.

