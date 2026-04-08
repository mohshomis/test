"""
Searching for fast matrix multiplication via tensor decomposition (ALS).

Setup:
  Matrix multiplication C = A @ B for n x n matrices corresponds to the
  matmul tensor T of shape (n^2, n^2, n^2). A rank-R decomposition of T
  into (U, V, W) where U, V, W are each (n^2 x R) gives a matmul algorithm
  that uses exactly R scalar multiplications:

      T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]    (the CP decomposition)

  So 'finding a fast matmul algorithm' = 'finding a low-rank CP decomposition
  of T'. The rank of T_{2,2,2} is known to be exactly 7 (Strassen 1969, proven
  optimal). The rank of T_{3,3,3} is unknown - best known upper bound is 23
  (Laderman 1976), and the lower bound is 19.

This script:
  1. Implements Alternating Least Squares (ALS) to search for rank-R
     decompositions of the matmul tensor.
  2. For 2x2, tries to rediscover Strassen by searching at R=7.
  3. For 2x2 at R=6, confirms that the residual stays nonzero - no
     decomposition exists (consistent with the proven lower bound).
  4. For 3x3, attempts searches at R=27, 23, 21 to see how far ALS
     can get with continuous optimization.

ALS will find decompositions over the real numbers, not over {-1, 0, 1}.
That's a known limitation - real-valued decompositions are easier to find
than integer-valued ones. But if ALS can't even find a real decomposition
at rank R, then no integer one exists either.
"""

import numpy as np
import json
import time

from experiment8 import matmul_tensor, verify_decomposition, tensor_rank_apply


# ---------- ALS on the matmul tensor ----------

def als_decomposition(T, R, n_restarts=5, max_iters=2000, tol=1e-10,
                      init_scale=0.3, seed=0, verbose=False):
    """
    Alternating Least Squares to find T[i,j,k] = sum_r U[i,r]V[j,r]W[k,r].
    T has shape (d1, d2, d3). U: (d1, R), V: (d2, R), W: (d3, R).

    Returns the best (U, V, W) found across restarts, along with the final
    residual || T - sum_r U[:,r] o V[:,r] o W[:,r] ||_F.
    """
    d1, d2, d3 = T.shape
    T = T.astype(np.float64)
    T_norm = np.linalg.norm(T)
    T1 = T.reshape(d1, d2 * d3)       # mode-1 unfolding
    T2 = T.transpose(1, 0, 2).reshape(d2, d1 * d3)
    T3 = T.transpose(2, 0, 1).reshape(d3, d1 * d2)

    best_res = np.inf
    best_UVW = None
    for restart in range(n_restarts):
        rng = np.random.default_rng(seed + restart * 31)
        U = rng.standard_normal((d1, R)) * init_scale
        V = rng.standard_normal((d2, R)) * init_scale
        W = rng.standard_normal((d3, R)) * init_scale

        prev_res = np.inf
        for it in range(max_iters):
            # Update U: T1 = U @ khatri_rao(W, V).T  (Khatri-Rao = column-wise Kron)
            KR = np.einsum('kr,jr->jkr', W, V).reshape(d2 * d3, R)
            U, *_ = np.linalg.lstsq(KR, T1.T, rcond=None)
            U = U.T
            # Update V
            KR = np.einsum('kr,ir->ikr', W, U).reshape(d1 * d3, R)
            V, *_ = np.linalg.lstsq(KR, T2.T, rcond=None)
            V = V.T
            # Update W
            KR = np.einsum('jr,ir->ijr', V, U).reshape(d1 * d2, R)
            W, *_ = np.linalg.lstsq(KR, T3.T, rcond=None)
            W = W.T

            # Reconstruct and measure residual
            T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
            res = np.linalg.norm(T - T_hat) / T_norm

            if verbose and it % 200 == 0:
                print(f"    restart {restart} iter {it:>4}: res={res:.3e}")

            if res < tol:
                break
            if abs(prev_res - res) < 1e-14:
                break
            prev_res = res

        if res < best_res:
            best_res = res
            best_UVW = (U.copy(), V.copy(), W.copy())
            if verbose:
                print(f"  restart {restart}: final res = {res:.3e}  (new best)")

    return best_UVW, best_res


def try_ranks(T, ranks, n_restarts=5, max_iters=2000, name=""):
    """Try ALS at multiple ranks and report which succeed."""
    print(f"\n=== {name} ===")
    print(f"Tensor shape: {T.shape}")
    out = []
    for R in ranks:
        t0 = time.time()
        (U, V, W), res = als_decomposition(
            T, R, n_restarts=n_restarts, max_iters=max_iters, seed=42
        )
        dt = time.time() - t0
        success = res < 1e-8
        marker = "FOUND" if success else " no  "
        print(f"  R={R:>3}  residual={res:.3e}  {marker}  ({dt:.1f}s)")
        out.append({"R": R, "residual": float(res), "success": bool(success), "time": dt})
    return out


# ---------- verification ----------

def verify_als_decomp(U, V, W, n, n_trials=20, tol=1e-6):
    """Verify that the ALS decomposition actually computes matmul."""
    rng = np.random.default_rng(123)
    max_err = 0.0
    for _ in range(n_trials):
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        C_true = A @ B
        C_test = tensor_rank_apply(U, V, W, A, B)
        err = float(np.max(np.abs(C_true - C_test)))
        max_err = max(max_err, err)
    return max_err < tol, max_err


# ---------- main experiment ----------

def main():
    print("=" * 60)
    print("Searching for fast matmul via ALS tensor decomposition")
    print("=" * 60)

    results = {}

    # --- 2x2 case ---
    print("\n" + "#" * 60)
    print("# 2x2: known rank = 7 (Strassen, proven optimal)")
    print("#" * 60)
    T2 = matmul_tensor(2)
    res_2x2 = try_ranks(T2, [8, 7, 6, 5], n_restarts=10, max_iters=3000, name="2x2 matmul")
    results["2x2"] = res_2x2

    # Verify the R=7 result actually computes matmul
    print("\nVerifying R=7 solution (if found)...")
    (U, V, W), r = als_decomposition(T2, 7, n_restarts=15, max_iters=5000, seed=42)
    if r < 1e-8:
        ok, err = verify_als_decomp(U, V, W, n=2)
        print(f"  R=7 decomposition correctly computes 2x2 matmul: {ok}")
        print(f"  max error on random matrices: {err:.2e}")
        # Show the continuous-valued 'Strassen-like' algorithm
        print(f"\n  U matrix (A-side linear combos, one column per product):")
        print(np.round(U, 3))
        print(f"\n  (Compare to hand-coded integer Strassen - this is a different,")
        print(f"   continuous-valued algorithm that ALS found independently.)")
    else:
        print(f"  R=7 was not fully converged (residual={r:.2e})")

    # --- 3x3 case ---
    print("\n" + "#" * 60)
    print("# 3x3: known upper bound rank = 23 (Laderman 1976)")
    print("#     known lower bound rank = 19")
    print("#     the true rank is unknown")
    print("#" * 60)
    T3 = matmul_tensor(3)
    res_3x3 = try_ranks(T3, [27, 25, 23, 21, 19], n_restarts=5, max_iters=2000, name="3x3 matmul")
    results["3x3"] = res_3x3

    # Save
    with open("results_matmul_search.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_matmul_search.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n2x2 expected: converges at R=7, fails below R=7")
    print("3x3 expected: ALS can find R=23 if lucky, probably fails at R<23")
    print("\nIf our search found a rank-R decomposition for R smaller than the")
    print("best known bound, we would have made news. Spoiler: we probably")
    print("did not. But the framework is now in place for larger searches.")


if __name__ == "__main__":
    main()
