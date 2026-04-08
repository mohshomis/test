"""
Stronger ALS search for 3x3 matmul tensor decomposition.

experiment9 showed that basic ALS converges at rank 7 for 2x2 (matching
Strassen) but fails even at rank 27 for 3x3. That's a sign our optimizer
is getting stuck in local minima, not that the decomposition doesn't exist.

This script hardens the search by:
  1. Much more restarts with diverse initializations.
  2. Longer iterations with adaptive stopping.
  3. Warm-starting from a known-valid decomposition (the naive one,
     perturbed) to see if that helps escape bad basins.
  4. Testing at several ranks around the known bounds.

Scientific questions:
  Q1. Does beefed-up ALS converge at R=27 for 3x3? (Should trivially, it's naive.)
  Q2. Does it converge at R=23 (Laderman's bound)?
  Q3. Does it get close at R=22 or R=21? (Where the true rank might live.)
  Q4. At what R does the residual floor clearly separate from zero?

Honest expectations:
  - R=27 should converge (basic sanity).
  - R=23 might converge with enough restarts (Laderman is known).
  - R<23 almost certainly will NOT converge. If it did, that would be a
    major result, but real-valued tensor decomposition of matmul tensors
    has been studied a lot and nobody has found continuous rank < 23 for 3x3
    via ALS-type methods.

We are not claiming to beat state-of-the-art. We are verifying that our
framework actually works on the concrete test case it was built for.
"""

import numpy as np
import json
import time

from experiment8 import matmul_tensor, naive_decomposition, tensor_rank_apply


def als_decomposition_strong(T, R, init=None, max_iters=5000, tol=1e-11,
                              reg=0.0, verbose=False):
    """
    Single-run ALS with optional warm start and L2 regularization.
    """
    d1, d2, d3 = T.shape
    T = T.astype(np.float64)
    T_norm = np.linalg.norm(T)
    T1 = T.reshape(d1, d2 * d3)
    T2 = T.transpose(1, 0, 2).reshape(d2, d1 * d3)
    T3 = T.transpose(2, 0, 1).reshape(d3, d1 * d2)

    if init is None:
        U = np.random.standard_normal((d1, R)) * 0.3
        V = np.random.standard_normal((d2, R)) * 0.3
        W = np.random.standard_normal((d3, R)) * 0.3
    else:
        U, V, W = [x.copy().astype(np.float64) for x in init]

    def solve_mode(KR, T_mode):
        if reg > 0:
            A = KR.T @ KR + reg * np.eye(R)
            b = KR.T @ T_mode.T
            X = np.linalg.solve(A, b)
            return X.T
        X, *_ = np.linalg.lstsq(KR, T_mode.T, rcond=None)
        return X.T

    prev_res = np.inf
    stagnant = 0
    for it in range(max_iters):
        KR = np.einsum('kr,jr->jkr', W, V).reshape(d2 * d3, R)
        U = solve_mode(KR, T1)
        KR = np.einsum('kr,ir->ikr', W, U).reshape(d1 * d3, R)
        V = solve_mode(KR, T2)
        KR = np.einsum('jr,ir->ijr', V, U).reshape(d1 * d2, R)
        W = solve_mode(KR, T3)

        T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
        res = np.linalg.norm(T - T_hat) / T_norm

        if verbose and it % 500 == 0:
            print(f"      iter {it:>4}: res={res:.3e}")

        if res < tol:
            return (U, V, W), res
        if abs(prev_res - res) < 1e-15:
            stagnant += 1
            if stagnant > 50:
                return (U, V, W), res
        else:
            stagnant = 0
        prev_res = res

    return (U, V, W), res


def search_at_rank(T, R, n_restarts=30, max_iters=4000, verbose=False):
    """Many restarts, return best result found."""
    best = np.inf
    best_UVW = None
    t0 = time.time()
    for restart in range(n_restarts):
        np.random.seed(42 + restart * 113 + R * 7)
        scale = 0.1 * (1 + (restart % 5))
        U0 = np.random.standard_normal((T.shape[0], R)) * scale
        V0 = np.random.standard_normal((T.shape[1], R)) * scale
        W0 = np.random.standard_normal((T.shape[2], R)) * scale
        (U, V, W), res = als_decomposition_strong(
            T, R, init=(U0, V0, W0), max_iters=max_iters
        )
        if res < best:
            best = res
            best_UVW = (U.copy(), V.copy(), W.copy())
            if verbose:
                print(f"    restart {restart}: res={res:.3e} (new best, "
                      f"{time.time()-t0:.1f}s)")
        if best < 1e-9:
            break
    return best_UVW, best, time.time() - t0


def verify_on_matrices(U, V, W, n, n_trials=30):
    rng = np.random.default_rng(7777)
    max_err = 0.0
    for _ in range(n_trials):
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        C_true = A @ B
        C_test = tensor_rank_apply(U, V, W, A, B)
        max_err = max(max_err, float(np.max(np.abs(C_true - C_test))))
    return max_err


def main():
    print("=" * 60)
    print("Stronger ALS for 3x3 matrix multiplication tensor")
    print("=" * 60)

    T3 = matmul_tensor(3)
    results = []

    # First test: can we warm-start from naive at R=27 and converge exactly?
    print("\n--- Sanity: warm-start from naive decomposition at R=27 ---")
    U_naive, V_naive, W_naive = naive_decomposition(3)
    U0 = U_naive.astype(np.float64) + np.random.randn(*U_naive.shape) * 0.01
    V0 = V_naive.astype(np.float64) + np.random.randn(*V_naive.shape) * 0.01
    W0 = W_naive.astype(np.float64) + np.random.randn(*W_naive.shape) * 0.01
    (U, V, W), res = als_decomposition_strong(T3, 27, init=(U0, V0, W0),
                                              max_iters=3000, verbose=True)
    print(f"  Warm-started R=27 residual: {res:.3e}")
    err = verify_on_matrices(U, V, W, n=3) if res < 1e-6 else None
    if err is not None:
        print(f"  Matmul verification max error: {err:.3e}")

    # Now: aggressive search at target ranks
    ranks_to_try = [27, 25, 23, 22, 21, 20, 19]
    print(f"\n--- Aggressive search at ranks: {ranks_to_try} ---")

    for R in ranks_to_try:
        print(f"\n  Rank R = {R}")
        best_UVW, best_res, dt = search_at_rank(
            T3, R, n_restarts=40, max_iters=3000, verbose=False
        )
        success = best_res < 1e-8
        marker = "FOUND" if success else " no  "
        print(f"    best residual over 40 restarts = {best_res:.3e}  "
              f"{marker}  ({dt:.1f}s)")

        # Verify if successful
        matmul_err = None
        if success and best_UVW is not None:
            matmul_err = verify_on_matrices(*best_UVW, n=3)
            print(f"    matmul verification max error: {matmul_err:.3e}")

        results.append({
            "R": R,
            "best_residual": float(best_res),
            "success": bool(success),
            "matmul_error": float(matmul_err) if matmul_err is not None else None,
            "time_seconds": dt,
        })

    with open("results_matmul_search_strong.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_matmul_search_strong.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'R':>4}  {'residual':>12}  {'converged':>10}")
    for r in results:
        print(f"{r['R']:>4}  {r['best_residual']:>12.3e}  "
              f"{'yes' if r['success'] else 'no':>10}")


if __name__ == "__main__":
    main()
