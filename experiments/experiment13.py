"""
Structural search for fast matrix multiplication.

Key insight from experiment 12:
  - Each "product" in a fast matmul algorithm is a bilinear form:
       M_r = (sum_i u_i * A_flat[i]) * (sum_j v_j * B_flat[j])
  - When expanded, this produces a sum of monomials a_i * b_j.
  - The full algorithm uses sum_r W[k,r] * M_r = the required monomials
    for output k.
  - So each product corresponds to a vector in R^(n^2 * n^2) (its
    "monomial signature"), and the problem of fast matmul becomes:
       Find R rank-1 vectors in R^(n^4) whose LINEAR SPAN contains the
       n^2 specific "output target vectors".

This script:
  1. Enumerates candidate products (u, v) with small support over {-1,0,1}.
  2. Computes each candidate's monomial signature.
  3. Assembles the matrix M whose columns are candidate signatures.
  4. Computes the target vectors Y (one per output, each is the sum of
     the n monomials needed for that output).
  5. Checks: does the column span of M contain Y? If yes, we can solve for
     the output coefficients and the decomposition exists. The *rank* of
     the solution is the minimum number of candidates needed.

For 3x3 with support-2 candidates (each u and v has at most 2 nonzero
{-1, 0, 1} entries), this is a small enough search to brute-force.
"""

import numpy as np
from itertools import combinations, product
from experiment8 import matmul_tensor


# ---------- generate small-support bilinear products ----------

def small_support_vectors(d, max_support, signs=(-1, 1)):
    """Generate all {-1, 0, 1}^d vectors with support at most max_support."""
    vecs = []
    # The zero vector is degenerate; skip.
    for sup_size in range(1, max_support + 1):
        for positions in combinations(range(d), sup_size):
            for sign_assignment in product(signs, repeat=sup_size):
                v = np.zeros(d, dtype=np.int64)
                for pos, s in zip(positions, sign_assignment):
                    v[pos] = s
                vecs.append(v)
    return vecs


def monomial_signature(u, v):
    """
    Outer product u v^T flattened. Entry [i*len(v) + j] gives coefficient
    of the monomial a_i * b_j in the expansion of (u . a)(v . b).
    """
    return np.outer(u, v).flatten()


def target_vectors(n):
    """
    For each of the n^2 output entries C[i,k], the required bilinear
    expression is sum_j a_{i,j} * b_{j,k}. The monomial vector for this
    output has a 1 at position (i*n+j)*(n*n) + (j*n+k) for each j in [n],
    and 0 everywhere else.
    Returns a matrix Y of shape (n^4, n^2) where columns are target vecs.
    """
    Y = np.zeros((n * n * n * n, n * n), dtype=np.int64)
    for i in range(n):
        for k in range(n):
            out_idx = i * n + k  # column in Y
            for j in range(n):
                a_idx = i * n + j       # index of A[i,j] in flat A
                b_idx = j * n + k       # index of B[j,k] in flat B
                pos = a_idx * (n * n) + b_idx
                Y[pos, out_idx] = 1
    return Y


# ---------- build the candidate matrix M ----------

def build_candidate_matrix(n, max_support_u, max_support_v):
    """
    Columns of M are the flattened monomial signatures of all candidate
    products (u, v) with support constraints. Deduplicate columns.
    """
    u_vecs = small_support_vectors(n * n, max_support_u)
    v_vecs = small_support_vectors(n * n, max_support_v)

    # deduplicate up to sign (since -M is the same product with sign flip
    # absorbed into W[k,r])
    seen = set()
    cols = []
    for u in u_vecs:
        for v in v_vecs:
            sig = monomial_signature(u, v)
            # Normalize sign: first nonzero should be positive
            nz = np.flatnonzero(sig)
            if len(nz) == 0:
                continue
            if sig[nz[0]] < 0:
                sig = -sig
            key = tuple(sig.tolist())
            if key in seen:
                continue
            seen.add(key)
            cols.append(sig)
    M = np.array(cols, dtype=np.int64).T  # shape (n^4, num_candidates)
    return M


def target_in_span(M, Y, tol=1e-9):
    """
    Check if all columns of Y are in the column span of M.
    Returns per-output whether it's reachable, and if so, the coefficients.
    """
    # Solve M @ c = y for each column y of Y
    results = []
    for k in range(Y.shape[1]):
        y = Y[:, k].astype(np.float64)
        c, residuals, rank, _ = np.linalg.lstsq(M.astype(np.float64), y, rcond=None)
        # Check residual
        reconstructed = M @ c
        err = float(np.linalg.norm(reconstructed - y))
        results.append({
            "output_k": k,
            "in_span": err < tol,
            "error": err,
            "coeffs": c,
        })
    return results


def minimum_covering_rank(M, Y):
    """
    If Y is in the column span of M, find the MINIMUM number of columns of
    M whose span contains all of Y.

    This is NP-hard in general, but we can compute:
      - A lower bound: rank(Y) (any covering must have at least this many)
      - An upper bound: rank(span(Y) cap col_span(M))
    If the rank of Y is R_Y, then we need at least R_Y columns.
    If the full joint span M @ c = y is solvable for every y in Y with some
    coefficient vector c, then the minimum set of columns is at most
    rank(Y) if we can find rank(Y) columns that span Y exactly.

    Here we compute a simple upper bound via iterative greedy selection.
    """
    d = M.shape[0]
    k = Y.shape[1]
    # First check feasibility
    for col in range(k):
        y = Y[:, col].astype(np.float64)
        c, _, _, _ = np.linalg.lstsq(M.astype(np.float64), y, rcond=None)
        err = float(np.linalg.norm(M @ c - y))
        if err > 1e-6:
            return None, f"output {col} not in column span"

    # Greedy selection: iteratively add the column of M that best reduces
    # the residual rank of the remaining uncovered target.
    M_float = M.astype(np.float64)
    remaining = Y.astype(np.float64).copy()
    selected = []
    while np.linalg.norm(remaining) > 1e-6 and len(selected) < 200:
        # Compute the best single column to add
        best_j = None
        best_reduction = 0.0
        for j in range(M.shape[1]):
            if j in selected:
                continue
            new_set = selected + [j]
            M_sub = M_float[:, new_set]
            # Project Y onto col span of M_sub and measure residual
            Y_float = Y.astype(np.float64)
            C_sub, _, _, _ = np.linalg.lstsq(M_sub, Y_float, rcond=None)
            proj = M_sub @ C_sub
            new_res = np.linalg.norm(Y_float - proj)
            old_res = np.linalg.norm(remaining)
            reduction = old_res - new_res
            if reduction > best_reduction:
                best_reduction = reduction
                best_j = j
        if best_j is None:
            break
        selected.append(best_j)
        M_sub = M_float[:, selected]
        C_sub, _, _, _ = np.linalg.lstsq(M_sub, Y.astype(np.float64), rcond=None)
        remaining = Y.astype(np.float64) - M_sub @ C_sub

    return len(selected), None


# ---------- main ----------

def main():
    print("=" * 60)
    print("Structural search: covering matmul monomials with low-support")
    print("bilinear products")
    print("=" * 60)

    # --- 2x2 first (sanity check) ---
    print("\n### 2x2 matmul ###")
    Y2 = target_vectors(2)
    print(f"  Target matrix Y shape: {Y2.shape}")
    print(f"  Rank of Y: {np.linalg.matrix_rank(Y2)}")
    print("  (rank of Y is a LOWER BOUND on the number of products needed)")

    for (u_sup, v_sup) in [(1, 1), (2, 2)]:
        M = build_candidate_matrix(2, u_sup, v_sup)
        print(f"\n  max_support u={u_sup}, v={v_sup}:")
        print(f"    M shape: {M.shape}  ({M.shape[1]} unique candidates)")
        feasible_results = target_in_span(M, Y2)
        n_reachable = sum(1 for r in feasible_results if r["in_span"])
        print(f"    outputs reachable: {n_reachable}/4")
        if n_reachable == 4:
            min_r, err_msg = minimum_covering_rank(M, Y2)
            if err_msg:
                print(f"    greedy covering: {err_msg}")
            else:
                print(f"    greedy covering rank: {min_r}")

    # --- 3x3 (the real target) ---
    print("\n### 3x3 matmul ###")
    Y3 = target_vectors(3)
    print(f"  Target matrix Y shape: {Y3.shape}")
    print(f"  Rank of Y: {np.linalg.matrix_rank(Y3)}")

    for (u_sup, v_sup) in [(1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]:
        M = build_candidate_matrix(3, u_sup, v_sup)
        print(f"\n  max_support u={u_sup}, v={v_sup}:")
        print(f"    M shape: {M.shape}")
        feasible_results = target_in_span(M, Y3)
        n_reachable = sum(1 for r in feasible_results if r["in_span"])
        print(f"    outputs reachable: {n_reachable}/9")
        if n_reachable == 9:
            min_r, err_msg = minimum_covering_rank(M, Y3)
            if err_msg:
                print(f"    greedy covering: {err_msg}")
            else:
                print(f"    greedy covering rank: {min_r}")


if __name__ == "__main__":
    main()
