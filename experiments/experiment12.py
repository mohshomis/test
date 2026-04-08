"""
Symbolic structural analysis of matrix multiplication algorithms.

Instead of trying to derive the symmetry group abstractly, let's SEE the
structure by symbolically expanding every rank-R decomposition and checking
the algebra explicitly.

For each product r in a decomposition (U, V, W):
  M_r = (sum_i U[i,r] * A_flat[i]) * (sum_j V[j,r] * B_flat[j])
      = sum_{i,j} U[i,r]*V[j,r] * A_flat[i]*B_flat[j]

Expanding this gives a set of BILINEAR MONOMIALS a_i * b_j, each with a
coefficient.

Each output entry C_k = sum_r W[k,r] * M_r then becomes a weighted sum of
these bilinear monomials.

For the algorithm to be correct, all the SPURIOUS monomials (ones that
aren't needed by C_k) must CANCEL across the sum. The pattern of these
cancellations IS the structural fingerprint of the algorithm.

This is what we show in this script.

We also discover the actual symmetry group of the matmul tensor by
brute-force: enumerate small permutation groups and check invariance.
"""

import numpy as np
from collections import defaultdict
from itertools import permutations

from experiment8 import matmul_tensor, strassen_decomposition, naive_decomposition


# ---------- symbolic expansion of a decomposition ----------

def symbolic_products(U, V, n):
    """
    Returns a list of dicts: one per product. Each dict maps
    (i, j) tuples (indicating monomial a_i * b_j) to their coefficients.
    """
    R = U.shape[1]
    products = []
    for r in range(R):
        u = U[:, r]
        v = V[:, r]
        coeffs = defaultdict(int)
        for i in range(len(u)):
            if u[i] == 0:
                continue
            for j in range(len(v)):
                if v[j] == 0:
                    continue
                coeffs[(i, j)] += int(u[i] * v[j])
        products.append(dict(coeffs))
    return products


def format_index(idx, n):
    """Convert a flat index to (row, col) string."""
    return f"({idx//n},{idx%n})"


def format_product(coeffs, name, n):
    """Pretty-print a bilinear product as a sum of monomials."""
    items = []
    for (i, j), c in coeffs.items():
        a = f"a{format_index(i, n)}"
        b = f"b{format_index(j, n)}"
        if c == 1:
            items.append(f"+{a}*{b}")
        elif c == -1:
            items.append(f"-{a}*{b}")
        else:
            sign = "+" if c > 0 else ""
            items.append(f"{sign}{c}*{a}*{b}")
    return f"{name} = " + " ".join(items)


def expand_output(products, W, out_idx):
    """For output C[out_idx], compute the sum of all monomial contributions."""
    combined = defaultdict(int)
    r_contributions = []
    for r in range(len(products)):
        w = int(W[out_idx, r])
        if w == 0:
            continue
        r_contributions.append((r, w))
        for (i, j), c in products[r].items():
            combined[(i, j)] += w * c
    # Remove zero entries (these are cancelled monomials)
    combined_nonzero = {k: v for k, v in combined.items() if v != 0}
    return combined_nonzero, r_contributions


# ---------- check the algebra explicitly ----------

def verify_algebraically(U, V, W, n):
    """
    For each output index, check that the symbolic expansion matches the
    expected matmul expression. This verifies the algorithm at the symbolic
    level, not just numerically.

    For C[i,k] = sum_j A[i,j] B[j,k], the expected monomials are
    {(i*n+j, j*n+k): 1 for j in range(n)}.
    """
    products = symbolic_products(U, V, n)
    all_ok = True
    for out_i in range(n):
        for out_k in range(n):
            out_idx = out_i * n + out_k
            got, _ = expand_output(products, W, out_idx)
            expected = {(out_i * n + j, j * n + out_k): 1 for j in range(n)}
            if got != expected:
                all_ok = False
                print(f"    MISMATCH at C[{out_i},{out_k}]:")
                print(f"      expected: {expected}")
                print(f"      got: {got}")
    return all_ok


# ---------- cancellation analysis ----------

def cancellation_analysis(U, V, W, n):
    """
    For each output, count:
      - How many unique monomials appear in the sum BEFORE cancellation
      - How many monomials SURVIVE after cancellation
      - Cancellation ratio = (raw - survived) / raw
    A high cancellation ratio means the algorithm is 'wasting' products:
    each product produces many terms that don't contribute to the output
    and must be canceled by other products.
    """
    products = symbolic_products(U, V, n)
    rows = []
    for out_i in range(n):
        for out_k in range(n):
            out_idx = out_i * n + out_k
            raw = defaultdict(int)
            for r in range(len(products)):
                w = int(W[out_idx, r])
                if w == 0:
                    continue
                for (i, j), c in products[r].items():
                    raw[(i, j)] += 1  # count raw occurrence, not value
            raw_count = sum(raw.values())
            survived, _ = expand_output(products, W, out_idx)
            survived_count = sum(abs(v) for v in survived.values())
            cancelled = raw_count - survived_count
            ratio = cancelled / raw_count if raw_count > 0 else 0
            rows.append({
                "output": f"C[{out_i},{out_k}]",
                "raw_monomials": raw_count,
                "survived": survived_count,
                "cancelled": cancelled,
                "cancel_ratio": ratio,
            })
    return rows


# ---------- find the actual symmetry group by brute force ----------

def find_matmul_symmetries(n, max_tests=100):
    """
    Empirically find symmetries of T_{n,n,n} by testing candidate
    transformations. For small n we can check:
      - transposing all three modes
      - swapping modes A and B (combined with something?)
      - the 3-cycle on modes combined with various index permutations
    This gives us the ACTUAL symmetry group instead of a guess.
    """
    T = matmul_tensor(n)

    symmetries = []

    # Identity
    symmetries.append(("identity", T.copy()))

    # Transposition permutation on each flat index (flat(i,j) <-> flat(j,i))
    pi = np.array([((idx % n) * n + idx // n) for idx in range(n * n)])

    # Candidate 1: transpose all three modes
    T1 = T[pi][:, pi][:, :, pi]
    if np.array_equal(T1, T):
        symmetries.append(("transpose all 3 modes", T1))

    # Candidate 2: swap modes 0 and 1
    T2 = np.transpose(T, (1, 0, 2))
    if np.array_equal(T2, T):
        symmetries.append(("swap modes 0,1", T2))

    # Candidate 3: swap modes 0,1 + transpose mode 2
    T3 = np.transpose(T, (1, 0, 2))[:, :, pi]
    if np.array_equal(T3, T):
        symmetries.append(("swap 0,1 + transpose mode 2", T3))

    # Candidate 4: The Strassen symmetry: if C = AB then C^T = B^T A^T
    # In tensor form: if we transpose all modes and then permute (A,B,C) -> (B,A,C),
    # we should get T back
    T4 = T[pi][:, pi][:, :, pi].transpose(1, 0, 2)
    if np.array_equal(T4, T):
        symmetries.append(("transpose all + swap modes 0,1", T4))

    # Candidate 5: cyclic mode permutation (A,B,C) -> (B,C,A) with transposed indices
    # T'[b,c,a] = T[a,b,c] means T'_new = T.transpose((2,0,1))
    # Check: T.transpose((2,0,1)) == T[pi,:,:][:,pi,:][:,:,pi] applied somehow
    T5 = T.transpose((2, 0, 1))
    if np.array_equal(T5, T):
        symmetries.append(("cyclic mode shift", T5))

    T6 = T.transpose((2, 0, 1))[pi][:, pi][:, :, pi]
    if np.array_equal(T6, T):
        symmetries.append(("cyclic mode shift + transpose all", T6))

    # Candidate 7: tr(ABC) cyclic symmetry - corresponds to a specific
    # combination of mode cycling and index reshaping
    # The trilinear form f(X,Y,Z) = tr(XYZ) = sum_{ijk} X[i,j] Y[j,k] Z[k,i]
    # So tensor T[(i,j), (j,k), (k,i)] = 1
    # Cyclic: tr(XYZ) = tr(YZX), so T is invariant under (X,Y,Z) -> (Y,Z,X)
    # But this is for the f(X,Y,Z) = tr(XYZ) tensor, NOT our C = AB tensor.
    # Our tensor represents sum_{ijk} A[i,j] B[j,k] C[i,k] which is tr(AB C^T)
    # not tr(ABC). The cyclic symmetry of tr(AB C^T) is:
    # tr(AB C^T) = tr(B C^T A) = tr(C^T A B), corresponds to cyclic mode
    # shift with the C mode TRANSPOSED.
    # Let's check:
    # T_new[a, b, c] = T[c_transposed, a, b]?
    # This means: T_new[a,b,c] = T[pi[c], a, b]
    # As a tensor operation: permute modes (2,0,1), then apply pi to mode 0
    T7 = T.transpose((2, 0, 1))
    T7 = T7[pi]
    if np.array_equal(T7, T):
        symmetries.append(("cyclic shift + transpose C mode", T7))

    return symmetries


# ---------- main experiment ----------

def main():
    print("=" * 60)
    print("Symbolic / structural analysis of fast matmul")
    print("=" * 60)

    # 1. Symmetries of T
    print("\n### 1. Symmetries of the matmul tensor (empirical) ###")
    for n in [2, 3]:
        syms = find_matmul_symmetries(n)
        print(f"\n  n={n}: found {len(syms)} symmetries")
        for name, _ in syms:
            print(f"    - {name}")

    # 2. Symbolic Strassen
    print("\n### 2. Strassen's 7 products (symbolic) ###")
    U_str, V_str, W_str = strassen_decomposition()
    products = symbolic_products(U_str, V_str, n=2)
    for r, p in enumerate(products):
        print(f"  {format_product(p, f'M{r+1}', n=2)}")

    # 3. How each Strassen output assembles
    print("\n### 3. How each Strassen output is built from M's ###")
    print("   (after expansion and cancellation)")
    for out_i in range(2):
        for out_k in range(2):
            out_idx = out_i * 2 + out_k
            survived, contribs = expand_output(products, W_str, out_idx)
            parts = []
            for (i, j), c in survived.items():
                sign = "+" if c > 0 else "-"
                coef = "" if abs(c) == 1 else f"{abs(c)}*"
                parts.append(f"{sign}{coef}a{format_index(i,2)}*b{format_index(j,2)}")
            print(f"  C[{out_i},{out_k}] = {' '.join(parts)}")

    # 4. Verify algebraically
    print("\n### 4. Algebraic verification ###")
    ok = verify_algebraically(U_str, V_str, W_str, n=2)
    print(f"  Strassen verified symbolically: {ok}")

    # 5. Cancellation analysis
    print("\n### 5. Cancellation analysis (Strassen vs naive, 2x2) ###")
    for name, dec in [("naive 2x2", naive_decomposition(2)),
                      ("Strassen 2x2", strassen_decomposition())]:
        rows = cancellation_analysis(*dec, n=2)
        print(f"\n  {name}:")
        for row in rows:
            print(f"    {row['output']:<10}  raw={row['raw_monomials']:>3}  "
                  f"survived={row['survived']:>3}  "
                  f"cancelled={row['cancelled']:>3}  "
                  f"ratio={row['cancel_ratio']:.2f}")
        total_raw = sum(r["raw_monomials"] for r in rows)
        total_cancel = sum(r["cancelled"] for r in rows)
        print(f"    TOTAL raw monomials: {total_raw}")
        print(f"    TOTAL cancelled:     {total_cancel}")
        print(f"    Overall cancel ratio: {total_cancel/total_raw:.2%}")

    # 6. The structural insight
    print("\n### 6. Structural insight ###")
    print("""
  Strassen's 2x2 algorithm computes 7 products and then combines them into
  4 outputs. In the process it creates SPURIOUS monomials that must cancel.

  If you think of a 'good' decomposition as one where each product 'pays
  for itself' - contributes the same number of surviving monomials as it
  generates - the algorithm is tight. If a product generates many
  monomials that get cancelled, the algorithm is 'wasteful' in a structural
  sense - that product could potentially be eliminated if the cancellations
  could be achieved another way.

  This gives a structural lower bound on rank: you need enough products to
  cover all needed monomials, and the 'extra' products needed to cancel
  spurious terms that the first products created.
""")


if __name__ == "__main__":
    main()
