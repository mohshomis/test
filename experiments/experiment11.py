"""
Structural analysis of the matrix multiplication tensor.

Goal: Instead of searching blindly for low-rank decompositions, study the
*structure* of the matmul tensor directly. Fast matmul breakthroughs
(Strassen 1969, Laderman 1976, Pan 1980, Smirnov 2013, AlphaTensor 2022)
were all found by exploiting one or more of:

  1. Cyclic symmetry: the tensor T_{n,n,n} is invariant under a 3-cycle
     that corresponds to the algebraic identity (AB)^T = B^T A^T.

  2. Block structure: matrix multiplication is recursive. Strassen's rank-7
     applied recursively gives matrices of size 2^k in O(n^log2(7)) ops.

  3. GL(n) x GL(n) x GL(n) symmetry: if (U,V,W) is a decomposition, then
     changing bases A' = PAP^-1 etc gives another valid decomposition.
     This means decompositions come in huge equivalence classes.

  4. The 'contribution map': which products serve which outputs. Sparsity
     patterns in this map reveal structure.

This script:
  - Proves the cyclic symmetry numerically for n=2, 3, 4.
  - Shows Strassen's 7 products broken down by their 'sharing pattern'.
  - Computes how many independent free parameters exist in a rank-R
     decomposition modulo the GL symmetries (gives a reality check on
     how hard the search actually is).
  - Visualizes the contribution graph: which products feed which outputs.
  - Tries to identify 'structural motifs' that any valid decomposition
     must contain.
"""

import numpy as np
from experiment8 import matmul_tensor, strassen_decomposition, naive_decomposition


# ---------- cyclic symmetry of the matmul tensor ----------

def cyclic_permute_tensor(T, n):
    """
    Apply the algebraic cyclic symmetry to T_{n,n,n}.

    Derivation:
      Matrix multiplication C = A @ B is a trilinear form:
          sum_{i,j,k} A[i,j] B[j,k] C'[i,k]    where C' is the dual.
      Transposing: A'B' C where A' = A^T, B' = B^T etc, corresponds to
      permuting the roles (A, B, C) -> (B^T, C^T, A^T) or similar.
      This induces a permutation on the tensor T:
          T[a,b,c] -> T[pi(b), pi(c), pi(a)]
      where pi is the 'transpose' permutation of the n*n flat indices
      that sends flat(i,j) to flat(j,i).

    If T is invariant under this map, the matmul tensor has 3-fold
    cyclic symmetry. We check this numerically.
    """
    # Permutation pi: flat(i,j) -> flat(j,i)
    pi = np.zeros(n * n, dtype=int)
    for i in range(n):
        for j in range(n):
            pi[i * n + j] = j * n + i

    # Cyclic map: T_new[a,b,c] = T[pi[b], pi[c], pi[a]]
    T_new = T[pi][:, pi][:, :, pi]  # apply permutation to each mode first
    # Now cyclic shift the modes
    T_new = np.transpose(T_new, (2, 0, 1))
    return T_new


def verify_cyclic_symmetry(n):
    T = matmul_tensor(n)
    T_cyc = cyclic_permute_tensor(T, n)
    equal = np.array_equal(T, T_cyc)
    return equal


# ---------- contribution analysis of a decomposition ----------

def contribution_matrix(U, V, W):
    """
    For each product r, which A-entries, B-entries, C-entries does it involve?
    Returns boolean masks and summary stats.
    """
    R = U.shape[1]
    A_involve = (U != 0)  # shape (n^2, R)
    B_involve = (V != 0)
    C_involve = (W != 0)
    return A_involve, B_involve, C_involve


def contribution_graph_text(U, V, W, n=2):
    """Print a human-readable table of which product feeds which output."""
    R = U.shape[1]
    n_out = W.shape[0]
    # For each output entry, list the products that touch it
    out_labels = [f"C[{i//n},{i%n}]" for i in range(n_out)]
    print(f"\n{'output':<10}  products")
    for i, label in enumerate(out_labels):
        ps = [r for r in range(R) if W[i, r] != 0]
        signs = [("+" if W[i, r] > 0 else "-") + f"M{r+1}" for r in ps]
        print(f"  {label:<8}  {' '.join(signs)}")


def sharing_distribution(W):
    """How many outputs does each product contribute to?"""
    counts = (W != 0).sum(axis=0)
    return counts


# ---------- GL symmetries of the decomposition ----------

def counting_free_parameters(n, R):
    """
    A rank-R decomposition of T_{n,n,n} has 3 * n^2 * R real parameters
    in (U, V, W). But many of these are redundant due to GL x GL x GL
    symmetry: you can 'gauge away' (n^2 - 1) * 3 parameters per scaling of
    columns, plus the action of GL(n) on each mode.

    This function computes the naive count and a rough 'effective' count
    after modding out symmetries. It's a reality check on how many real
    degrees of freedom you are actually searching over.
    """
    raw = 3 * n * n * R
    # GL(n) on each of the three factors: 3 * n^2 group generators
    gl_sym = 3 * n * n
    # Column rescaling: each column has a scale freedom; (a_r, b_r, c_r)
    # with a_r * b_r * c_r = 1 means 2 free scales per column
    col_scale = 2 * R
    effective = raw - gl_sym - col_scale
    return raw, effective


# ---------- structural constraints on any decomposition ----------

def output_coverage_constraint(n):
    """
    Any rank-R decomposition must cover all n^3 = n*n*n nonzero entries of
    T. Each product (u_r, v_r, w_r) covers the outer-product pattern
    u_r o v_r o w_r, which in the *nonzero* sense has max nnz = (n^2)^3.
    But to actually represent T, the sum of outer products must equal T
    exactly, not just 'cover' the nonzeros.

    Give a simple lower bound via pigeonhole: T has n^3 nonzeros,
    each product contributes 'some' nonzeros to the reconstruction.
    """
    T = matmul_tensor(n)
    nnz = int((T != 0).sum())
    return {
        "nnz_of_T": nnz,
        "n3_check": n ** 3,
        "n_squared": n * n,
    }


# ---------- main experiment ----------

def main():
    print("=" * 60)
    print("Structural analysis of the matmul tensor")
    print("=" * 60)

    # 1. Cyclic symmetry
    print("\n### 1. Cyclic symmetry (from (AB)^T = B^T A^T) ###")
    for n in [2, 3, 4]:
        ok = verify_cyclic_symmetry(n)
        T = matmul_tensor(n)
        print(f"  n={n}: T_{{{n},{n},{n}}} has cyclic symmetry: {ok}  "
              f"(nnz={int((T!=0).sum())}, shape={T.shape})")
    print("  -> The matmul tensor is invariant under a 3-cycle of its modes")
    print("     combined with transposition. Any decomposition can be")
    print("     'cycled' to get another valid one.")

    # 2. Sharing structure of Strassen
    print("\n### 2. Strassen's sharing structure ###")
    U_str, V_str, W_str = strassen_decomposition()
    n_outputs_per_product = sharing_distribution(W_str)
    print(f"  Number of outputs each of the 7 Strassen products contributes to:")
    for r, c in enumerate(n_outputs_per_product):
        print(f"    M{r+1}: contributes to {int(c)} output(s)")
    print(f"  Mean outputs per product: {n_outputs_per_product.mean():.2f}")
    print(f"  Naive would have mean 1.0 (each product -> exactly one output).")

    contribution_graph_text(U_str, V_str, W_str, n=2)

    # 3. Free parameter counting
    print("\n### 3. Free parameters in a rank-R decomposition ###")
    print(f"  {'(n,R)':<10} {'raw params':<14} {'after GL + scaling':<22}")
    for (n, R) in [(2, 7), (2, 8), (3, 23), (3, 21), (3, 19)]:
        raw, eff = counting_free_parameters(n, R)
        print(f"  {f'({n},{R})':<10} {raw:<14} {eff}")
    print("  -> The effective search space grows only linearly in R after")
    print("     accounting for GL(n) symmetries on all three modes.")

    # 4. A structural invariant: total "support size"
    print("\n### 4. Structural invariant: support sums ###")
    for name, dec in [("naive 2x2", naive_decomposition(2)),
                      ("Strassen 2x2", strassen_decomposition()),
                      ("naive 3x3", naive_decomposition(3))]:
        U, V, W = dec
        R = U.shape[1]
        u_sum = int((U != 0).sum())
        v_sum = int((V != 0).sum())
        w_sum = int((W != 0).sum())
        print(f"  {name:<15} R={R:<3}  |U|={u_sum}  |V|={v_sum}  |W|={w_sum}  "
              f"total={u_sum+v_sum+w_sum}")
    print("  -> Strassen trades multiplications for additions. The total")
    print("     'wiring' (sum of nnz) is HIGHER than naive in exchange for")
    print("     fewer expensive scalar multiplications.")

    # 5. The constraint that any rank-R decomposition must satisfy
    print("\n### 5. Hard constraint: exact equality to T ###")
    for n in [2, 3]:
        cc = output_coverage_constraint(n)
        print(f"  n={n}: T has {cc['nnz_of_T']} nonzero entries")
        print(f"         n^3 = {cc['n3_check']} (= number of needed 'atomic' products)")
        print(f"         n^2 = {cc['n_squared']} (= size of each input/output vector)")
    print("  -> Any rank-R decomposition with R < n^3 must encode at least")
    print("     (n^3 - R) nonzero entries of T as *cancellations* between")
    print("     otherwise spurious product terms. That cancellation is the")
    print("     entire source of difficulty.")


if __name__ == "__main__":
    main()
