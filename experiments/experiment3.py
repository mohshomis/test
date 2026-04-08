"""
Selection vs Optimization experiment.

Question:
  Can you 'train' a neural network by SELECTING from a precomputed library
  of random tiny networks, instead of by gradient descent?

Setup:
  - Build a LIBRARY of N random tiny ReLU networks (the 'atoms').
    Each atom is a frozen function R^2 -> R.
  - For a target function y(x), find scalar coefficients c_i such that
        sum_i  c_i * atom_i(x)  ≈  y(x)
    This is just linear regression in atom-space - no backprop.
  - Compare to a standard gradient-descent-trained ReLU MLP of comparable
    trainable-parameter budget on the same target.

Three selection variants:
  1. Plain least squares (LS) over all atoms.
  2. Ridge regression (RR) over all atoms.
  3. Orthogonal Matching Pursuit (OMP) - greedily pick k atoms (sparse).

Targets:
  T1. A held-out random ReLU teacher (size 4 hidden).
  T2. A held-out random ReLU teacher (size 8 hidden).
  T3. y = x1 * x2     (simple multiplicative interaction)
  T4. y = sin(x1) + cos(x2)   (smooth analytic)
  T5. y = (x1 + x2 > 0)       (step / threshold)
"""

import numpy as np
import json
import time

from experiment import init_mlp, forward, backward, Adam


# ---------- library ----------

def make_library(n_atoms, in_dim=2, hidden=4, seed=0):
    rng = np.random.default_rng(seed)
    atoms = []
    for _ in range(n_atoms):
        atoms.append(init_mlp(in_dim, hidden, 1, rng))
    return atoms

def atom_matrix(atoms, X):
    """Columns = atom outputs on X. Shape (n_samples, n_atoms)."""
    cols = []
    for a in atoms:
        out, _ = forward(a, X)
        cols.append(out[:, 0])
    return np.stack(cols, axis=1)


# ---------- selection methods ----------

def fit_ls(A, y):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c

def fit_ridge(A, y, alpha=1e-2):
    AtA = A.T @ A
    n = AtA.shape[0]
    c = np.linalg.solve(AtA + alpha * np.eye(n), A.T @ y)
    return c

def fit_omp(A, y, k):
    n, m = A.shape
    residual = y.copy()
    selected = []
    norms = np.linalg.norm(A, axis=0) + 1e-12
    for _ in range(k):
        scores = np.abs(A.T @ residual) / norms
        for s in selected:
            scores[s] = -1.0
        idx = int(np.argmax(scores))
        selected.append(idx)
        A_sel = A[:, selected]
        c_sel, *_ = np.linalg.lstsq(A_sel, y, rcond=None)
        residual = y - A_sel @ c_sel
    c = np.zeros(m)
    c[selected] = c_sel
    return c, selected


# ---------- gradient descent baseline ----------

def gd_train(X_train, y_train, hidden, in_dim=2, epochs=2500, lr=0.02, seed=0):
    rng = np.random.default_rng(seed)
    student = init_mlp(in_dim, hidden, 1, rng)
    opt = Adam(student, lr=lr)
    y_train_2d = y_train.reshape(-1, 1)
    n = X_train.shape[0]
    for _ in range(epochs):
        pred, cache = forward(student, X_train)
        diff = pred - y_train_2d
        dout = (2.0 / n) * diff
        grads = backward(student, cache, dout)
        opt.step(grads)
    return student

def gd_predict(student, X):
    pred, _ = forward(student, X)
    return pred[:, 0]


# ---------- metric ----------

def nmse(y_pred, y_true):
    mse = float(np.mean((y_pred - y_true) ** 2))
    var = float(np.var(y_true))
    return mse / var if var > 1e-12 else mse


# ---------- targets ----------

def make_targets(rng):
    targets = {}

    # T1, T2: held-out random teachers
    t_rng = np.random.default_rng(7777)
    teacher4 = init_mlp(2, 4, 1, t_rng)
    targets["T1_teacher_h4"] = lambda X, T=teacher4: forward(T, X)[0][:, 0]

    t_rng2 = np.random.default_rng(8888)
    teacher8 = init_mlp(2, 8, 1, t_rng2)
    targets["T2_teacher_h8"] = lambda X, T=teacher8: forward(T, X)[0][:, 0]

    # T3: x1 * x2
    targets["T3_x1_times_x2"] = lambda X: X[:, 0] * X[:, 1]

    # T4: sin(x1) + cos(x2)
    targets["T4_sin_cos"] = lambda X: np.sin(X[:, 0]) + np.cos(X[:, 1])

    # T5: step (x1 + x2 > 0)
    targets["T5_step"] = lambda X: (X[:, 0] + X[:, 1] > 0).astype(np.float64)

    return targets


# ---------- run experiment ----------

def run():
    rng = np.random.default_rng(42)
    in_dim = 2

    # Build a library of 1000 random tiny ReLU atoms (each 2->4->1)
    print("Building library of 1000 random ReLU atoms...")
    library = make_library(n_atoms=1000, in_dim=in_dim, hidden=4, seed=123)

    # Probe data
    X_train = rng.standard_normal((4000, in_dim))
    X_test = rng.standard_normal((4000, in_dim))

    print("Computing atom matrix on train and test...")
    A_train = atom_matrix(library, X_train)
    A_test = atom_matrix(library, X_test)
    print(f"  A_train shape = {A_train.shape}")

    targets = make_targets(rng)

    # Comparison budgets
    omp_k_values = [5, 10, 20, 50]
    gd_hidden_values = [4, 8, 16, 32]

    results = []
    for tname, tfn in targets.items():
        print(f"\n=== Target: {tname} ===")
        y_train = tfn(X_train)
        y_test = tfn(X_test)

        row = {"target": tname}

        # 1. Plain LS over the full library (1000 atoms, 4000 samples => over-determined)
        t0 = time.time()
        c_ls = fit_ls(A_train, y_train)
        t_ls = time.time() - t0
        nmse_ls_train = nmse(A_train @ c_ls, y_train)
        nmse_ls_test = nmse(A_test @ c_ls, y_test)
        print(f"  LS (1000 atoms)         train NMSE={nmse_ls_train:.4f}  test NMSE={nmse_ls_test:.4f}  ({t_ls:.2f}s)")
        row["ls_test"] = nmse_ls_test
        row["ls_train"] = nmse_ls_train
        row["ls_time"] = t_ls

        # 2. Ridge
        t0 = time.time()
        c_rr = fit_ridge(A_train, y_train, alpha=1e-2)
        t_rr = time.time() - t0
        nmse_rr_test = nmse(A_test @ c_rr, y_test)
        print(f"  Ridge (1000 atoms)      train NMSE={nmse(A_train @ c_rr, y_train):.4f}  test NMSE={nmse_rr_test:.4f}  ({t_rr:.2f}s)")
        row["ridge_test"] = nmse_rr_test
        row["ridge_time"] = t_rr

        # 3. OMP (sparse selection)
        for k in omp_k_values:
            t0 = time.time()
            c_omp, sel = fit_omp(A_train, y_train, k)
            t_omp = time.time() - t0
            nmse_omp_test = nmse(A_test @ c_omp, y_test)
            print(f"  OMP k={k:<3d}              train NMSE={nmse(A_train @ c_omp, y_train):.4f}  test NMSE={nmse_omp_test:.4f}  ({t_omp:.2f}s)")
            row[f"omp_k{k}_test"] = nmse_omp_test
            row[f"omp_k{k}_time"] = t_omp

        # 4. Gradient descent baselines
        for h in gd_hidden_values:
            t0 = time.time()
            best_test = float("inf")
            for seed in range(3):
                student = gd_train(X_train, y_train, hidden=h, seed=seed * 17 + 1)
                yp = gd_predict(student, X_test)
                test_nmse = nmse(yp, y_test)
                if test_nmse < best_test:
                    best_test = test_nmse
            t_gd = time.time() - t0
            print(f"  GD hidden={h:<3d}          best test NMSE={best_test:.4f}  ({t_gd:.2f}s, 3 seeds)")
            row[f"gd_h{h}_test"] = best_test
            row[f"gd_h{h}_time"] = t_gd

        results.append(row)

    return results


def summarize(results):
    print("\n\n========== SUMMARY ==========")
    print("Test NMSE for each method on each target. Lower is better.\n")
    header = (
        f"{'target':<22}"
        f"{'LS_1k':>10}"
        f"{'Ridge_1k':>10}"
        f"{'OMP_5':>10}"
        f"{'OMP_10':>10}"
        f"{'OMP_50':>10}"
        f"{'GD_h4':>10}"
        f"{'GD_h8':>10}"
        f"{'GD_h16':>10}"
        f"{'GD_h32':>10}"
    )
    print(header)
    for r in results:
        print(
            f"{r['target']:<22}"
            f"{r['ls_test']:>10.4f}"
            f"{r['ridge_test']:>10.4f}"
            f"{r['omp_k5_test']:>10.4f}"
            f"{r['omp_k10_test']:>10.4f}"
            f"{r['omp_k50_test']:>10.4f}"
            f"{r['gd_h4_test']:>10.4f}"
            f"{r['gd_h8_test']:>10.4f}"
            f"{r['gd_h16_test']:>10.4f}"
            f"{r['gd_h32_test']:>10.4f}"
        )


if __name__ == "__main__":
    results = run()
    summarize(results)
    with open("results_selection.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_selection.json")
