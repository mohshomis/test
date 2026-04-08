"""
Composition cost experiment.

Question:
  If a function A has 'essential complexity' N(A) and B has N(B),
  what is the essential complexity of B(A(x))?

  - Sub-additive   N(B o A) < N(A) + N(B):  composition collapses information.
  - Additive       N(B o A) = N(A) + N(B):  composition is neutral.
  - Super-additive N(B o A) > N(A) + N(B):  composition creates emergent
                                            complexity (more than the sum
                                            of the parts).

Setup:
  - A: random ReLU MLP, 2 -> n_a -> 2  (2D->2D)
  - B: random ReLU MLP, 2 -> n_b -> 1  (2D->1D)
  - Composed function:  C(x) = B(A(x))
  - Find smallest student (2 -> H_s -> 1) that mimics each of A, B, C.
  - Compare to the parts and to a single random reference teacher.

We measure 'smallest student' as the smallest H_s that achieves
NMSE < 0.01 on a held-out test set.
"""

import numpy as np
import json
import time

from experiment import init_mlp, forward, backward, Adam


# ---------- training a student against an arbitrary target ----------

def train_student_for_target(X_train, y_train, student_size, in_dim, out_dim,
                             rng, epochs=2500, lr=0.02):
    student = init_mlp(in_dim, student_size, out_dim, rng)
    opt = Adam(student, lr=lr)
    n = X_train.shape[0]
    for _ in range(epochs):
        pred, cache = forward(student, X_train)
        diff = pred - y_train
        dout = (2.0 / n) * diff
        grads = backward(student, cache, dout)
        opt.step(grads)
    return student


def measure_nmse(student, X_test, y_test):
    pred, _ = forward(student, X_test)
    mse = float(np.mean((pred - y_test) ** 2))
    var = float(np.var(y_test))
    return mse / var if var > 1e-12 else mse


def smallest_student(X_train, y_train, X_test, y_test, in_dim, out_dim,
                     sizes, rng, eps=0.01, n_runs=2):
    """Returns smallest student size achieving NMSE < eps, plus full curve."""
    curve = {}
    for s in sizes:
        best = float("inf")
        for r in range(n_runs):
            seed = int(rng.integers(0, 1 << 30))
            sub_rng = np.random.default_rng(seed)
            student = train_student_for_target(
                X_train, y_train, s, in_dim, out_dim, sub_rng
            )
            nmse = measure_nmse(student, X_test, y_test)
            if nmse < best:
                best = nmse
        curve[s] = best
    smallest = None
    for s in sorted(sizes):
        if curve[s] < eps:
            smallest = s
            break
    return smallest, curve


# ---------- composition ----------

def composed_outputs(A, B, X):
    z, _ = forward(A, X)
    y, _ = forward(B, z)
    return y


# ---------- experiment ----------

def run():
    rng = np.random.default_rng(42)
    sizes = [1, 2, 3, 4, 5, 6, 8, 10]
    pairs = [(2, 2), (3, 3), (4, 4), (2, 4), (4, 2)]
    n_seeds = 4
    eps = 0.01

    rows = []
    t0 = time.time()
    for (na, nb) in pairs:
        for seed in range(n_seeds):
            t_rng_a = np.random.default_rng(1000 + seed * 7 + na * 13)
            t_rng_b = np.random.default_rng(2000 + seed * 11 + nb * 17)
            A = init_mlp(2, na, 2, t_rng_a)
            B = init_mlp(2, nb, 1, t_rng_b)

            data_rng = np.random.default_rng(3000 + seed * 5 + na * 19 + nb * 23)
            X_train = data_rng.standard_normal((2048, 2))
            X_test = data_rng.standard_normal((2048, 2))

            # Composition target
            y_train_C = composed_outputs(A, B, X_train)
            y_test_C = composed_outputs(A, B, X_test)
            n_C, curve_C = smallest_student(
                X_train, y_train_C, X_test, y_test_C, 2, 1, sizes, rng, eps=eps
            )

            # A alone (2->2)
            y_train_A, _ = forward(A, X_train)
            y_test_A, _ = forward(A, X_test)
            n_A, curve_A = smallest_student(
                X_train, y_train_A, X_test, y_test_A, 2, 2, sizes, rng, eps=eps
            )

            # B alone (2->1) -- input is the latent space, not original input.
            # We give it Gaussian inputs in its own input space (2D), since that
            # is the canonical setting.
            X_train_B = data_rng.standard_normal((2048, 2))
            X_test_B = data_rng.standard_normal((2048, 2))
            y_train_B, _ = forward(B, X_train_B)
            y_test_B, _ = forward(B, X_test_B)
            n_B, curve_B = smallest_student(
                X_train_B, y_train_B, X_test_B, y_test_B, 2, 1, sizes, rng, eps=eps
            )

            row = {
                "na": na, "nb": nb, "seed": seed,
                "n_A": n_A, "n_B": n_B, "n_C": n_C,
                "sum": (n_A or 99) + (n_B or 99),
                "max": max(n_A or 99, n_B or 99),
                "curve_C": curve_C,
            }
            rows.append(row)
            print(f"  pair=({na},{nb}) seed={seed}  "
                  f"n_A={n_A} n_B={n_B} n_C={n_C}  "
                  f"sum={row['sum']} max={row['max']}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    return rows, pairs, sizes


def _coerce(x, sentinel=11):
    """None (no sufficient student found at any tested size) -> sentinel."""
    return sentinel if x is None else x


def summarize(rows, pairs):
    print("\n=== Composition cost summary ===")
    print("  pair          mean(n_A)  mean(n_B)  mean(n_C)  mean(n_A+n_B)   verdict")
    out_summary = []
    for (na, nb) in pairs:
        sub = [r for r in rows if r["na"] == na and r["nb"] == nb]
        mA = np.mean([_coerce(r["n_A"]) for r in sub])
        mB = np.mean([_coerce(r["n_B"]) for r in sub])
        mC = np.mean([_coerce(r["n_C"]) for r in sub])
        mSum = np.mean([_coerce(r["n_A"]) + _coerce(r["n_B"]) for r in sub])
        if mC < mSum - 0.5:
            verdict = "SUB-additive"
        elif mC > mSum + 0.5:
            verdict = "SUPER-additive"
        else:
            verdict = "additive"
        print(f"  ({na},{nb})        {mA:.2f}       {mB:.2f}       {mC:.2f}       {mSum:.2f}        {verdict}")
        out_summary.append({
            "pair": [na, nb], "mean_n_A": mA, "mean_n_B": mB,
            "mean_n_C": mC, "mean_sum": mSum, "verdict": verdict,
        })
    return out_summary


if __name__ == "__main__":
    rows, pairs, sizes = run()
    summary = summarize(rows, pairs)

    out = {
        "sizes": sizes,
        "rows": [{k: (v if not isinstance(v, dict)
                       else {str(kk): vv for kk, vv in v.items()})
                  for k, v in r.items()} for r in rows],
        "summary": summary,
    }
    with open("results_composition.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results_composition.json")
