"""
Cross-architecture compression experiment.

Question: How small can a student network be while still perfectly mimicking
a randomly-initialized teacher network of a given size?

Setup:
  - Teacher: 1-hidden-layer MLP with H_t ReLU units, random weights, frozen.
  - Student: 1-hidden-layer MLP with H_s ReLU units, trained to copy teacher
    outputs on random Gaussian inputs.
  - Vary H_t and H_s independently.
  - Measure NMSE (normalized MSE) on a held-out test set.
    NMSE = 1.0 means "as bad as predicting the teacher's mean".
    NMSE ~ 0 means "perfect mimicry".

Pure NumPy. No PyTorch needed.
"""

import numpy as np
import json
import time

# ---------- tiny MLP ----------

def init_mlp(in_dim, hidden, out_dim, rng):
    W1 = rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(out_dim)
    return [W1, b1, W2, b2]

def forward(params, x):
    W1, b1, W2, b2 = params
    h_pre = x @ W1 + b1
    h = np.maximum(0.0, h_pre)
    out = h @ W2 + b2
    return out, (x, h_pre, h)

def backward(params, cache, dout):
    W1, b1, W2, b2 = params
    x, h_pre, h = cache
    dW2 = h.T @ dout
    db2 = dout.sum(axis=0)
    dh = dout @ W2.T
    dh_pre = dh * (h_pre > 0)
    dW1 = x.T @ dh_pre
    db1 = dh_pre.sum(axis=0)
    return [dW1, db1, dW2, db2]

# ---------- Adam ----------

class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ---------- mimicry training ----------

def train_student(teacher, student_size, in_dim, rng, epochs=2500, lr=0.02,
                  n_train=2048, n_test=2048):
    X = rng.standard_normal((n_train, in_dim)).astype(np.float64)
    y, _ = forward(teacher, X)

    student = init_mlp(in_dim, student_size, 1, rng)
    opt = Adam(student, lr=lr)

    for epoch in range(epochs):
        pred, cache = forward(student, X)
        diff = pred - y
        dout = (2.0 / X.shape[0]) * diff
        grads = backward(student, cache, dout)
        opt.step(grads)

    X_test = rng.standard_normal((n_test, in_dim)).astype(np.float64)
    y_test, _ = forward(teacher, X_test)
    pred_test, _ = forward(student, X_test)
    mse = float(np.mean((pred_test - y_test) ** 2))
    var = float(np.var(y_test))
    nmse = mse / var if var > 1e-12 else mse
    return nmse

# ---------- experiment ----------

def run(in_dim=2, sizes=(1, 2, 3, 4, 5, 6, 8, 10), n_teachers=5, n_student_runs=3):
    rng = np.random.default_rng(0)
    table = {}
    t0 = time.time()
    for ht in sizes:
        for hs in sizes:
            losses = []
            for ti in range(n_teachers):
                t_rng = np.random.default_rng(1000 + ti * 31 + ht * 7)
                teacher = init_mlp(in_dim, ht, 1, t_rng)
                # 'effective' rank of teacher: drop dead/duplicate units
                best = float("inf")
                for si in range(n_student_runs):
                    s_rng = np.random.default_rng(5000 + ti * 97 + hs * 13 + si * 3)
                    nmse = train_student(teacher, hs, in_dim, s_rng)
                    if nmse < best:
                        best = nmse
                losses.append(best)
            table[(ht, hs)] = float(np.mean(losses))
            print(f"  teacher={ht:2d} student={hs:2d}  NMSE={table[(ht,hs)]:.4f}")
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    return table, sizes

def threshold_per_teacher(table, sizes, eps=0.01):
    """For each teacher, smallest student that achieves NMSE < eps."""
    out = {}
    for ht in sizes:
        smallest = None
        for hs in sizes:
            if table[(ht, hs)] < eps:
                smallest = hs
                break
        out[ht] = smallest
    return out

if __name__ == "__main__":
    table, sizes = run()

    print("\n=== NMSE matrix (rows=teacher H_t, cols=student H_s) ===")
    header = "       " + "".join(f"  s={s:<3d}" for s in sizes)
    print(header)
    for ht in sizes:
        row = f"  t={ht:<3d}" + "".join(f"  {table[(ht,hs)]:.3f}" for hs in sizes)
        print(row)

    print("\n=== Smallest student that mimics teacher (NMSE < 0.01) ===")
    thr = threshold_per_teacher(table, sizes, eps=0.01)
    for ht, hs in thr.items():
        print(f"  teacher H_t={ht:<3d}  ->  needs student H_s>={hs}")

    print("\n=== Smallest student for NMSE < 0.001 ===")
    thr2 = threshold_per_teacher(table, sizes, eps=0.001)
    for ht, hs in thr2.items():
        print(f"  teacher H_t={ht:<3d}  ->  needs student H_s>={hs}")

    out = {
        "sizes": list(sizes),
        "table": {f"{ht},{hs}": v for (ht, hs), v in table.items()},
        "threshold_001": {str(k): v for k, v in threshold_per_teacher(table, sizes, 0.01).items()},
        "threshold_0001": {str(k): v for k, v in threshold_per_teacher(table, sizes, 0.001).items()},
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results.json")
