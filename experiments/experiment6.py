"""
Concept multiplicity vs adversarial robustness.

Hypothesis (built on Finding 1):
  Higher dream attractor count -> more competing internal templates per class
  -> more 'weak points' an adversary can exploit -> LESS robust to FGSM.
  Lower attractor count -> more unified internal concepts -> MORE robust.

Setup:
  - Train classifiers on the 3-class blob task with different training
    knobs that should change concept multiplicity:
      * weight decay (L2) at several strengths
      * training set size at several scales
      * training epochs at several lengths
  - For each trained model, measure both:
      * Dream attractor count (mean across classes) - the multiplicity
      * Robustness under FGSM attack at multiple epsilons
  - Compute Pearson and Spearman correlations between multiplicity and
    robustness across all trained models.

If multiplicity is a meaningful predictor of robustness, the correlation
will be clearly negative (more attractors -> lower robustness).
If not, we have a real null result on a fresh hypothesis.
"""

import numpy as np
import json
import time

from experiment import init_mlp, forward, backward, Adam
from experiment4 import make_dataset, dream, cluster, softmax


# ---------- training with weight decay ----------

def train_with_wd(X, y, hidden, weight_decay, epochs=2500, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    net = init_mlp(2, hidden, n_classes, rng)
    opt = Adam(net, lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        logits, cache = forward(net, X)
        p = softmax(logits)
        dout = p.copy()
        dout[np.arange(n), y] -= 1
        dout /= n
        grads = backward(net, cache, dout)
        # L2 weight decay on weight matrices only (not biases)
        grads[0] = grads[0] + weight_decay * net[0]
        grads[2] = grads[2] + weight_decay * net[2]
        opt.step(grads)
    logits, _ = forward(net, X)
    train_acc = float((logits.argmax(axis=1) == y).mean())
    return net, train_acc


# ---------- FGSM attack ----------

def input_loss_grad(net, X, y):
    """Returns gradient of cross-entropy loss w.r.t. input X."""
    W1, b1, W2, b2 = net
    h_pre = X @ W1 + b1
    h = np.maximum(0.0, h_pre)
    logits = h @ W2 + b2
    p = softmax(logits)
    n = X.shape[0]
    d_logits = p.copy()
    d_logits[np.arange(n), y] -= 1
    d_logits /= n
    dh = d_logits @ W2.T
    dh_pre = dh * (h_pre > 0)
    dx = dh_pre @ W1.T
    return dx


def fgsm_attack(net, X, y, eps):
    dx = input_loss_grad(net, X, y)
    return X + eps * np.sign(dx)


def measure_robustness(net, X_test, y_test, eps_values):
    out = {}
    for eps in eps_values:
        if eps == 0.0:
            X_adv = X_test
        else:
            X_adv = fgsm_attack(net, X_test, y_test, eps)
        logits, _ = forward(net, X_adv)
        out[float(eps)] = float((logits.argmax(axis=1) == y_test).mean())
    return out


# ---------- multiplicity ----------

def measure_multiplicity(net, n_classes, dream_seed=0, eps=0.4, min_size=15,
                          n_starts=1000):
    counts = []
    for k in range(n_classes):
        endpoints = dream(net, target_class=k, n_starts=n_starts, steps=600,
                          lr=0.05, lam=0.02, seed=dream_seed * 100 + k)
        clusters = cluster(endpoints, eps=eps, min_size=min_size)
        counts.append(len(clusters))
    return counts


# ---------- experiment ----------

def make_harder_dataset(n_per_class=400, seed=0):
    """Closer, more overlapping blobs so adversarial attacks actually flip
    classifications at small eps. Centers ~1.0 apart instead of ~3.0."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [ 0.9,  0.9],
        [-0.9,  0.9],
        [ 0.0, -0.9],
    ])
    X = []
    y = []
    for k, c in enumerate(centers):
        pts = c + rng.standard_normal((n_per_class, 2)) * 0.45
        X.append(pts)
        y.extend([k] * n_per_class)
    return np.vstack(X), np.array(y)


def run():
    # Build a harder, more overlapping 3-class task so robustness varies
    X, y = make_harder_dataset(n_per_class=400, seed=0)
    n_classes = int(y.max()) + 1
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(X))
    n_test = 400
    X_train_full, X_test = X[perm[n_test:]], X[perm[:n_test]]
    y_train_full, y_test = y[perm[n_test:]], y[perm[:n_test]]

    eps_attack = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]

    # Many cells with different (wd, train_size, seed) combinations to get
    # a wide range of (multiplicity, robustness) pairs.
    weight_decays = [0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    train_sizes = [200, 800]  # subset sizes to vary regularization indirectly
    train_seeds = list(range(5))

    rows = []
    t0 = time.time()
    for wd in weight_decays:
        for tsize in train_sizes:
            for s in train_seeds:
                # Subsample training set
                rng_sub = np.random.default_rng(s * 17 + 1)
                idx = rng_sub.choice(len(X_train_full), size=min(tsize, len(X_train_full)), replace=False)
                X_train = X_train_full[idx]
                y_train = y_train_full[idx]

                net, train_acc = train_with_wd(
                    X_train, y_train, hidden=16, weight_decay=wd, seed=s, epochs=2500
                )
                test_acc = float(
                    (forward(net, X_test)[0].argmax(axis=1) == y_test).mean()
                )
                mult = measure_multiplicity(net, n_classes, dream_seed=s)
                mean_mult = float(np.mean(mult))
                rob = measure_robustness(net, X_test, y_test, eps_attack)
                row = {
                    "wd": wd, "tsize": tsize, "seed": s,
                    "train_acc": train_acc, "test_acc": test_acc,
                    "multiplicity_per_class": mult,
                    "mean_multiplicity": mean_mult,
                    "robustness": rob,
                }
                rows.append(row)
                print(
                    f"  wd={wd:7.4f} tsize={tsize:>4} seed={s}  "
                    f"test_acc={test_acc:.3f}  "
                    f"mult={mean_mult:5.2f}  rob@0.2={rob[0.2]:.3f}"
                )
    print(f"\nTotal time: {time.time()-t0:.1f}s ({len(rows)} models)")
    return rows


def correlate(rows, eps_for_corr=0.2):
    print("\n========== ANALYSIS ==========")
    print("\nPer-cell summary (mean over seeds and tsizes):")
    print(f"{'wd':>10} {'mult':>8} {'test_acc':>10} {'rob@0.2':>10}")
    by_wd = {}
    for r in rows:
        by_wd.setdefault(r["wd"], []).append(r)
    for wd in sorted(by_wd):
        rs = by_wd[wd]
        m_mult = np.mean([r["mean_multiplicity"] for r in rs])
        m_acc = np.mean([r["test_acc"] for r in rs])
        m_rob = np.mean([r["robustness"][eps_for_corr] for r in rs])
        print(f"{wd:>10.4f} {m_mult:>8.2f} {m_acc:>10.3f} {m_rob:>10.3f}")

    # Correlations across all individual data points (not just averages)
    mults = np.array([r["mean_multiplicity"] for r in rows])
    test_accs = np.array([r["test_acc"] for r in rows])
    print("\nCorrelations across all individual trained models:")
    for eps_attack_val in [0.1, 0.2, 0.3, 0.5]:
        robs = np.array([r["robustness"][eps_attack_val] for r in rows])
        if np.std(mults) > 0 and np.std(robs) > 0:
            pearson = float(np.corrcoef(mults, robs)[0, 1])
            ranks_m = np.argsort(np.argsort(mults))
            ranks_r = np.argsort(np.argsort(robs))
            spearman = float(np.corrcoef(ranks_m, ranks_r)[0, 1])
            print(
                f"  multiplicity vs robustness@eps={eps_attack_val}:  "
                f"Pearson={pearson:+.3f}  Spearman={spearman:+.3f}"
            )

    # Also test multiplicity vs clean test accuracy
    if np.std(mults) > 0 and np.std(test_accs) > 0:
        pearson = float(np.corrcoef(mults, test_accs)[0, 1])
        ranks_m = np.argsort(np.argsort(mults))
        ranks_a = np.argsort(np.argsort(test_accs))
        spearman = float(np.corrcoef(ranks_m, ranks_a)[0, 1])
        print(
            f"  multiplicity vs CLEAN test acc:        "
            f"Pearson={pearson:+.3f}  Spearman={spearman:+.3f}"
        )

    # CRITICAL: drop the wd=0 cells and re-correlate. If the correlation
    # only exists when wd=0 is included, the finding is just "no
    # regularization is bad" - not a real multiplicity-robustness link.
    print("\n--- Correlations EXCLUDING wd=0 (regularized models only) ---")
    sub_rows = [r for r in rows if r["wd"] > 0]
    sub_mults = np.array([r["mean_multiplicity"] for r in sub_rows])
    sub_test_accs = np.array([r["test_acc"] for r in sub_rows])
    for eps_attack_val in [0.1, 0.2, 0.3]:
        sub_robs = np.array([r["robustness"][eps_attack_val] for r in sub_rows])
        if np.std(sub_mults) > 0 and np.std(sub_robs) > 0:
            pearson = float(np.corrcoef(sub_mults, sub_robs)[0, 1])
            ranks_m = np.argsort(np.argsort(sub_mults))
            ranks_r = np.argsort(np.argsort(sub_robs))
            spearman = float(np.corrcoef(ranks_m, ranks_r)[0, 1])
            print(
                f"  mult vs rob@eps={eps_attack_val}:  "
                f"Pearson={pearson:+.3f}  Spearman={spearman:+.3f}  "
                f"(n={len(sub_rows)})"
            )
    if np.std(sub_mults) > 0 and np.std(sub_test_accs) > 0:
        pearson = float(np.corrcoef(sub_mults, sub_test_accs)[0, 1])
        ranks_m = np.argsort(np.argsort(sub_mults))
        ranks_a = np.argsort(np.argsort(sub_test_accs))
        spearman = float(np.corrcoef(ranks_m, ranks_a)[0, 1])
        print(
            f"  mult vs CLEAN test acc:  "
            f"Pearson={pearson:+.3f}  Spearman={spearman:+.3f}"
        )

    return {
        "mults": mults.tolist(),
        "test_accs": test_accs.tolist(),
        "robs_at_eps_02": [r["robustness"][0.2] for r in rows],
    }


if __name__ == "__main__":
    rows = run()
    corr_data = correlate(rows)
    out = {
        "rows": rows,
        "correlations_inputs": corr_data,
    }
    with open("results_robustness_correlation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results_robustness_correlation.json")
