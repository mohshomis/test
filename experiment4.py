"""
Counting concept attractors experiment.

Question:
  When a trained classifier 'dreams' of a class - i.e. you do gradient
  ascent on log P(class=k | x) starting from a random input x - how many
  distinct attractors does it converge to?

  Does the network have ONE internal template per class, or MANY?

Setup:
  - Synthetic 2D classification: 3 Gaussian blobs.
  - Train a small ReLU MLP (2 -> hidden -> 3 logits) with cross-entropy.
  - For each class k:
      * Sample 1000 random starting points in a bounded box.
      * Run gradient ascent on  log P(class=k | x) - lam * ||x||^2
      * Record final positions.
      * Greedy-cluster the endpoints.
      * Count clusters with >= min_size members. That count is the
        'concept multiplicity' for class k.
  - Repeat for several hidden sizes.

Open question being attacked:
  Nguyen et al. 2016 noted automatically counting facets per concept is
  an open problem. We attack it with the simplest possible method.
"""

import numpy as np
import json
import time

from experiment import init_mlp, forward, backward, Adam


# ---------- dataset ----------

def make_dataset(n_per_class=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([
        [ 2.0,  2.0],
        [-2.0,  2.0],
        [ 0.0, -2.0],
    ])
    X = []
    y = []
    for k, c in enumerate(centers):
        pts = c + rng.standard_normal((n_per_class, 2)) * 0.6
        X.append(pts)
        y.extend([k] * n_per_class)
    return np.vstack(X), np.array(y)


# ---------- classifier ----------

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def train_classifier(X, y, hidden, epochs=3000, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    net = init_mlp(2, hidden, 3, rng)
    opt = Adam(net, lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        logits, cache = forward(net, X)
        p = softmax(logits)
        # cross-entropy gradient w.r.t. logits
        dout = p.copy()
        dout[np.arange(n), y] -= 1
        dout /= n
        grads = backward(net, cache, dout)
        opt.step(grads)
    # accuracy
    logits, _ = forward(net, X)
    acc = float((logits.argmax(axis=1) == y).mean())
    return net, acc


# ---------- dream (activation maximization on input) ----------

def dream(net, target_class, n_starts=1000, steps=500, lr=0.05, lam=0.02,
          bound=5.0, seed=0):
    """Gradient ascent on log P(target | x) - lam * ||x||^2 from random starts."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-bound, bound, (n_starts, 2))
    W1, b1, W2, b2 = net

    for _ in range(steps):
        h_pre = X @ W1 + b1
        h = np.maximum(0.0, h_pre)
        logits = h @ W2 + b2
        p = softmax(logits)
        # d log p[target] / d logits = e_target - p
        d_logits = -p.copy()
        d_logits[:, target_class] += 1.0
        # backprop to input
        dh = d_logits @ W2.T
        dh_pre = dh * (h_pre > 0)
        dx = dh_pre @ W1.T
        # regularization gradient: d(-lam*||x||^2)/dx = -2*lam*x
        dx -= 2.0 * lam * X
        # ascent
        X = X + lr * dx
        X = np.clip(X, -bound, bound)
    return X


# ---------- greedy clustering ----------

def cluster(points, eps, min_size):
    """Online greedy clustering by Euclidean distance to running centroid."""
    centroids = []
    members = []
    for i, p in enumerate(points):
        if not centroids:
            centroids.append(p.copy())
            members.append([i])
            continue
        ds = [np.linalg.norm(p - c) for c in centroids]
        j = int(np.argmin(ds))
        if ds[j] < eps:
            members[j].append(i)
            centroids[j] = points[members[j]].mean(axis=0)
        else:
            centroids.append(p.copy())
            members.append([i])
    keep = [(c, m) for c, m in zip(centroids, members) if len(m) >= min_size]
    return keep


# ---------- experiment ----------

def run():
    X, y = make_dataset(n_per_class=200, seed=0)

    hidden_sizes = [4, 8, 16, 32]
    n_train_seeds = 3

    all_results = []
    t0 = time.time()
    for h in hidden_sizes:
        for tseed in range(n_train_seeds):
            net, acc = train_classifier(X, y, hidden=h, seed=tseed)
            print(f"\nhidden={h}  train_seed={tseed}  train_acc={acc:.3f}")
            row = {"hidden": h, "train_seed": tseed, "acc": acc, "per_class": {}}
            for k in range(3):
                endpoints = dream(net, target_class=k, n_starts=1000,
                                  steps=600, lr=0.05, lam=0.02,
                                  seed=tseed * 100 + k)
                clusters = cluster(endpoints, eps=0.4, min_size=15)
                row["per_class"][k] = {
                    "n_attractors": len(clusters),
                    "attractors": [
                        {"centroid": c.tolist(), "size": len(m)}
                        for c, m in clusters
                    ],
                }
                summary = ", ".join(
                    f"({c[0]:+.2f},{c[1]:+.2f})x{len(m)}"
                    for c, m in clusters
                )
                print(f"  class {k}: {len(clusters)} attractors  -> {summary}")
            all_results.append(row)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    return all_results


def summarize(results):
    print("\n========== SUMMARY ==========")
    print("Concept multiplicity (attractor count) per class.\n")
    print(f"{'hidden':>8}{'seed':>6}{'acc':>8}{'C0':>6}{'C1':>6}{'C2':>6}")
    for r in results:
        c0 = r["per_class"][0]["n_attractors"]
        c1 = r["per_class"][1]["n_attractors"]
        c2 = r["per_class"][2]["n_attractors"]
        print(f"{r['hidden']:>8}{r['train_seed']:>6}{r['acc']:>8.3f}"
              f"{c0:>6}{c1:>6}{c2:>6}")

    # average per hidden size
    print("\nMean attractor count per class, averaged across train seeds:")
    print(f"{'hidden':>8}{'mean_C0':>10}{'mean_C1':>10}{'mean_C2':>10}{'mean_total':>12}")
    by_h = {}
    for r in results:
        by_h.setdefault(r["hidden"], []).append(r)
    for h, rs in by_h.items():
        m0 = np.mean([r["per_class"][0]["n_attractors"] for r in rs])
        m1 = np.mean([r["per_class"][1]["n_attractors"] for r in rs])
        m2 = np.mean([r["per_class"][2]["n_attractors"] for r in rs])
        print(f"{h:>8}{m0:>10.2f}{m1:>10.2f}{m2:>10.2f}{m0+m1+m2:>12.2f}")


if __name__ == "__main__":
    results = run()
    summarize(results)
    with open("results_attractors.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_attractors.json")
