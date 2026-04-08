"""
Robustness checks for the 'concept multiplicity is invariant to network size'
hint from experiment4.

If any of these checks fail, the finding is dead and we will not promote it
to the findings folder.

R1. Robustness to clustering eps.
    Vary eps in {0.2, 0.3, 0.4, 0.5, 0.6}. Does the count stay roughly stable
    (or at least: does the *invariance to network size* hold at every eps)?

R2. Wider range of network sizes.
    Vary hidden in {2, 4, 8, 16, 32, 64, 128}.  Does the count stay flat?

R3. Untrained baseline (the critical test).
    Run the dream procedure on UNtrained random networks of the same sizes.
    If untrained networks give similar attractor counts, the 'multiplicity'
    is just a property of random ReLU geometry and has nothing to do with
    training. The finding would be dead.
    If trained networks give SYSTEMATICALLY different counts than untrained,
    then training really is doing something measurable.

R4. Replication on a second task.
    Run the same protocol on a 4-class dataset with different geometry.
    If the counts on this task are reproducible across seeds, the
    measurement is stable. If 'invariant to hidden size' replicates,
    the finding is supported.
"""

import numpy as np
import json
import time

from experiment import init_mlp
from experiment4 import (
    make_dataset, train_classifier, dream, cluster,
)


# ---------- second task: 4-class cross ----------

def make_dataset_4class(n_per_class=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([
        [ 2.5,  0.0],
        [-2.5,  0.0],
        [ 0.0,  2.5],
        [ 0.0, -2.5],
    ])
    X = []
    y = []
    for k, c in enumerate(centers):
        pts = c + rng.standard_normal((n_per_class, 2)) * 0.5
        X.append(pts)
        y.extend([k] * n_per_class)
    return np.vstack(X), np.array(y)


def make_untrained(hidden, n_classes, seed):
    rng = np.random.default_rng(seed * 9 + hidden)
    return init_mlp(2, hidden, n_classes, rng)


# ---------- per-network attractor counts ----------

def count_attractors_for_net(net, n_classes, eps_values, n_starts=1000, dream_seed=0):
    """For a network, dream from random starts for each class, then count
    attractors at multiple eps values. Returns nested dict
    {class: {eps: count}}."""
    out = {}
    for k in range(n_classes):
        endpoints = dream(
            net, target_class=k, n_starts=n_starts,
            steps=600, lr=0.05, lam=0.02, seed=dream_seed * 100 + k,
        )
        out[k] = {}
        for eps in eps_values:
            clusters = cluster(endpoints, eps=eps, min_size=15)
            out[k][eps] = len(clusters)
    return out


# ---------- run ----------

def run():
    eps_values = [0.2, 0.3, 0.4, 0.5, 0.6]
    hidden_sizes = [2, 4, 8, 16, 32, 64, 128]
    n_seeds = 3

    tasks = {
        "task1_3class": (make_dataset(n_per_class=200, seed=0), 3),
        "task2_4class": (make_dataset_4class(n_per_class=200, seed=0), 4),
    }

    results = {}
    t0 = time.time()
    for task_name, ((X, y), n_classes) in tasks.items():
        print(f"\n========== {task_name} ==========")
        results[task_name] = {}
        for h in hidden_sizes:
            for variant in ("trained", "untrained"):
                key = f"h{h:03d}_{variant}"
                rows = []
                for seed in range(n_seeds):
                    if variant == "trained":
                        net, acc = train_classifier(
                            X, y, hidden=h, seed=seed, epochs=2500
                        )
                    else:
                        net = make_untrained(h, n_classes, seed=seed)
                        acc = float("nan")

                    counts = count_attractors_for_net(
                        net, n_classes, eps_values, dream_seed=seed,
                    )
                    # mean across classes per eps
                    mean_per_eps = {
                        str(eps): float(
                            np.mean([counts[k][eps] for k in range(n_classes)])
                        )
                        for eps in eps_values
                    }
                    rows.append({
                        "seed": seed,
                        "acc": acc,
                        "mean_per_eps": mean_per_eps,
                    })
                results[task_name][key] = rows
                # condensed log line for the canonical eps=0.4
                m04 = [r["mean_per_eps"]["0.4"] for r in rows]
                accs = [r["acc"] for r in rows if not np.isnan(r["acc"])]
                acc_str = f"{np.mean(accs):.2f}" if accs else " --- "
                print(
                    f"  h={h:>3}  {variant:>9}  acc={acc_str}  "
                    f"mean@eps=0.4: {np.mean(m04):5.2f}  "
                    f"(seeds: {[round(x,1) for x in m04]})"
                )
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    return results


def summarize(results):
    print("\n\n========== ROBUSTNESS SUMMARY ==========\n")
    eps_values = [0.2, 0.3, 0.4, 0.5, 0.6]
    hidden_sizes = [2, 4, 8, 16, 32, 64, 128]

    for task_name, by_key in results.items():
        print(f"=== {task_name} ===")
        print(f"{'hidden':>8}  {'trained_eps0.4':>16}  {'untrained_eps0.4':>18}  {'ratio (trained/untrained)':>28}")
        for h in hidden_sizes:
            tr = by_key[f"h{h:03d}_trained"]
            un = by_key[f"h{h:03d}_untrained"]
            tr_mean = np.mean([r["mean_per_eps"]["0.4"] for r in tr])
            un_mean = np.mean([r["mean_per_eps"]["0.4"] for r in un])
            ratio = tr_mean / un_mean if un_mean > 1e-6 else float("nan")
            print(f"{h:>8}  {tr_mean:>16.2f}  {un_mean:>18.2f}  {ratio:>28.2f}")

        # eps sensitivity at hidden=8 (typical case)
        print(f"\n  eps sensitivity at hidden=8 (trained):")
        rows = by_key["h008_trained"]
        for eps in eps_values:
            vals = [r["mean_per_eps"][str(eps)] for r in rows]
            print(f"    eps={eps}  mean={np.mean(vals):.2f}  values={vals}")
        print()


if __name__ == "__main__":
    results = run()
    summarize(results)
    with open("results_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results_robustness.json")
