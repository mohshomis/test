"""
Cross-network attractor agreement: do two trained networks dream the same dream?

Question:
  When two different networks are trained on the SAME task (same data, same
  loss, same training procedure), do they end up with dream attractors at
  the SAME positions in input space? Or does each network have its own
  private internal templates?

Setup:
  - Train many networks on the same 3-class blob task with:
      * Different random seeds (15 networks with hidden=16)
      * Different architectures (hidden in {8, 16, 32}, 5 seeds each)
  - For each network, extract dream attractors (centroids of clustered
    gradient-ascent endpoints).
  - For each PAIR of networks, compute the matching distance:
      D(A, B) = mean over each attractor in A of (distance to nearest
                                                  attractor of same class in B)
    averaged with the symmetric direction.
  - Compare to a random baseline: distance from each attractor to a random
    point in the data region.

Interpretation:
  - If D(A, B) << random baseline → networks find the SAME attractors.
    Templates are an objective property of the data.
  - If D(A, B) ~ random baseline → networks find different attractors.
    Templates are private to each network.
  - In between → partial agreement.
"""

import numpy as np
import json
import time

from experiment4 import make_dataset, train_classifier, dream, cluster
from experiment import forward


# ---------- attractor extraction ----------

def get_attractors(net, n_classes, dream_seed=0, eps=0.4, min_size=15,
                   n_starts=1000):
    """Returns dict {class_k: list of np.ndarray attractor centroids}."""
    out = {}
    for k in range(n_classes):
        endpoints = dream(net, target_class=k, n_starts=n_starts, steps=600,
                          lr=0.05, lam=0.02, seed=dream_seed * 100 + k)
        clusters = cluster(endpoints, eps=eps, min_size=min_size)
        out[k] = [np.array(c[0]) for c in clusters]
    return out


# ---------- matching distance ----------

def directional_match_distance(attrs_A, attrs_B):
    """For each attractor in A, distance to nearest same-class attractor in B."""
    distances = []
    for k, alist in attrs_A.items():
        blist = attrs_B.get(k, [])
        if not alist or not blist:
            continue
        for a in alist:
            ds = [float(np.linalg.norm(a - b)) for b in blist]
            distances.append(min(ds))
    return float(np.mean(distances)) if distances else float("nan")


def symmetric_match_distance(attrs_A, attrs_B):
    """Symmetric mean: average of A->B and B->A directional distances."""
    d_ab = directional_match_distance(attrs_A, attrs_B)
    d_ba = directional_match_distance(attrs_B, attrs_A)
    return (d_ab + d_ba) / 2.0


# ---------- random baseline ----------

def random_baseline_distance(attrs, data_range=(-4.0, 4.0), n_random_per=20,
                              seed=42):
    """For each attractor, distance to random points in input region."""
    rng = np.random.default_rng(seed)
    distances = []
    for k, alist in attrs.items():
        for a in alist:
            for _ in range(n_random_per):
                rand_pt = rng.uniform(data_range[0], data_range[1], size=2)
                distances.append(float(np.linalg.norm(a - rand_pt)))
    return float(np.mean(distances))


# ---------- run experiment ----------

def make_alt_dataset(n_per_class=200, seed=0):
    """3-class blobs at DIFFERENT centers - the cross-task control."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [-2.0, -2.0],
        [ 2.0, -2.0],
        [ 0.0,  1.5],
    ])
    X = []
    y = []
    for k, c in enumerate(centers):
        pts = c + rng.standard_normal((n_per_class, 2)) * 0.6
        X.append(pts)
        y.extend([k] * n_per_class)
    return np.vstack(X), np.array(y)


def within_network_attractor_spread(attrs):
    """Mean nearest-neighbor distance among attractors of the same class."""
    distances = []
    for k, alist in attrs.items():
        if len(alist) < 2:
            continue
        for i, a in enumerate(alist):
            others = [b for j, b in enumerate(alist) if j != i]
            ds = [float(np.linalg.norm(a - b)) for b in others]
            distances.append(min(ds))
    return float(np.mean(distances)) if distances else float("nan")


def run():
    X, y = make_dataset(n_per_class=200, seed=0)
    n_classes = int(y.max()) + 1

    # Train multiple networks
    print("Training networks...")
    t0 = time.time()
    networks = []
    for hidden in [8, 16, 32]:
        for seed in range(5):
            net, acc = train_classifier(X, y, hidden=hidden, seed=seed,
                                        epochs=2500)
            networks.append({"hidden": hidden, "seed": seed, "net": net, "acc": acc})
            print(f"  hidden={hidden:>3} seed={seed}  acc={acc:.3f}")

    print(f"\nTrained {len(networks)} networks in {time.time()-t0:.1f}s")

    # Extract attractors for each
    print("\nExtracting attractors...")
    t0 = time.time()
    for n in networks:
        attrs = get_attractors(n["net"], n_classes, dream_seed=n["seed"])
        n["attractors"] = attrs
        total = sum(len(v) for v in attrs.values())
        per_class = [len(attrs[k]) for k in range(n_classes)]
        print(f"  hidden={n['hidden']:>3} seed={n['seed']}: total={total} per_class={per_class}")
    print(f"Extracted attractors in {time.time()-t0:.1f}s")

    # Pairwise matching distances
    print("\nComputing pairwise matching distances...")
    pairs = []
    for i in range(len(networks)):
        for j in range(i + 1, len(networks)):
            d = symmetric_match_distance(
                networks[i]["attractors"], networks[j]["attractors"]
            )
            same_arch = networks[i]["hidden"] == networks[j]["hidden"]
            pairs.append({
                "i": i, "j": j,
                "h_i": networks[i]["hidden"], "h_j": networks[j]["hidden"],
                "seed_i": networks[i]["seed"], "seed_j": networks[j]["seed"],
                "same_arch": bool(same_arch),
                "distance": d,
            })

    # Random baseline (using the first network's attractors)
    rand_baseline = random_baseline_distance(networks[0]["attractors"])

    # Self-baseline: dream from same network with DIFFERENT dream seed and
    # see how stable that is. (Lower bound for "same network = same dreams")
    print("\nComputing self-baseline (same network, different dream seeds)...")
    self_distances = []
    for n in networks[:5]:
        attrs_a = n["attractors"]
        attrs_b = get_attractors(n["net"], n_classes, dream_seed=n["seed"] + 999)
        self_distances.append(symmetric_match_distance(attrs_a, attrs_b))
    self_baseline = float(np.mean(self_distances))

    # CROSS-TASK CONTROL: train networks on a DIFFERENT 3-class task and
    # compare their attractors to the original task's networks. If our
    # matching metric is meaningful, cross-task pairs should have a much
    # larger distance than same-task pairs.
    print("\nComputing cross-task baseline (different blob positions)...")
    X_alt, y_alt = make_alt_dataset(n_per_class=200, seed=0)
    alt_networks = []
    for seed in range(5):
        net, acc = train_classifier(X_alt, y_alt, hidden=16, seed=seed, epochs=2500)
        attrs = get_attractors(net, n_classes, dream_seed=seed)
        alt_networks.append({"hidden": 16, "seed": seed, "attractors": attrs})
    cross_task_distances = []
    for n_orig in networks:
        for n_alt in alt_networks:
            d = symmetric_match_distance(n_orig["attractors"], n_alt["attractors"])
            cross_task_distances.append(d)
    cross_task_baseline = float(np.mean(cross_task_distances))

    # Within-network attractor spread - the natural scale of "different attractor"
    within_network_spreads = [
        within_network_attractor_spread(n["attractors"]) for n in networks
    ]
    within_spread = float(np.mean(within_network_spreads))

    # Summarize
    same_arch_d = [p["distance"] for p in pairs if p["same_arch"]]
    cross_arch_d = [p["distance"] for p in pairs if not p["same_arch"]]

    print("\n========== RESULTS ==========")
    print(f"Self-baseline (same net, different dream seed):     {self_baseline:.3f}")
    print(f"Within-network attractor spread (natural scale):    {within_spread:.3f}")
    print(f"Same-architecture pair distance (mean):             {np.mean(same_arch_d):.3f}")
    print(f"Cross-architecture pair distance (mean):            {np.mean(cross_arch_d):.3f}")
    print(f"CROSS-TASK pair distance (different task):          {cross_task_baseline:.3f}")
    print(f"Random baseline (random point in input region):     {rand_baseline:.3f}")
    print()
    print(f"Self / random:           {self_baseline/rand_baseline:.3f}")
    print(f"Same-arch / random:      {np.mean(same_arch_d)/rand_baseline:.3f}")
    print(f"Cross-arch / random:     {np.mean(cross_arch_d)/rand_baseline:.3f}")
    print(f"Cross-TASK / random:     {cross_task_baseline/rand_baseline:.3f}")
    print()
    print(f"Same-task / cross-task:  {np.mean(same_arch_d)/cross_task_baseline:.3f}")
    print(f"  (this should be SMALL if same-task agreement is real)")

    # Interpretation
    print("\nInterpretation:")
    if np.mean(same_arch_d) < 0.3 * rand_baseline:
        print("  STRONG agreement: same-arch networks dream nearly identical attractors.")
    elif np.mean(same_arch_d) < 0.6 * rand_baseline:
        print("  MODERATE agreement: same-arch networks dream similar (not identical) attractors.")
    else:
        print("  WEAK agreement: same-arch networks dream essentially different attractors.")

    out = {
        "networks": [
            {"hidden": n["hidden"], "seed": n["seed"], "acc": n["acc"],
             "n_attractors": sum(len(v) for v in n["attractors"].values())}
            for n in networks
        ],
        "pairs": pairs,
        "self_baseline": self_baseline,
        "within_network_spread": within_spread,
        "same_arch_mean": float(np.mean(same_arch_d)),
        "cross_arch_mean": float(np.mean(cross_arch_d)),
        "cross_task_baseline": cross_task_baseline,
        "random_baseline": rand_baseline,
    }
    return out


if __name__ == "__main__":
    out = run()
    with open("results_cross_network.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results_cross_network.json")
