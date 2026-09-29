"""
Runs the sparse-grid detector on both benchmark scenarios and plots the results.

    python demo.py
"""

from rffmmd import detect_sparse, make_scenario1, make_scenario2
from rffmmd.simulate import _D


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    sys.setrecursionlimit(20000)
 
    rr = 100        # number of random features; small relative to n
    alpha = 0.1   # significance level for the threshold; 0 < alpha < 1
    lam = None     # regularization added to the covariance before whitening.
                   # None uses 1/log(n/W), which satisfies
                   # lam^-1 >> sqrt(2 log(n/W)).
 
for scen, gen in [("Scenario 1", make_scenario1),
                  ("Scenario 2", make_scenario2)]:
    X, true_cps, seg_names = gen(seed=800)

    sparse = detect_sparse(X, rr=rr, seed=42, lam=lam, alpha=alpha)
    sparse_cps = sorted(sparse["cps"])

    print(f"{scen}: true={true_cps}")
    print(f"   shape={X.shape}  W={sparse['W']}  R={sparse['R']}  alpha={alpha}")
    print(f"   lam={sparse['lam']}  thresh={sparse['thresh']}")
    print(f"   segments: {' -> '.join(seg_names)}")
    print(f"   sparse cps ({len(sparse_cps)}): {sparse_cps}")

    plt.figure(figsize=(13, 4.5))
    for jj in range(X.shape[1]):
        plt.plot(X[:, jj], color='gray', alpha=0.7)

    # shaded detection intervals (behind the CP lines)
    for i, iv in enumerate(sparse["intervals"]):
        plt.axvspan(iv["start"], iv["end"], color='r', alpha=0.10,
                    label='Sparse interval' if i == 0 else None)


    for i, c in enumerate(sparse_cps):
        plt.axvline(x=c, color='r', linestyle='-', lw=1,
                    label='Sparse CP' if i == 0 else None)

    edges = [0] + true_cps + [len(X)]
    ymax = X.max()
    for s, e, name in zip(edges[:-1], edges[1:], seg_names):
        plt.text(int((s + e) / 2), ymax, name, rotation=90, ha='center',
                 va='top', fontsize=12, color='black', alpha=0.9)


    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    plt.show()