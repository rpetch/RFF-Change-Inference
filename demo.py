"""
Demo: run both detectors on both benchmark scenarios.

Usage:
    python demo.py                       # defaults (alpha=1, H=128)
    python demo.py --alpha 2             # less sensitive
    python demo.py --no-plot             # print results only
    python demo.py --scenario 2 --save out/
"""

import argparse
import sys
from pathlib import Path

from rffmmd import (detect_sparse, detect_mosum, make_scenario1, make_scenario2,
                    plot_detection)

SCENARIOS = {1: ("Scenario 1", make_scenario1),
             2: ("Scenario 2", make_scenario2)}


def parse_args(argv=None):
    """Build and parse the command-line arguments."""
    p = argparse.ArgumentParser(description="RFF-MMD change point detection demo.")
    p.add_argument("--alpha", type=float, default=1,
                   help="sensitivity knob; higher = fewer detections (default 1)")
    p.add_argument("--HH", type=int, default=128,
                   help="MOSUM window width (default 128)")
    p.add_argument("--rr", type=int, default=500,
                   help="number of random Fourier features (default 500)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the RFF map (default 42)")
    p.add_argument("--data-seed", type=int, default=5,
                   help="RNG seed for the generated series (default 5)")
    p.add_argument("--scenario", type=int, choices=[1, 2], default=None,
                   help="run only this scenario (default: both)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip plotting, print results only")
    p.add_argument("--no-intervals", action="store_true",
                   help="do not shade the coarse detection windows")
    p.add_argument("--save", type=str, default=None,
                   help="directory to write PNG figures to instead of showing them")
    return p.parse_args(argv)


def run_one(name, gen, args):
    """Generate one scenario, run both detectors, print a summary."""
    XX, true_cps, seg_names = gen(seed=args.data_seed)

    sparse = detect_sparse(XX, alpha=args.alpha, rr=args.rr, seed=args.seed)
    mosum = detect_mosum(XX, alpha=args.alpha, rr=args.rr, seed=args.seed,
                         HH=args.HH)

    print(f"{name}: true={true_cps}")
    print(f"   segments: {' -> '.join(seg_names)}")
    print(f"   sparse cps ({len(sparse['cps'])}): {sorted(sparse['cps'])}")
    print(f"   mosum  cps ({len(mosum['cps'])}): {sorted(mosum['cps'])}")
    return XX, true_cps, seg_names, sparse, mosum


def main(argv=None):
    """Run the demo end to end."""
    args = parse_args(argv)
    sys.setrecursionlimit(20000)          # greedy_interval_search recurses deeply

    chosen = ([args.scenario] if args.scenario else sorted(SCENARIOS))
    results = []
    for key in chosen:
        name, gen = SCENARIOS[key]
        results.append((key, name, run_one(name, gen, args)))

    if args.no_plot:
        return 0

    import matplotlib
    if args.save:
        matplotlib.use("Agg")             # headless backend for file output
    import matplotlib.pyplot as plt

    for key, name, (XX, true_cps, seg_names, sparse, mosum) in results:
        plot_detection(XX, true_cps, seg_names, sparse, mosum, title=name,
                       alpha=args.alpha, HH=args.HH,
                       show_intervals=not args.no_intervals)
        plt.tight_layout()
        if args.save:
            out = Path(args.save)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"scenario{key}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"   saved {path}")
        else:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
