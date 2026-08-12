# rffmmd

Two **Random Fourier Feature MMD** change point detectors for multivariate time series.

Both detectors test whether the distribution of the data changes over a window, using the Maximum Mean Discrepancy between the two halves of that window. The MMD is computed in Random Fourier Feature space from a prefix sum of the feature matrix, so each window costs O(1) regardless of its width:

```
stat(l, w) = ||sum_B - sum_A|| / sqrt(w)
           = sqrt(w)/2 * ||mean_B - mean_A||
```

The two detectors differ only in *which* windows they test:

- **`detect_sparse`** — dyadic `2^k` windows searched by greedy recursion. Finds an unknown number of change points without being told how many to look for.
- **`detect_mosum`** — a single fixed-width window slid along the series (moving sum).

Both return coarse intervals; a shared post-processing step (`localise`) then pins the exact change-point index inside each interval.

## Installation

```bash
git clone https://github.com/<your-username>/rffmmd.git
cd rffmmd
pip install -e ".[dev]"
```

Requires Python 3.9+. The core package needs only NumPy; matplotlib is needed for plotting.

## Usage

```python
import numpy as np
from rffmmd import detect_sparse, detect_mosum

rng = np.random.default_rng(0)
X = np.concatenate([rng.normal(0, 1, 300),
                    rng.normal(3, 1, 300),
                    rng.normal(3, 4, 400)])[:, None]

sparse = detect_sparse(X, alpha=1, rr=500, seed=42)
mosum  = detect_mosum(X, alpha=1, rr=500, seed=42, HH=128)

print(sparse["cps"])       # detected change point indices
print(sparse["intervals"]) # coarse windows, each with its localised cp
```

Both return a dict with `intervals`, `cps`, `thresh`, `sigma`, and `gamma`.

Key arguments:

| Argument | Meaning |
| --- | --- |
| `alpha` | Sensitivity. Higher means a lower threshold and more detections. Not a Type-I error rate. |
| `rr` | Number of random features; the feature dimension is `2*rr`. More is more accurate but slower. |
| `gamma` | Gaussian kernel bandwidth. `None` uses the median heuristic (`1/med(X)`). |
| `seed` | RNG seed for the random frequencies, so runs are reproducible. |
| `HH` | MOSUM window width only. Too small misses wide or subtle changes; too large smears nearby ones. |

## Demo

`demo.py` runs both detectors on two synthetic benchmark scenarios and plots the results against the ground truth.

```bash
python demo.py                  # both scenarios, show plots
python demo.py --alpha 2        # less sensitive
python demo.py --no-plot        # print detections only
python demo.py --scenario 2 --save out/   # write PNGs instead of showing
```

The benchmark follows Arlot, Celisse & Harchaoui (2019), Section 6.1: n = 1000, 11 segments, 10 change points.

- **Scenario 1** — seven distributions, mean and variance both shift across boundaries. The easier case.
- **Scenario 2** — three distributions all with mean 0.5 and variance 0.25, so only the *shape* changes. The hard case, and the one that motivates a kernel method: a mean-based detector sees nothing here.

The demo defaults to a conservative sensitivity (`alpha=1`), which finds only the strongest few boundaries. Raise `--alpha` to detect more.

## Repository structure

```
rffmmd/
    __init__.py    the public API
    core.py        the algorithm, in pipeline order:
                     1. features   RFF map, median heuristic, feature prefix sum
                     2. grid       noise scale, threshold, window enumeration
                     3. statistic  local MMD on a window, change-point localisation
                     4. detectors  detect_sparse, detect_mosum
    simulate.py    synthetic benchmark scenarios
    plotting.py    plotting helper
demo.py            command-line demo
test.py            pytest suite
```

## Testing

```bash
pytest
```

## References

- Arlot, S., Celisse, A. and Harchaoui, Z. (2019). A kernel multiple change-point algorithm via model selection. *JMLR* 20(162).
- Rahimi, A. and Recht, B. (2007). Random features for large-scale kernel machines. *NeurIPS*.

## License

MIT — see [LICENSE](LICENSE).
