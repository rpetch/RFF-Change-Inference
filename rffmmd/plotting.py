"""Plotting helper: overlay true and detected change points on a series."""

__all__ = ["plot_detection"]


def plot_detection(XX, true_cps, seg_names, sparse, mosum, title="",
                   alpha=1, HH=128, show_intervals=True, ax=None):
    """
    Draw the series with true and detected change points overlaid.

    Green dashed lines are the truth, red solid lines the sparse-grid
    detections, blue dotted lines the MOSUM detections. With show_intervals,
    the coarse detection windows are shaded behind the lines.

    Args:
        XX (array, shape (n, d)) : the series that was analysed.
        true_cps (list[int]) : ground-truth change point indices.
        seg_names (list[str]) : distribution name per segment, for annotation.
        sparse (dict) : the return value of detect_sparse().
        mosum (dict) : the return value of detect_mosum().
        title (str) : scenario label for the plot title.
        alpha (float) : sensitivity used, shown in the title.
        HH (int) : MOSUM window width used, shown in the title.
        show_intervals (bool) : shade the coarse detection windows.
        ax (matplotlib Axes or None) : axes to draw on; a new figure if None.
    Returns:
        matplotlib Axes : the axes drawn on.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.5))

    sparse_cps = sorted(sparse["cps"])
    mosum_cps = sorted(mosum["cps"])

    ax.plot(XX[:, 0], label="Feature 1", alpha=0.7)
    if XX.shape[1] > 1:
        ax.plot(XX[:, 1], label="Feature 2", alpha=0.7)

    if show_intervals:                       # shaded windows, behind the lines
        for i, iv in enumerate(sparse["intervals"]):
            ax.axvspan(iv["start"], iv["end"], color="r", alpha=0.10,
                       label="Sparse interval" if i == 0 else None)
        for i, iv in enumerate(mosum["intervals"]):
            ax.axvspan(iv["start"], iv["end"], color="b", alpha=0.10,
                       label="MOSUM interval" if i == 0 else None)

    for i, t in enumerate(true_cps):
        ax.axvline(x=t, color="g", linestyle="--", lw=1,
                   label="True Change Point" if i == 0 else None)
    for i, c in enumerate(sparse_cps):
        ax.axvline(x=c, color="r", linestyle="-", lw=1,
                   label="Sparse CP" if i == 0 else None)
    for i, c in enumerate(mosum_cps):
        ax.axvline(x=c, color="b", linestyle=":", lw=1.2,
                   label="MOSUM CP" if i == 0 else None)

    edges = [0] + list(true_cps) + [len(XX)]      # segment name annotations
    ymax = XX[:, 0].max()
    for s, e, name in zip(edges[:-1], edges[1:], seg_names):
        ax.text((s + e) / 2, ymax, name, rotation=90, ha="center",
                va="top", fontsize=7, color="dimgray", alpha=0.9)

    ax.set_title(f"{title}  (alpha={alpha}, MOSUM H={HH})\n" + "  ".join(seg_names),
                 fontsize=9)
    ax.set_xlabel("Time step (n)")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=8)
    return ax
