"""
Evidence for choosing rmax: bootstrap alpha for English with three fit ranges.

For each rmax (None = all words, 1000, 500) we compute alpha on the full corpus
(dot) and its bootstrap 95% CI (bar) for all 7 tokenizations. If the fit range
is good, every dot should fall inside its own bar.

Output: figures/rmax_comparison_en.png
Run from the project root: python src/rmax_comparison.py
"""

import random
from collections import Counter
import matplotlib.pyplot as plt
from config import OPTIONS, SEED, FIG_DIR
from cache_io import load_tokens
from analysis import ls_alpha
from bootstrap import bootstrap_alpha

LANGUAGE = "en"
N_BOOT = 1000                   
RMAX_VALUES = [None, 1000, 500]


def make_fit(rmax):
    """
    Builds a fit function that computes alpha with a given rmax.

    Args:
        rmax: number of top ranks used in the fit (None = all words).

    Returns:
        A function that takes verses and returns alpha, like ls_fit,
        so it can be passed to bootstrap_alpha.
    """
    def fit(verses):
        tokens = [t for verse in verses for t in verse]
        counts = sorted(Counter(tokens).values(), reverse=True)
        return ls_alpha(counts, rmax=rmax)
    return fit


if __name__ == "__main__":
    # one panel per rmax value, side by side
    fig, axes = plt.subplots(1, len(RMAX_VALUES), figsize=(15, 4.5))

    for ax, rmax in zip(axes, RMAX_VALUES):
        fit = make_fit(rmax)
        outside = 0   # how many dots fall outside their own CI

        for x, scheme in enumerate(OPTIONS):
            random.seed(SEED)
            verses = load_tokens(LANGUAGE, scheme)
            alpha = fit(verses)
            low, high = bootstrap_alpha(verses, fit, n_boot=N_BOOT)
            inside = low <= alpha <= high
            if not inside:
                outside += 1
            print(rmax, scheme, round(alpha, 3), round(low, 3), round(high, 3),
                  "ok" if inside else "OUTSIDE")

            # CI as a light yellow bar, alpha as a purple dot
            ax.vlines(x, low, high, color="#eda100", lw=8, alpha=0.5)
            ax.plot(x, alpha, "o", ms=7, color="#6250d6")

        name = "all words" if rmax is None else f"top {rmax}"
        ax.set_title(f"rmax = {name}: {outside}/{len(OPTIONS)} outside CI")
        ax.set_xticks(range(len(OPTIONS)))
        ax.set_xticklabels(OPTIONS, rotation=45)
        ax.grid(axis="y", color="#e5e5e2")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Zipf alpha (least squares)")
    fig.suptitle(f"{LANGUAGE}: alpha (purple dot) and bootstrap 95% CI (yellow bar), n_boot={N_BOOT}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"rmax_comparison_{LANGUAGE}.png", dpi=150)
    print("saved")