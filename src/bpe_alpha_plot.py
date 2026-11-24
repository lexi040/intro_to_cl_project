"""
how alpha changes with the BPE vocabulary size.

Reads the bootstrap results (cache/bootstrap_alpha.csv, made by bootstrap.py)
and plots alpha against the BPE vocabulary size (2000, 8000, 32000), one line
per language. Error bars are the bootstrap 95% confidence intervals.

"""

import pandas as pd
import matplotlib.pyplot as plt
from config import LANGUAGES, BPE_VOCAB_SIZES, CACHE_DIR, FIG_DIR

# one color and one marker per language, so lines can be told apart
# even without color (print, colorblind readers)
COLORS = {
    "en": "#228B22",   # forest green
    "fr": "#8B0000",   # dark red
    "de": "#E3BE00",   # yellow
    "fi": "#461fb3",   # dark purple
    "tr": "#000080",   # navy
}
MARKERS = {"en": "o", "fr": "s", "de": "^", "fi": "D", "tr": "v"}


if __name__ == "__main__":
    table = pd.read_csv(CACHE_DIR / "bootstrap_alpha.csv")

    fig, ax = plt.subplots(figsize=(7, 5))

    for language in LANGUAGES:
        # alpha and CI for bpe2000, bpe8000, bpe32000 (in this order)
        alphas, lower, upper = [], [], []
        for size in BPE_VOCAB_SIZES:
            row = table[(table["language"] == language) & (table["scheme"] == f"bpe{size}")]
            alpha = row["alpha"].iloc[0]
            alphas.append(alpha)
            # errorbar wants the distance from the dot, not the CI ends
            lower.append(alpha - row["low"].iloc[0])
            upper.append(row["high"].iloc[0] - alpha)

        ax.errorbar(BPE_VOCAB_SIZES, alphas, yerr=[lower, upper],
                    color=COLORS[language], marker=MARKERS[language],
                    markersize=8, linewidth=2, capsize=4, label=language)

    # the sizes grow x4 each step, so a log x axis spaces them evenly
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(BPE_VOCAB_SIZES)
    ax.set_xticklabels([str(s) for s in BPE_VOCAB_SIZES])
    ax.set_xlabel("BPE vocabulary size")
    ax.set_ylabel("Zipf alpha (least squares, top 500 ranks)")
    ax.set_title("Alpha vs BPE vocabulary size (bootstrap 95% CI)")
    ax.grid(color="#e5e5e2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="language", frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "bpe_alpha.png", dpi=150)
    print("saved")