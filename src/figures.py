"""
A3 figures: rank-frequency (Zipf) plots.

For each of the 5 languages, plots the rank-frequency curves of all 7
tokenizations (ws_raw, ws_lower, spacy, xlmr, bpe2000, bpe8000, bpe32000)
on a log-log scale, one panel per language, side by side in one figure.
A straight line on a log-log plot means the data follows Zipf's law.

Output: figures/overlay_all.png
Run from the project root: python src/figures.py
"""

from collections import Counter
import matplotlib.pyplot as plt
from config import OPTIONS, FIG_DIR, LANGUAGES
from cache_io import load_tokens


def rank_frequency(verses):
    """
    Counts every token and returns the frequencies sorted from most to least frequent.

    Input: a list of verses, each verse is a list of tokens.
    Output: a list of counts, e.g. [5000, 4000, 2800, ...].
    The first number is the count of the most frequent token (rank 1),
    the second is the count of the second most frequent token (rank 2), etc.
    """
    # flatten the verses into one list of tokens
    tokens = [t for verse in verses for t in verse]
    # count how many times each token appears
    counts = Counter(tokens)
    # keep only the counts (not the tokens) and sort them from high to low
    return sorted(counts.values(), reverse=True)


if __name__ == "__main__":
    # one row of 5 panels (one per language), all sharing the same y axis
    # so the languages can be compared directly
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    # zip pairs each panel with one language: axes[0] with "en", axes[1] with "fr", ...
    for ax, language in zip(axes, LANGUAGES):
        # draw one line per tokenization in this language's panel
        for scheme in OPTIONS:
            frequencies = rank_frequency(load_tokens(language, scheme))
            # ranks are 1, 2, 3, ... up to the number of distinct tokens
            # (+1 because range does not include the end number)
            ranks = range(1, len(frequencies) + 1)
            # log-log scale: Zipf's law shows up as a roughly straight line
            # alpha=0.7 makes lines slightly transparent so overlapping lines stay visible
            ax.loglog(ranks, frequencies, label=scheme, alpha=0.7)

        ax.set_title(language)
        ax.set_xlabel("Rank")
        ax.grid(True)

    # y label and legend only on the first panel, so they are not repeated 5 times
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # dpi=150 for a sharper image, bbox_inches="tight" trims extra white space
    plt.savefig(FIG_DIR / "overlay_all.png", dpi=150, bbox_inches="tight")
    print("saved")