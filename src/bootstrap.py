"""
A3 bootstrap: confidence intervals for the Zipf exponent alpha.

Idea: we resample the verses with replacement (some verses appear several
times, some not at all) to build many slightly different versions of the
corpus. We compute alpha on each version. The spread of these alphas shows
how stable our alpha estimate is.

The fit function is passed as a parameter, so the same bootstrap works for
both alpha methods: least squares (ls_fit) and MLE (mle_fit).

Run from the project root:
    python src/bootstrap.py              -> real texts     -> cache/bootstrap_alpha.csv
    python src/bootstrap.py baselines    -> baseline texts -> cache/bootstrap_baselines.csv
"""

import random
import sys
import numpy as np
import csv
import powerlaw
from mle_alpha import mle_alpha
from config import SEED, LANGUAGES, OPTIONS, CACHE_DIR
from cache_io import load_tokens
from collections import Counter
from analysis import ls_alpha

# Number of top ranks used for the least squares fit.
# 500 keeps every language x tokenization out of the noisy rare-word tail
# (see figures/rmax_comparison_en.png and Clauset et al. 2009).
RMAX = 500

# Baseline texts: bigram (Öykü) and random typing (Ani), same tokenizations
BASELINES = ([f"bigram_{s}" for s in ["spacy", "xlmr", "bpe2000", "bpe8000", "bpe32000"]]
             + [f"random_{s}" for s in OPTIONS])


def resample(verses):
    """
    Draws verses at random, with replacement (one bootstrap sample).

    Args:
        verses: list of verses, each verse is a list of tokens.

    Returns:
        A new list of verses of the same size. Some verses can appear
        more than once, some can be missing.
    """
    random_verses = random.choices(verses, k=len(verses))
    return random_verses


def ls_fit(verses):
    """
    Computes alpha with least squares (Lexi's ls_alpha) from a list of verses.

    ls_alpha expects sorted token counts, not verses, so this function
    flattens the verses into one token list and sorts the counts first.

    Args:
        verses: list of verses, each verse is a list of tokens.

    Returns:
        alpha as a float.
    """
    # flatten the verses into one list of tokens
    tokens = []
    for verse in verses:
        for t in verse:
            tokens.append(t)
    # count and sort the tokens, then fit alpha on the log-log curve
    return ls_alpha(sorted(Counter(tokens).values(), reverse=True), rmax=RMAX)


def mle_fit(verses):
    """
    Computes alpha with MLE (same method as Ani's mle_alpha) from a list of verses.

    Ani's mle_alpha reads the cache file itself, but the bootstrap needs to
    pass resampled verses, so this function does the same fit on any verses.
    xmin is chosen again on every call, so its uncertainty is part of the CI.

    Args:
        verses: list of verses, each verse is a list of tokens.

    Returns:
        alpha as a float.
    """
    # flatten the verses and count how many times each token appears
    tokens = [t for verse in verses for t in verse]
    counts = list(Counter(tokens).values())
    # MLE fit on integer counts, xmin chosen automatically
    fit = powerlaw.Fit(counts, discrete=True, verbose=False)
    # convert the power-law exponent gamma to the Zipf exponent alpha
    return 1 / (fit.power_law.alpha - 1)


def bootstrap_alpha(verses, fit_fn, n_boot=1000):
    """
    Bootstrap 95% confidence interval for alpha.

    Args:
        verses: list of verses, each verse is a list of tokens.
        fit_fn: function that takes verses and returns alpha (ls_fit or mle_fit).
        n_boot: number of bootstrap samples. Default 1000.

    Returns:
        [lower, upper]: the 2.5th and 97.5th percentiles of the bootstrap alphas.
    """
    alphas = []
    # build n_boot resampled corpora and compute alpha on each one
    for i in range(n_boot):
        sample = resample(verses)
        alphas.append(fit_fn(sample))
    # 95% confidence interval: drop the lowest 2.5% and highest 2.5% of the
    # bootstrap alphas. 95% is the conventional choice in most scientific work.
    return np.percentile(alphas, [2.5, 97.5])


if __name__ == "__main__":
    N_BOOT = 1000
    rows = []

    # "python src/bootstrap.py" runs the real texts,
    # "python src/bootstrap.py baselines" runs the baseline texts
    if len(sys.argv) > 1 and sys.argv[1] == "baselines":
        schemes = BASELINES
        out_file = "bootstrap_baselines.csv"
    else:
        schemes = OPTIONS
        out_file = "bootstrap_alpha.csv"

    # bootstrap every language x scheme
    for language in LANGUAGES:
        for scheme in schemes:
            verses = load_tokens(language, scheme)

            # least squares: alpha on the full corpus + 95% CI
            # same random draws for every combination, so results are reproducible
            random.seed(SEED)
            alpha = ls_fit(verses)
            low, high = bootstrap_alpha(verses, ls_fit, n_boot=N_BOOT)

            # MLE: alpha and xmin on the full corpus from Ani's mle_alpha
            r = mle_alpha(language, scheme)
            xmin = r["xmin"]
            mle = r["alpha"]
            # MLE 95% CI, with the same random draws as least squares
            random.seed(SEED)
            mle_low, mle_high = bootstrap_alpha(verses, mle_fit, n_boot=N_BOOT)

            rows.append([language, scheme, alpha, low, high,
                         xmin, mle, mle_low, mle_high])
            print(language, scheme, round(alpha, 3), round(low, 3), round(high, 3),
                  "| mle", round(mle, 3), round(mle_low, 3), round(mle_high, 3),
                  "xmin", xmin)

    # save the results so the table and plots can read them later
    # without rerunning the bootstrap (which is slow)
    with open(CACHE_DIR / out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "scheme", "alpha", "low", "high",
                         "xmin", "mle_alpha", "mle_low", "mle_high"])  # header
        writer.writerows(rows)                                         # one row per combination
    print("saved", out_file)