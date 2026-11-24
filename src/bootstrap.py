"""
A3 bootstrap: confidence intervals for the Zipf exponent alpha.

Idea: we resample the verses with replacement (some verses appear several
times, some not at all) to build many slightly different versions of the
corpus. We compute alpha on each version. The spread of these alphas shows
how stable our alpha estimate is.

The fit function is passed as a parameter, so the same bootstrap works for
any alpha method: least squares or MLE.

"""

import random
import numpy as np
import csv
from config import SEED, LANGUAGES, OPTIONS, CACHE_DIR
from cache_io import load_tokens
from collections import Counter
from analysis import ls_alpha

# Number of top ranks used for the least squares fit.
# 500 keeps every language x tokenization out of the noisy rare-word tail
# (see figures/rmax_comparison_en.png and Clauset et al. 2009).
RMAX = 500

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


def bootstrap_alpha(verses, fit_fn, n_boot=1000):
    """
    Bootstrap 95% confidence interval for alpha.

    Args:
        verses: list of verses, each verse is a list of tokens.
        fit_fn: function that takes verses and returns alpha (e.g. ls_fit).
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

    # bootstrap every language x tokenization (5 x 7 = 35 combinations)
    for language in LANGUAGES:
        for scheme in OPTIONS:
            # same random draws for every combination, so results are reproducible
            random.seed(SEED)
            verses = load_tokens(language, scheme)
            # alpha on the full corpus (no resampling)
            alpha = ls_fit(verses)
            # 95% CI from N_BOOT resampled corpora
            low, high = bootstrap_alpha(verses, ls_fit, n_boot=N_BOOT)
            rows.append([language, scheme, alpha, low, high])
            print(language, scheme, round(alpha, 3), round(low, 3), round(high, 3))

    # save the results so the table and plots can read them later
    # without rerunning the bootstrap (which is slow)
    with open(CACHE_DIR / "bootstrap_alpha.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "scheme", "alpha", "low", "high"])  # header
        writer.writerows(rows)                                           # one row per combination
    print("saved")
