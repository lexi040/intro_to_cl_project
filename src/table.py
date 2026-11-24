"""
corpus statistics and alpha with bootstrap CI
for every language and tokenization.

Run from the project root (after bootstrap.py): python src/table.py
"""

import pandas as pd
from config import LANGUAGES, OPTIONS, CACHE_DIR
from cache_io import load_tokens
from analysis import a2_counts

if __name__ == "__main__":
    # corpus statistics (N, V, hapax) for every language x tokenization
    stats = []
    for language in LANGUAGES:
        for scheme in OPTIONS:
            counts = a2_counts(load_tokens(language, scheme))
            stats.append({
                "language": language,
                "scheme": scheme,
                "N": counts["tokens"],
                "V": counts["vocabulary"],
                "hapax": counts["hapax"],
            })
    stats = pd.DataFrame(stats)
    # alpha and its bootstrap CI, saved by bootstrap.py
    boot = pd.read_csv(CACHE_DIR / "bootstrap_alpha.csv")
    # join the two tables on language + scheme
    table = stats.merge(boot, on=["language", "scheme"])
    # share of the vocabulary that appears only once
    table["hapax_ratio"] = table["hapax"] / table["V"]
    table = table.round(3)
    table.to_csv(CACHE_DIR / "a3_table.csv", index=False)
    print(table.to_string(index=False))