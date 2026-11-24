from config import CACHE_DIR, LANGUAGES
from cache_io import load_tokens
from collections import Counter
import powerlaw
import numpy as np


def mle_alpha(language, scheme="xlmr"):
    """MLE Zipf exponent + corpus stats for one language × tokenization scheme."""
    data = load_tokens(lang, "xlmr")            # one list of per-verse
    flat = [tok for verse in data for tok in verse]
    counts = list(Counter(flat).values())       # how many times a token appears


    fit = powerlaw.Fit(counts, discrete=True)   # MLE fit on integer counts. discrete=True because frequencies are integers. With the discrete=True, word frequencies are treated as integer counts, without it, the powerlaw treats them as continuous and γ comes out biased#
    gamma = fit.power_law.alpha                 # γ = exponent of the fitted power-law PDF (NOT the Zipf α)#
    xmin = fit.power_law.xmin                   # lower bound of the fitted tail, auto-chosen by KS minimization. The xmin was was autoselected by the library.
    alpha_zipf = 1 / (gamma - 1)                # convert PDF exponent γ to Zipf rank exponent α. this is what my number comparable with lexi's part.

    n_tokens = sum(counts)                      #N = total token occurrences
    vocab_size = len(counts)                    # |V| = number of distinct tokens (full vocabulary)
    hapax = sum(1 for c in counts if c == 1)    # tokens that appear exactly once

    return {
        "language": language, "scheme": scheme,
        "gamma": gamma, "xmin": xmin, "alpha": alpha_zipf,
        "N": n_tokens, "V": vocab_size,
        "hapax": hapax, "hapax_prop": hapax / vocab_size,
    }

if __name__ == "__main__":
    for lang in LANGUAGES:
        r = mle_alpha(lang)                                                     # fit one language
        print(f"{r['language']}: γ={r['gamma']:.4f} xmin={r['xmin']} "          # report γ to 4 decimals
              f"α={r['alpha']:.4f} |V|={r['V']} N={r['N']} "                    # and α, |V|, N on the same line
              f"hapax={r['hapax']} ({r['hapax_prop']:.1%})")                    # hapax count and % of vocabulary

    "-α test"  
    # Validation: recover a known γ from synthetic data
    rng = np.random.default_rng(0)                                              # fixed seed → reproducible synthetic sample
    synthetic = powerlaw.Power_Law(                                             # build a synthetic discrete power law…
        parameters=[2.0], xmin=1, discrete=True).generate_random(10000)         # with known γ=2.0, integer values, lower bound 1…# …and draw 10,000 samples from it
    fit_syn = powerlaw.Fit(synthetic, discrete=True)                            # run the same fit on the synthetic data
    print("recovered γ:", fit_syn.power_law.alpha)                              # should be close to 2.0 #





#this was what I used to test wether this worked or not. It is a very visual explanation. 
#data = load_tokens("en", "xlmr")
#print(type(data))        # expect: <class 'list'>
#print(len(data))         # expect: number of aligned verses
#print(type(data[0]))     # expect: <class 'list'>
#print(data[0][:5])       

#flat = [tok for verse in data for tok in verse]
#print(type(flat))        # expect: <class 'list'>
#print(len(flat))         # expect: much larger than len(data) — total tokens
#print(flat[:5])          # expect: same first tokens as data[0][:5]
#print(all(isinstance(t, str) for t in flat[:100]))  # expect: True

#counts = list(Counter(flat).values())
#print(len(counts))       # expect: vocab size |V|, in the thousands to tens of thousands
#print(sum(counts))       # expect: equals len(flat) — total tokens N
#print(max(counts))       # expect: thousands (the most frequent token)
#print(counts[:10])       # expect: ints, roughly descending? (not guaranteed, but plausible)
#print(all(isinstance(c, int) for c in counts))  # expect: True

#fit = powerlaw.Fit(counts, discrete=True)
#print("γ =", fit.power_law.alpha)
#print("xmin =", fit.power_law.xmin)
#print("α_zipf =", 1 / (fit.power_law.alpha - 1))

