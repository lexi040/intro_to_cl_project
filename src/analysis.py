from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

def get_word_frequencies(tokens):
    counter = Counter(tokens)
    sorted_frequencies = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    rank_frequency_distribution = []
    for rank, (word, frequency) in enumerate(sorted_frequencies, start=1):
        rank_frequency_distribution.append({
            "rank": rank,
            "token": word,
            "frequency": frequency
        })
    return rank_frequency_distribution

def plot_zipf(rank_frequency_distribution, language, sp = False, size = None):
    #scalable i would like to say should work for both sentencepiece bpe tokenizer w mutiple vocab sizes and roberta and spacy tokenizers
    plt.figure(figsize=(7, 7))
    for scheme_name, tokens in rank_frequency_distribution.items():
        if isinstance(tokens[0], str) and isinstance(tokens, list):
            dist = get_word_frequencies(tokens)
        else:
            dist = tokens

        ranks = [item["rank"] for item in dist]
        frequencies = [item["frequency"] for item in dist]

        plt.loglog(ranks, frequencies, marker='.', linestyle='none', alpha=0.7, label=scheme_name)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title(f'Zipf\'s Law for {language}')
    plt.legend()
    plt.grid(True)
    if sp and size is not None:
        filename = (f"cache/zipf_{language}_bpe{size}.png")
    else:
        filename = (f"cache/zipf_{language}.png")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
