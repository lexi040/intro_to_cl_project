import random
from pathlib import Path
from collections import Counter
from data_processing import read_bible
from cache_io import save_tokens
from config import SEED

random.seed(SEED)

corpus = read_bible()
for language, verses in corpus.items():
    all_text = " ".join(v["text"] for v in verses)
    char_counts = Counter(all_text)
    chars = list(char_counts.keys())
    weights = list(char_counts.values())

    random_verses = []
    for verse in verses:
        n_tokens = len(verse["text"].split())
        words = []
        for _ in range(n_tokens):
            word = ""
            while True:
                c = random.choices(chars, weights=weights, k=1)[0]
                if c == " ":
                    break
                word += c
            words.append(word)
        random_verses.append(words)


    save_tokens(language, "random", random_verses)
    print(language, "done")