import json
import random
from config import CACHE_DIR, SEED, LANGUAGES
from cache_io import load_tokens, save_tokens


def build_bigram_table(verses):
    table = {}
    for verse in verses:
        for i in range(len(verse) - 1):
            word = verse[i]
            next_word = verse[i + 1]
            if word not in table:
                table[word] = []
            table[word].append(next_word)
    return table

def generate_verse(table, all_tokens, first_word, length):
    verse = [first_word]
    while len(verse) < length:
        current = verse[-1]
        if current in table:
            next_word = random.choice(table[current])
        else:
            next_word = random.choice(all_tokens)
        verse.append(next_word)
    return verse

def generate_corpus(table, all_tokens, verses):
    fake_verses = []
    for verse in verses:
        fake = generate_verse(table, all_tokens, verse[0], len(verse))
        fake_verses.append(fake)
    return fake_verses

if __name__ == "__main__":
    for language in LANGUAGES:
        for scheme in ["spacy", "xlmr", "bpe2000", "bpe8000", "bpe32000"]:
            random.seed(SEED)
            verses = load_tokens(language, scheme)
            table = build_bigram_table(verses)
            all_tokens = [t for v in verses for t in v]
            fake_verses = generate_corpus(table, all_tokens, verses)
            save_tokens(language, f"bigram_{scheme}", fake_verses)
            print(language, scheme, len(fake_verses), fake_verses[0][:6])