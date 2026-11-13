import json
import random
from config import CACHE_DIR, SEED, LANGUAGES

def load_tokens(language, scheme):
    path = CACHE_DIR / f"{language}_{scheme}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_tokens(language, scheme, verses):
    path = CACHE_DIR / f"{language}_{scheme}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False)

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

def generate_verse(table, first_word, length):
    verse = [first_word]
    while len(verse) < length:
        current = verse[-1]
        if current in table:
            next_word = random.choice(table[current])
        else:
            next_word = random.choice(list(table))
        verse.append(next_word)
    return verse

def generate_corpus(table, verses):
    fake_verses = []
    for verse in verses:
        fake = generate_verse(table,verse[0], len(verse))
        fake_verses.append(fake)
    return fake_verses

if __name__ == "__main__":
    for language in LANGUAGES:
        random.seed(SEED)
        verses = load_tokens(language, "spacy")
        table = build_bigram_table(verses)
        fake_verses = generate_corpus(table, verses)
        save_tokens(language, "bigram_spacy", fake_verses)
        print(language, len(fake_verses), fake_verses[0][:6])