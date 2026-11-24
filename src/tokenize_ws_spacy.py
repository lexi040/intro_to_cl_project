"""
Whitespace and spaCy tokenizers for the aligned verses of all 5 languages.
Saves ws_raw, ws_lower and spacy token files to the cache.
"""

from data_processing import read_bible, align_verses
from config import SPACY_MODELS, CACHE_DIR
import json
import spacy
from analysis import get_word_frequencies, plot_zipf

def tokenize_ws_raw(verses):
    """
    Splits each verse on whitespace without any other changes.
    Returns a list of verses, each verse is a list of tokens.
    """
    result = []
    for verse in verses:
        tokens=verse["text"].split()
        result.append(tokens)
    return result

def tokenize_ws_lower(verses, language):
    """
    Lowercases each verse (with Turkish rules for tr) and splits it on whitespace.
    """
    result = []
    for verse in verses:
        tokens = lowercase(verse["text"], language).split()
        result.append(tokens)
    return result

def lowercase(text, language):
    """
    Lowercases text. For Turkish, maps İ to i and I to ı first,
    since Python's lower() does not handle the Turkish dotted/dotless i.
    """
    if language =="tr":
        text = text.replace("İ", "i").replace("I", "ı")
    return text.lower()

def tokenize_spacy(verses, language):
    """
    Tokenizes each verse with spaCy, removes punctuation and spaces, and lowercases tokens.
    Turkish uses spacy.blank("tr") because there is no small Turkish model.
    """
    if language == "tr":
        nlp = spacy.blank("tr")
    else:
        nlp = spacy.load(SPACY_MODELS[language])
    result = []
    for verse in verses:
        doc = nlp(verse["text"])
        tokens=[]
        for t in doc:
            if not t.is_punct and not t.is_space:
                tokens.append(lowercase(t.text, language))
        result.append(tokens)
    return result

def save_tokens(language, scheme, per_verse_lists):
    """
    Saves a list of tokenized verses to cache/{language}_{scheme}.json.
    """
    path = CACHE_DIR / f"{language}_{scheme}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(per_verse_lists, f, ensure_ascii=False)

if __name__ == "__main__":
    corpus = align_verses(read_bible())
    for language, verses in corpus.items():
        save_tokens(language, "ws_raw", tokenize_ws_raw(verses))
        save_tokens(language, "ws_lower", tokenize_ws_lower(verses, language))
        save_tokens(language, "spacy", tokenize_spacy(verses, language))
        get_word_frequencies([token for verse in tokenize_ws_raw(verses) for token in verse])
        plot_zipf({"ws_raw": [token for verse in tokenize_ws_raw(verses) for token in verse]}, language)
        print(language, "done")