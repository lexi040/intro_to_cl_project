from data_processing import read_bible
from config import SPACY_MODELS, CACHE_DIR
import json
import spacy

def tokenize_ws_raw(verses):
    result = []
    for verse in verses:
        tokens=verse["text"].split()
        result.append(tokens)
    return result

def tokenize_ws_lower(verses, language):
    result = []
    for verse in verses:
        tokens = lowercase(verse["text"], language).split()
        result.append(tokens)
    return result

def lowercase(text, language):   
    if language =="tr":
        text = text.replace("İ", "i").replace("I", "ı")
    return text.lower()
        
def tokenize_spacy(verses, language):
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
    path = CACHE_DIR / f"{language}_{scheme}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(per_verse_lists, f, ensure_ascii=False)

if __name__ == "__main__":
    corpus = read_bible()
    for language, verses in corpus.items():
        save_tokens(language, "ws_raw", tokenize_ws_raw(verses))
        save_tokens(language, "ws_lower", tokenize_ws_lower(verses, language))
        save_tokens(language, "spacy", tokenize_spacy(verses, language))

        print(language, "done")