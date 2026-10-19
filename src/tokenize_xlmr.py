from data_processing import read_bible
from transformers import AutoTokenizer
from config import XLMR_MODEL
import json
from config import CACHE_DIR
from analysis import get_word_frequencies, plot_zipf

def save_tokens(language, scheme, per_verse_lists):
    path = CACHE_DIR / f"{language}_{scheme}.json"
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(per_verse_lists, f, ensure_ascii=False)

tokenizer = AutoTokenizer.from_pretrained(XLMR_MODEL)

def tokenize_xlmr(verses):
    result = []
    for verse in verses:
        tokens = tokenizer.tokenize(verse["text"])
        result.append(tokens)
    return result

corpus = read_bible()
for language, verses in corpus.items():
    save_tokens(language, "xlmr", tokenize_xlmr(verses))
    get_word_frequencies([token for verse in tokenize_xlmr(verses) for token in verse])
    plot_zipf({"xlmr": [token for verse in tokenize_xlmr(verses) for token in verse]}, language)    
    print(language, "done")


