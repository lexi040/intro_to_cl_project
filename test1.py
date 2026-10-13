from src.config import *
import spacy, sentencepiece, transformers, powerlaw

for code, fname in LANGUAGES.items():
    p = CORPUS_DIR / fname
    print(code, p.exists())