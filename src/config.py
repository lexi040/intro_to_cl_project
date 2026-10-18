from pathlib import Path

SEED = 42

#makes path relative to the repo root directory=> works regardless of where the script is run from
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "bible-corpus" / "bibles"   
CACHE_DIR = ROOT / "cache"
FIG_DIR = ROOT / "figures"

LANGUAGES = {
    "en": "English.xml",
    "fr": "French.xml",
    "de": "German.xml",
    "fi": "Finnish.xml",
    "tr": "Turkish.xml",
}

SPACY_MODELS = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "fi": "fi_core_news_sm",
    "tr": None,   
}

BPE_VOCAB_SIZES = [2000, 8000, 32000]
XLMR_MODEL = "FacebookAI/xlm-roberta-base"

for d in (CACHE_DIR, FIG_DIR):
    d.mkdir(exist_ok=True)