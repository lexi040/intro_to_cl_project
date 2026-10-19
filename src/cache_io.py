import json
from pathlib import Path
 
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
def save_tokens(language, scheme, per_verse_lists):
    path = CACHE_DIR / f"{language}_{scheme}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(per_verse_lists, f, ensure_ascii=False)
