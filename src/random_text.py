import random
from collections import Counter
from config import LANGUAGES, OPTIONS, BPE_VOCAB_SIZES, SEED, CACHE_DIR
from cache_io import load_tokens, save_tokens
from tokenize_ws_spacy import tokenize_ws_raw, tokenize_ws_lower, tokenize_spacy
from tokenize_xlmr import tokenize_xlmr                 # adjust to real module name
from bpe_tokenization import train_bpe, tokenize_bpe    # adjust to Lexi's file name


MODEL_DIR = CACHE_DIR / "spm_models"             # same dir Lexi uses

def word_stream(chars, weights):
    """Yield random words forever, drawing 1000 chars at a time (fast)."""
    buf = ""
    while True:
        buf += "".join(random.choices(chars, weights=weights, k=1000))
        *done, buf = buf.split(" ")              # keep last partial word in buf
        yield from (w for w in done if w)        # drop empty words ← the bug fix
if __name__ == "__main__":
    for language in LANGUAGES:
        print("language =", repr(language))                        # keys: "en","fr","de","fi","tr"
        random.seed(SEED) 

        # size reference: spaCy (word-level), per Lexi's "use one natural scheme"
        # Turkish has no spaCy model → fall back to ws_lower for tr only
        ref_scheme = "spacy"
        verses = load_tokens(language, ref_scheme)

        # letter + space frequencies of the real language (A2 requirement)
        counts = Counter(" ".join(" ".join(v) for v in verses))
        chars, weights = zip(*counts.items())

        # ONE random text, shaped like the reference verses
        stream = word_stream(chars, weights)
        fake_verses = [[next(stream) for _ in v] for v in verses]
        # convert to dicts so Öykü's tokenizers (which read verse["text"]) accept it
        fake_dicts = [{"text": " ".join(words)} for words in fake_verses]

        # --- save one baseline per scheme, all from the SAME fake_dicts ---
        save_tokens(language, "random_ws_raw",   tokenize_ws_raw(fake_dicts))
        save_tokens(language, "random_ws_lower", tokenize_ws_lower(fake_dicts, language))
        save_tokens(language, "random_spacy", tokenize_spacy(fake_dicts, language))
        save_tokens(language, "random_xlmr",     tokenize_xlmr(fake_dicts))

        # BPE: model is trained on REAL verses (it's a property of the language),
        # then applied to the random text.
        for size in BPE_VOCAB_SIZES:
            # if the model doesn't exist yet, train it on the real corpus:
            model_path = str(MODEL_DIR / f"_{language}_bpe{size}.model")
            save_tokens(language, f"random_bpe{size}", tokenize_bpe(fake_dicts, model_path))
            
        # size check Lexi couldn't do: real vs random token counts
        real_flat = [t for v in verses for t in v]
        rand_flat = [t for v in fake_verses for t in v]
        print(f"{language}: real={len(real_flat)} random={len(rand_flat)} "
                        f"ratio={len(rand_flat)/len(real_flat):.3f}")