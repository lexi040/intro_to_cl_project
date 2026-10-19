import sentencepiece as spm

from config import CACHE_DIR, BPE_VOCAB_SIZES, LANGUAGES    
from cache_io import save_tokens
from data_processing import read_bible

from analysis import get_word_frequencies, plot_zipf

MODEL_DIR = CACHE_DIR / "spm_models"
MODEL_DIR.mkdir(exist_ok=True)
dataset = read_bible()  

def train_bpe(verses, language, vocab_size):
    #turn token list into file to fir sentencepiece trainer
    prefix = MODEL_DIR / f"_{language}_bpe{vocab_size}"     
    model = prefix.with_suffix(".model")
    
    if not model.exists():
        corpus = prefix.with_suffix(".txt")
        corpus.write_text("\n".join(v["text"] for v in verses), encoding="utf-8")   
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            #change to bpe instead of defaulting to unigram
            model_type="bpe",   
            #make sure all ! chracters are included in the vocab             
            character_coverage=1.0,
            #preventively added in case for any of the languages the corpus doesnt have enough ! combinations to hit the 32k vocab size
            hard_vocab_limit=False,   
            #suppresses verbose logging output        
            #minloglevel=2,
        )
        corpus.unlink()
    return str(model)

def tokenize_bpe(verses, model_path):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    #reutrns a list of subowrd token strings not int ids 
    #matches spacy and whitespace tokenizers 
    #before running through rank frequency distribution must flatten the list of lists into a single list of tokens
    return [sp.encode(verse["text"], out_type=str) for verse in verses]


if __name__ == "__main__":
        corpus = read_bible()
        for language, verses in corpus.items():
            for size in BPE_VOCAB_SIZES:
                model = train_bpe(verses, language, size)
                verse_tokens = tokenize_bpe(verses, model)
                flat_tokens = [token for verse in verse_tokens for token in verse]
                save_tokens( language, f"bpe{size}", flat_tokens)
                get_word_frequencies(flat_tokens)
                plot_zipf({f"bpe{size}": flat_tokens}, language, sp=True, size=size)
                print(language, "done")

        
