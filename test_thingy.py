#this is just benoit s code from the project file ignore for now, shouldnt work



import collections
import numpy as np


def rank_freq(tokens):
    return np.array(sorted(collections.Counter(tokens).values(), reverse=True))


# Text to analyze. Replace with the actual corpus text if needed.
text = ""

# spaCy (python -m spacy download fi_core_news_sm)
import spacy

nlp = spacy.load("fi_core_news_sm", disable=["parser", "ner"])
words = [t.lower_ for t in nlp(text) if not (t.is_punct or t.is_space)]

# SentencePiece BPE trained on one language
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="fi.txt",
    model_prefix="fi_bpe8k",
    vocab_size=8000,
    model_type="bpe",
)
pieces = spm.SentencePieceProcessor(model_file="fi_bpe8k.model").encode(text, out_type=str)

# XLM-R tokenizer
from transformers import AutoTokenizer

xlmr = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
xlmr_tokens = xlmr.tokenize(text)

# Maximum likelihood fit. NB: powerlaw fits the frequency distribution
# P(c) ~ c^(-gamma), not the rank curve; for pure Zipf gamma = 1 + 1/alpha.
import powerlaw

fit = powerlaw.Fit(rank_freq(words), discrete=True)

# Least-squares estimate of alpha on the log-log rank-frequency points
c = rank_freq(words)
r = np.arange(1, len(c) + 1)
alpha_ls = -np.polyfit(np.log(r), np.log(c), 1)[0]

# Brown documents grouped by genre, and a Heaps curve
'''from nltk.corpus import brown

docs = [
    [w.lower() for w in brown.words(fileids=f) if w.isalpha()]
    for g in brown.categories()
    for f in brown.fileids(categories=g)
]


def heaps(docs, step=1000):
    seen, n, xs, ys = set(), 0, [], []
    for doc in docs:
        for tok in doc:
            seen.add(tok)
            n += 1
        if n % step == 0:
            xs.append(n)
            ys.append(len(seen))
    return np.array(xs), np.array(ys)


# Static embeddings: word2vec trained on Brown, pretrained GloVe
from gensim.models import Word2Vec
import gensim.downloader as api

w2v = Word2Vec(
    sentences=docs,
    vector_size=100,
    window=5,
    min_count=5,
    sg=1,
    seed=0,
    workers=1,
).wv
glove = api.load("glove-wiki-gigaword-100")


def doc_vector(doc, kv):
    return np.mean([kv[w] for w in doc if w in kv], axis=0)


print(glove.most_similar("church", topn=5))'''