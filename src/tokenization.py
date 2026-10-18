import sentencepiece as spm

from config import LANGUAGES, BPE_VOCAB_SIZES

#training the model might need iteration over languages/ vocab sizes
spm.SentencePieceTrainer.train(
    input = LANGUAGES, 
    model_prefix = 'm',
    vocab_size = BPE_VOCAB_SIZES
)

#loading trained model
sp = spm.SentencePieceProcessor(model_file = 'm.model')

#encode raw text into subowrd seqs + vocab ids
text = "dont know yet" 
pieces = sp.encode(text, return_type = str)
ids = sp.encode(text, return_type = int)

print(f"Pieces : {pieces}")

print(f"IDs : {ids}")

print(sp.decode(ids))
print(sp.decode(pieces))

