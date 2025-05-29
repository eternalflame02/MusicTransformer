# test_model.py

from model import MusicTransformer
import torch

vocab_size = 512  # update this based on your tokenizer
seq_len = 128

model = MusicTransformer(vocab_size=vocab_size)

dummy_input = torch.randint(0, vocab_size, (2, seq_len))  # batch size = 2
out = model(dummy_input)

print("Output shape:", out.shape)  # should be [2, 128, vocab_size]
