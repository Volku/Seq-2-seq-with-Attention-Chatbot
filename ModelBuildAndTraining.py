import torch
import torch.nn as nn
import torch.optim as optim

import TrainMethod
import seq2seqVocabPreparation
from model import EncoderRnn
from model import LuongAttnDecoderRNN

pairs =seq2seqVocabPreparation.pairs
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.00002
decoder_learning_ratio = 5.0
n_iteration = 1000
print_every = 50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = seq2seqVocabPreparation.vocab
hidden_size = 500
batch_size = 32

embedding = nn.Embedding(vocab.num_words, hidden_size)
encoder = EncoderRnn(hidden_size, embedding)
decoder = LuongAttnDecoderRNN(embedding, hidden_size, vocab.num_words)

encoder = encoder.to(device)
decoder = decoder.to(device)



encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

TrainMethod.trainIters( pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, n_iteration, batch_size,
           print_every, clip)