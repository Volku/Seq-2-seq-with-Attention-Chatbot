import torch

import TrainMethod
import eval
import seq2seqVocabPreparation
from model import AttnDecoderRNN
from model import EncoderRnn

vocab = seq2seqVocabPreparation.vocab
pairs =seq2seqVocabPreparation.pairs
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 1200
print_every = 50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = seq2seqVocabPreparation.vocab
hidden_size = 500
batch_size = 32

hidden_size = 256

encoder = EncoderRnn(vocab.num_words, hidden_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, vocab.num_words, dropout_p=0.1).to(device)

encoder = encoder.to(device)
attn_decoder = attn_decoder.to(device)

#encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
#decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

# Ensure dropout layers are in train mode
mode = 'Train'

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = eval.evaluate(
        encoder, attn_decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    eval.showAttention(input_sentence, output_words, attentions)

if mode == 'Train':
    encoder.train()
    attn_decoder.train()

    TrainMethod.trainIters(encoder, attn_decoder, 3000, print_every=100,gonnaLoad= False)
else:
   eval.evaluateRandomly(encoder,attn_decoder)