import os

import torch
import torch.nn as nn
import torch.optim as optim

import TrainMethod
import eval
import seq2seqVocabPreparation
from model import LuongAttnDecoderRNN, EncoderRNN, GreedySearchDecoder

model_name = 'cb_model'
attn_model = 'dot'
voc = seq2seqVocabPreparation.vocab
pairs =seq2seqVocabPreparation.pairs
decoder_learning_ratio = 5.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 500
batch_size = 32
loadFilename = None
checkpoint_iter =6000
dropout= 0.1

loadFilename = os.path.join("save", model_name,"Cornell_Movie_Dialogue",
                        '{}-{}_{}'.format(2, 2, hidden_size),
                       '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, 2, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words,2, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Ensure dropout layers are in train mode
mode = input("Train or Eval?:" )


if mode == 'Train':
    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 0.7
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 8000
    print_every = 1
    save_every = 500

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    TrainMethod.trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, 2, 2, "save", n_iteration, batch_size,
               print_every, save_every, clip, "Cornell_Movie_Dialogue", loadFilename)



else:
    encoder.eval()
    decoder.eval()
    searcher = GreedySearchDecoder(encoder, decoder)
    #1-5 and Compare
    eval.compareEval(encoder,decoder,searcher,voc)
    #Typing
    #eval.evaluateInput(encoder, decoder, searcher, voc)