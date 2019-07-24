import random

import torch

import SentenceToTensor
import seq2seqVocabPreparation

pairs = seq2seqVocabPreparation.pairs
EOS_token = seq2seqVocabPreparation.EOS_TOKEN
vocab = seq2seqVocabPreparation.vocab
device = ('gpu' if torch.cuda.is_available() else 'cpu')



def evaluate(encoder, decoder, searcher, voc, sentence, max_length=10):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [SentenceToTensor.indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = seq2seqVocabPreparation.normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 2 or x == 0)]
            print(output_words)
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

def compareEval(encoder,decoder,searcher,voc):

    for i in range(1,11):
        print("===============Short Answer Round ", i)
        pair = random.choice(pairs)
        question = pair[0]
        answer = pair[1]
        longEnough = len(question.split(' ')) < 7
        while not longEnough:
            pair = random.choice(pairs)
            question = pair[0]
            answer = pair[1]
            longEnough = len(question.split(' ')) <7
        print("input: ", question)
        output_words = evaluate(encoder, decoder, searcher, voc, question)
        output_words[:] = [x for x in output_words if not (x == 2 or x == 0)]
        print("Actual: ", answer)
        print('Chatbot:', ' '.join(output_words))
        pair = random.choice(pairs)

    for i in range(1,11):
        print("================Long Answer Round ", i)
        pair = random.choice(pairs)
        question = pair[0]
        answer = pair[1]
        longEnough = len(question.split(' '))>=7
        while not longEnough:
            pair = random.choice(pairs)
            question = pair[0]
            answer = pair[1]
            longEnough = len(question.split(' ')) >= 7
        print("input: ", question)
        output_words = evaluate(encoder, decoder, searcher, voc, question)
        output_words[:] = [x for x in output_words if not (x == 2 or x == 0)]
        print("Actual: ", answer)
        print('Chatbot:', ' '.join(output_words))
        pair = random.choice(pairs)

