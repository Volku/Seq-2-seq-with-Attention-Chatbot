import torch
import torch.nn

import seq2seqVocabPreparation

pairs = seq2seqVocabPreparation.pairs
PAD_TOKEN = seq2seqVocabPreparation.PAD_TOKEN
EOS_TOKEN = seq2seqVocabPreparation.EOS_TOKEN
vocab = seq2seqVocabPreparation.vocab

device = ("gpu" if torch.cuda.is_available() else "cpu")
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_TOKEN)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(vocab, pair[0])
    target_tensor = tensorFromSentence(vocab, pair[1])
    return (input_tensor, target_tensor)

