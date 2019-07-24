import os
import re
import unicodedata

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join(corpus_name)
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Vocab:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index ={}
        self.word2count ={}
        self.index2word ={PAD_TOKEN:"PAD",SOS_TOKEN:"SOS",EOS_TOKEN:"EOS"}
        self.num_words = 3

    def addSentence(self,sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self,min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k,v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print(' keep {}/{} = {:.2f}'.format(
            len(keep_words),len(self.word2index), len(keep_words)/len(self.word2index)
        ))

        #Re-init
        self.word2count = {}
        self.word2index = {}
        self.index2word = {PAD_TOKEN:0,SOS_TOKEN:1 ,EOS_TOKEN:2}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalizeString(s):
    # Unicode string to plain ASCII
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

datafile = 'train/dial.txt'

lines = open(datafile,encoding = 'utf-8').read().strip().split('\n')
pairs = []

index= 0
for id in range(0, len(lines), 2):
   # print(lines[0])
   pairs.append([])
   pairs[index].append(normalizeString(lines[id]))
   pairs[index].append(normalizeString(lines[id+1]))
   index+=1





vocab = Vocab(corpus)

MAX_LENGTH = 10

def filterPair(p):
   if p :
       return len(p[0].split(' '))< MAX_LENGTH and len(p[1].split(' '))< MAX_LENGTH
   else:
        return False

#Use only the pair that length is not more than 15 words
def filterPairs(pairs):
    pairs = [pair for pair in pairs if pair != ['']]
    return [pair for pair in pairs if filterPair(pair)]

print("There are {} pairs/conversations in the dataset".format(len(pairs)))
pairs = filterPairs(pairs)
print("After filtering there are {} pairs/conversations".format(len(pairs)))



for pair in pairs:
        vocab.addSentence(pair[1])
        vocab.addSentence(pair[1])

print("Counted words:", vocab.num_words)
for pair in pairs[:10]:
    print(pair)

MIN_COUNT = 3    # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

pairs = trimRareWords(vocab, pairs, MIN_COUNT)






