import os
import re


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text

def clean_file(filename):
    _file = open(filename, encoding='utf-8', errors='ignore').read().split('\n\n')
    lines= []
    for line in _file:
        lines.append(clean_text(line))

    
    questions = []
    answers = []
    for i in range(len(lines)):
        if(i%2 ==0):
            questions.append(lines[i])
        else:

            answers.append(lines[i])    
    limit = 0
    for i in (limit,limit+10) :
        print('------------------------------------QUESTION----------------------------------------')       
        print(questions[i])
        print('------------------------------------ANSWER----------------------------------------')       
        print(answers[i])

    return questions,answers

clean_file('data/train.enc')

class Sentence:
    def __init__(self):
        self.word2count ={}
        self.trim = FALSE
        
    