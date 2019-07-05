from __future__ import print_function
import gensim
import gensim.downloader as api
import torch 
import numpy as np

word_vectors = api.load("glove-wiki-gigaword-100") 
numpyVec = word_vectors['hot']
print(numpyVec)
tensor = torch.from_numpy(numpyVec)
print(tensor.view(-1,5))