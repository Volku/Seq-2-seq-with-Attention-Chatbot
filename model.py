import torch.nn as nn


class EncoderAttn(nn.module):

    def __init__(self,input_size,hidden_size):
        self.n_layer = n_layer
        self.