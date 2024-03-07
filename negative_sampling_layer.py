import numpy as np
from collections import Counter
from layers import *

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.sample_size = sample_size
        #self.sampler = UnigramSampler(corpus, power, sample_size)
        # # sampling
        # word_cnt = Counter(corpus)
        # word_cnt = list(word_cnt.values())
        # p = np.divide(word_cnt,np.sum(word_cnt))
        # p = np.power(p, power)
        # p /= np.sum(p)
        # self.loss_layers = []
        # for i in range(sample_size):
        #     corpus[i].append(np.random.choice(corpus, sample_size, replace=True, p=p))
        
        # self.corpus = corpus
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = 

    def forward(self, idx):
        #loss계산
        loss = 0
        for i in range(self.sample_size):
            y = self.corpus[idx][i]
            t = self.corpus[idx][0]
            loss += self.loss_layer[i].forward(y, y==t)

        return loss

    def backward(self, dout):
        #역전파
        for i in range(self.sample_size):
            loss += self.loss_layer[i].backward(dout)

        return None



