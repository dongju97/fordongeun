import sys
sys.path.append('..')
import numpy as np
from layers import *
from negative_sampling_layer import NegativeSmaplingLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V = vocab_size # 7
        H = hidden_size # 3
        
        #계중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        #계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in # 인스턴스 변수에 단어의 분산 표현을 저장한다.


    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None
 
class CBOW():
    def __init__(self, corpus, vocab_size, hidden_size):
        V = vocab_size # 7
        H = hidden_size # 3
        sample_size = 5
        #계중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        #계층 생성
        layers = []
        for i in range(sample_size):
            layers.append(Embedding(W_in))
            
        self.out_layer = Embedding(W_out)
        self.loss_layer = NegativeSmaplingLoss(W_in, corpus)

        #layers = [self.in_layer, self.out_layer, self.loss_layer]
        layers.extend([self.out_layer, self.loss_layer])

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = Embedding(W_in) # 인스턴스 변수에 단어의 분산 표현을 저장한다.


    def forward(self, contexts, target):
        h0 = self.in_layer.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = np.dot(self.out_layer.forward(target), h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None