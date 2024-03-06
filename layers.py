import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from funtions import *

class MatMul(nn.Module):
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        #self.W = W

    def forward(self, x):
        W, = self.params
        self.x = x
        out = np.dot(x, W)
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

# class Softmax():
#     def __init__(self):
#         self.params, self.grads = [], []
#         self.out = None

#     def forward(self, x):


#     def backward():

class SoftmaxWithLoss():
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블
        #self.softmax = nn.functional.softmax()
        #self.ce = nn.CrossEntropyLoss()

    def forward(self, score, target):
        self.t = target
        self.y = softmax(score)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1) # Returns the indices of the maximum values along an axis.
        
        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx









