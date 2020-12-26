from base import BaseNetwork
import numpy as np

# Here defines several basic layers, including rules for forward and backward propagating, etc.

class Linear(BaseNetwork):
    def __init__(self, in_channel, out_channel):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.w = np.random.normal(0, 1, (in_channel, out_channel))
        self.b = np.random.normal(0, 1, (1, out_channel))
        super(Linear, self).__init__('w', 'b')

    def forward(self, x):
        y = np.dot(x, self.w)+self.b
        return y

    def backward(self, g):
        delta_w = np.dot(self.x.reshape(-1, self.in_channel).T,
                         g.reshape(-1, self.out_channel))
        delta_b = g.reshape(-1, self.out_channel).sum(0)
        return np.dot(g, self.w.T), delta_w, delta_b


class Sigmoid(BaseNetwork):
    def forward(self, x):
        return 1/(1+np.exp(-x))

    def backward(self, g):
        return self.y*(1-self.y)*g


class ReLU(BaseNetwork):
    def forward(self, x):
        self.mask = x < 0
        x[self.mask] = 0
        return x

    def backward(self, g):
        g[self.mask] = 0
        return g


class SoftMax(BaseNetwork):
    def forward(self, x):
        exp = np.exp(x-x.max(-1)[..., np.newaxis])
        return exp/exp.sum(-1)[..., np.newaxis]

    def backward(self, x):
        return self.y-1


class MSELoss(BaseNetwork):
    def forward(self, y, y_):
        return (0.5*(y-y_)**2).mean()

    def backward(self, g=1):
        return (self.x[0]-self.x[1])/self.x[0].shape[0]


class CELoss(BaseNetwork):
    def forward(self, y, y_):
        return -(y_*np.log(y+1e-9)).sum()

    def backward(self, g=1):
        return -self.x[1]/self.x[0]


class CEwithSoftMax(BaseNetwork):
    def forward(self, pre, y_):
        self.sftm = np.exp(pre-pre.max(-1)[..., np.newaxis])
        self.sftm = self.sftm/self.sftm.sum(-1)[..., np.newaxis]
        return -(y_*np.log(self.sftm+1e-9)).sum()

    def backward(self, g=1):
        return (self.sftm-self.x[1])
