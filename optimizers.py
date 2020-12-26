import numpy as np

# Here defines six types of optimizers. They get the gradients and then decide how to update the parameters.

class SGD():
    def __init__(self, lr=0.01, decay=0.999):
        self.lr = lr
        self.decay=decay
    def __call__(self, w, g):
        w -= self.lr * g
        self.lr*=self.decay
        return w
    

class Momentum(): # 在梯度下降的基础上加入了动量，即前面的梯度将会影响本轮的梯度方向
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def __call__(self, w, g):
        if self.v is None:
            self.v = np.zeros_like(w)

        self.v = self.momentum * self.v - self.lr * g
        w += self.v
        return w
    
class Nesterov(): # 为了加速收敛，提前按照之前的动量走了一步，然后求导后按着梯度再走一步
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def __call__(self, w, g):
        if self.v is None:
            self.v = np.zeros_like(w)

        self.v = self.momentum * self.v - self.lr * g
        w += self.momentum * self.v - self.lr * g
        return w

class AdaGrad(): # 通过每个参数的历史梯度，动态更新每一个参数的学习率，使得每个参数的更新率都能够逐渐减小。
    def __init__(self, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.h = None

    def __call__(self, w, g):
        if self.h is None:
            self.h = np.zeros_like(w)

        self.h += g*g
        w -= self.lr * g / (np.sqrt(self.h) + self.epsilon)
        return w

class RMSprop(): # 使用指数衰减平均来慢慢丢弃先前得梯度历史, 防止学习率过早减小
    def __init__(self, lr=0.01, epsilon=1e-8, decay = 0.99):
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        self.h = None

    def __call__(self, w, g):
        if self.h is None:
            self.h = np.zeros_like(w)

        self.h *= self.decay
        self.h += (1 - self.decay) * (g ** 2)
        w -= self.lr * g / (np.sqrt(self.h) + self.epsilon)
        return w
    
class Adam(): # 利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率
    def __init__(self, lr=0.01, beta = (0.9, 0.999), epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta[0]
        self.beta2 = beta[1]
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.n = 0

    def __call__(self, w, g):
        if self.m is None:
            self.m = np.zeros_like(w)
        if self.v is None:
            self.v = np.zeros_like(w)
            
        self.n += 1
        alpha = self.lr * np.sqrt(1 - self.beta2 ** self.n) / (1 - self.beta1 ** self.n)
        
        self.m = self.beta1 * self.m + (1-self.beta1) * g
        self.v = self.beta2 * self.v + (1-self.beta2) * g ** 2

        w -= alpha * self.m / (np.sqrt(self.v) + self.epsilon)
        return w