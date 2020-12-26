from copy import deepcopy as dcp
# Here defines the basic classes.


class Sequential:  # used for compiling different layers.
    def __init__(self, *args):
        self.net = args
        self.loss = None

    def __call__(self, x):
        for n in self.net:
            x = n(x)
        return x

    def backward(self, g=1):
        if self.loss is not None:
            g = self.loss.backward()
        for n in self.net[::-1]:
            g = n.BACKWARD(g)

    def register(self, opt):
        for n in self.net:
            n.register(dcp(opt))

    def Compile(self, loss, opt):
        self.loss = loss
        self.register(opt)


class BaseNetwork: # Base class for layers.
    def __init__(self, *params):
        self.params = params

    def __call__(self, *x):
        if isinstance(x, tuple) and len(x) == 1:
            self.x = x[0]
        else:
            self.x = x
        self.y = self.forward(*x)
        return self.y

    def forward(self, *x):
        # need to be overloaded.
        pass

    def backward(self, g):
        # need to be overloaded.
        pass

    def register(self, opt):
        self.opt = []
        for i in range(len(self.params)):
            self.opt.append(dcp(opt))

    def BACKWARD(self, g):
        if len(self.params) == 0:
            G = self.backward(g)
        else:
            G, *delta = self.backward(g)
            for o, w, d in zip(self.opt, self.params, delta):
                self.__dict__[w] = o(self.__dict__[w], d)
        return G
