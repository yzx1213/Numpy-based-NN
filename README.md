# Numpy-based-NN

This repository implement some layers and optimizers used in NN entirely based on Numpy.

The defined **layers** include:

* Linear Layer
* ReLU
* Sigmoid
* Softmax
* CELoss
* MSELoss

The defined **optimizers** include:

* SGD
* Momentum
* Nesterov
* Adagrad
* RMSprop
* Adam

This project established a simple NN trained and tested on MNIST, and the accuracy and time used for all optimizers are listed below.

| optimizers   | SGD   | Momentum | Nesterov | Adagrad | RMSprop | Adam  |
| ------------ | ----- | -------- | -------- | ------- | ------- | ----- |
| accuracy(%)  | 94.22 | 94.96    | 94.73    | 93.96   | 96.11   | 96.16 |
| time used(s) | 47.15 | 49.67    | 55.20    | 50.48   | 56.17   | 63.25 |
