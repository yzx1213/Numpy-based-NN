from base import Sequential
from optimizers import *
from utils import *
import numpy as np
from layers import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Find the most suitable learning rate for each type of optimizer.

if __name__ == "__main__":
    batch_size = 50
    train_images, train_labels = load_mnist('mnist/', kind='train')
    train_data=np.column_stack((train_images, train_labels)).astype(float)
    test_images, test_labels = load_mnist('mnist/', kind='t10k')
    test_data =np.column_stack((test_images, test_labels)).astype(float)

    loss_ls=[]
    acc_ls=[]
    for i in range(4):
        net = Sequential(
            Linear(784,64),
            ReLU(),
            Linear(64, 10)
        )
        loss=CEwithSoftMax()
        # opt = SGD(lr=pow(10,-i-1), decay=0.9999)
        # opt = Nesterov(lr=pow(10, -i-1), momentum=0.9)
        # opt = Momentum(pow(10, -i-1), 0.9)
        # opt = AdaGrad(pow(10, -i-1), 1e-9)
        opt = RMSprop(pow(10, -i-1), 1e-8, 0.99)
        # opt=Adam(pow(10, -i-1), (0.9, 0.999), 1e-8)
        net.Compile(loss,opt)
        LOSS,ACC=[],[]
        r = list(range(0,10000,batch_size))
        if r[-1]!=10000:
            r.append(10000)
        for epoch in tqdm(range(100)):
            sum_loss = 0
            data, label = shuffle_data(train_data)
            for b in range(len(r[:-1])):
                batch = data[r[b]:r[b+1],:]
                lab = label[r[b]:r[b+1]]
                pre = net(batch)
                l = net.loss(pre,lab)
                sum_loss += l
                net.backward()
            avg_loss=sum_loss/(len(r)-1)
            LOSS.append(avg_loss)
            # print(avg_loss, end='\r')
            acc=test(net, test_data)
            ACC.append(acc)
            # print(acc, end='\r')
        loss_ls.append(avg_loss)
        acc_ls.append(acc)
        print('lr:{}'.format(pow(10, -i-1)), acc)
        label = '1e-{}'.format(i+1)
        r1 = plt.figure(1)
        plt.plot(LOSS, label=label)
        plt.title('loss')
        plt.xlabel('epoch') 
        plt.ylabel('loss')
        plt.legend()
        r2=plt.figure(2)
        plt.plot(ACC, label=label)
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
    r1.savefig('fig/RMSprop99_loss_lr.png')
    r2.savefig('fig/RMSprop99_acc_lr.png')