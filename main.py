import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from base import Sequential
from optimizers import *
from utils import *
import numpy as np
from layers import *
import matplotlib
matplotlib.use('Agg')

# implement training for all optimizers.

def all_optimizers(opt_name, opt_ls, batch_size, iter_num):
    train_images, train_labels = load_mnist('mnist/', kind='train')
    train_data = np.column_stack((train_images, train_labels)).astype(float)
    test_images, test_labels = load_mnist('mnist/', kind='t10k')
    test_data = np.column_stack((test_images, test_labels)).astype(float)
    loss_ls, acc_ls, time_ls = [], [], []
    for i in range(len(opt_ls)):
        opt = opt_ls[i]
        net = Sequential(
            Linear(784, 64),
            ReLU(),
            Linear(64, 10)
        )
        loss = CEwithSoftMax()
        net.Compile(loss, opt)
        # train
        # mini batch
        LOSS, ACC = [], []
        r = list(range(0, 10000, batch_size))
        if r[-1] != 10000:
            r.append(10000)
        t = time.time()
        for epoch in tqdm(range(iter_num)):
            sum_loss = 0
            data, label = shuffle_data(train_data)
            for b in range(len(r[:-1])):
                batch = data[r[b]:r[b+1], :]
                lab = label[r[b]:r[b+1]]
                pre = net(batch)
                l = net.loss(pre, lab)
                sum_loss += l
                net.backward()
            avg_loss = sum_loss/(len(r)-1)
            LOSS.append(avg_loss)
            # print(avg_loss, end='\r')
            acc = test(net, test_data)
            ACC.append(acc)
            # print(acc, end='\r')
        loss_ls.append(avg_loss)
        acc_ls.append(acc)
        time_ls.append(time.time()-t)
        print(opt_name[i], acc)
        np.save('npy/'+opt_name[i]+'_loss.npy', LOSS)
        np.save('npy/' + opt_name[i] + '_acc.npy', ACC)
    np.save('npy/time_{}.npy'.format(iter_num), time_ls)
    np.save('npy/acc_{}.npy'.format(iter_num), acc_ls)
    np.save('npy/loss_{}.npy'.format(iter_num), loss_ls)

# plot the outcome with saved npys.
def plot_all(opt_name):
    for opt in opt_name:
        ACC = np.load('npy/{}_acc.npy'.format(opt))
        LOSS = np.load('npy/{}_loss.npy'.format(opt))
        r1 = plt.figure(1)
        plt.plot(LOSS, label=opt)
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        r2 = plt.figure(2)
        plt.plot(ACC, label=opt)
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
    r1.savefig('fig/all_loss_{}.png'.format(len(ACC)))
    r2.savefig('fig/all_acc_{}.png'.format(len(ACC)))


if __name__ == "__main__":
    all_optimizers(opt_name, opt_ls, batch_size=50, iter_num=100)
    plot_all(opt_name)
