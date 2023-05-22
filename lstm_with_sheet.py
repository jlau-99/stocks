import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, exp
import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
import wandb
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import random
from time import time
from collections import defaultdict

# wandb.init(project='stocks', entity='jlau',
#     config = {
#         "epochs" :  100,
#         "hidden size" : 20
#     })

def phi(x, c):
    if x > c:
        return x/c-1
    else:
        return -c/x+1

def psi(r, c):
    if r > 0:
        return c*(r+1)
    else:
        return c/(1-r)

starttime = time()




def pad(arr, n):
    if len(arr) >= n:
        return arr[len(arr)-n:]
    a = []
    for i in range(n - len(arr)):
        a.append(1)
    a.extend(arr)
    return a

def getnumvars(ticker):
    filename = 'data/' + ticker + '/' + ticker + '.txt'
    with open(filename, 'r') as file:
        header = file.readline()
        freqs = [0] * len(header)
        for row in file:
            vals = row.strip().split('\t')
            for i in range(1, len(vals)):
                if vals[i] != '':
                    freqs[i-1] += 1
        i = 0
        while i + 1< len(freqs) and freqs[i] == freqs[i+1]:
            i += 1
        return i


def sigmoid(x):
    return 1/(1+math.exp(-x))

# scales the matrix's ith column
def scale(mat, j):
    max = 0
    for i in range(len(mat)):
        if abs(mat[i][j]) > max:
            max = abs(mat[i][j])
    if max != 0:
        for i in range(len(mat)):
            mat[i][j] = mat[i][j]/max

def get_random_pair(start, end, width):
    a = random.randint(start, end-width)
    return (a, a + width)

class LSTM1(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers,
                 dropout_prob):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  #number of classes
        self.num_layers = num_layers  #number of layers
        self.input_size = input_size  #input size
        self.hidden_size = hidden_size  #hidden state
        self.dropout_prob = dropout_prob  #dropout

        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob)  #lstm
        self.fc = nn.Linear(hidden_size,
                            num_classes)  #fully connected last layer

    def forward(self, x):
        x = self.dropout(x)
        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0),
                        self.hidden_size))  #hidden state
        c_0 = Variable(
            torch.zeros(self.num_layers, x.size(0),
                        self.hidden_size))  #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x, (h_0, c_0))  #lstm with input, hidden, and internal state
        out = self.fc(output[0])  #Final Output
        return out
    
def lstm_with_sheet(ticker):
    path = 'data/' + ticker + '/' + ticker + '.txt'
    times = []
    with open(path, 'r') as f:
        f.readline()
        for row in f:
            date = row.split('\t')[0]
            times.append(date)

    df = {}
    filename = 'prices/' + ticker + 'imputed.txt'
    with open(filename, 'r') as file:
        for row in file:
            d = row.strip().split('\t')
            df[d[0]] = float(d[1])

    arr = [[]]

    ind = 0
    period = times[0]
    prev = 1
    lastQ = 1
    target = []

    for key, val in df.items():
        if ind == 0:
            lastQ = prev
        s = key[:4]+key[5:7]+key[8:]
        if s > period:
            ind += 1
            if ind >= len(times):
                break
            period = times[ind]
            arr.append([])
            target.append(phi(prev, lastQ))
            lastQ = prev
        arr[-1].append(phi(val,prev))
        prev = val

    newarr = []
    for i in range(len(arr)):
        newarr.append(pad(arr[i], 90))
    arr = newarr
    # print(len(newarr))
    # print(len(times))

    mat = []
    header = []
    d = {}
    numvars = 30
    freqs = []
    numvars = getnumvars(ticker)
    if numvars < 30:
        numvars = 30

    with open(path, 'r') as f:
        header = f.readline().strip().split('\t')
        for s in header:
            d[s.strip()] = 0
            if len(d) >= numvars + 1:
                break
        for row in f:
            data = row.split('\t')
            mat.append([])
            for i in range(1, numvars + 1):
                if data[i].strip() != "":
                    val = float(data[i].strip())
                    # if val > 10000:
                    #     val = val / 100000
                    mat[-1].append(val)
                    d[header[i-1]] += 1
                else:
                    mat[-1].append(np.nan)

    for i in range(numvars):
        scale(mat, i)
    # print(mat)
    # print(d)
    imp = IterativeImputer(max_iter=200)
    # print('imputing...')
    mat = imp.fit_transform(mat)
    vals = defaultdict(int)
    max = (0, 0, 0)

    for j in range(len(mat[0])):
        for i in range(len(mat)-1, 0, -1):
            if mat[i-1][j] == 0 or mat[i][j] == 0:
                if mat[i-1][j] == 0 and mat[i][j] == 0:
                    mat[i][j] = 0.5
                elif mat[i][j] > mat[i-1][j]:
                    mat[i][j] = 1
                else:
                    mat[i][j] = 0
            elif abs(phi(mat[i][j], mat[i-1][j]))>10:
                if phi(mat[i][j], mat[i-1][j]) > 0:
                    mat[i][j] = 1
                else:
                    mat[i][j] = 0
            else:
                mat[i][j] = sigmoid(phi(mat[i][j], mat[i-1][j]))
            # print(mat[i][j])
        mat[0][j] = 1

    dataset = arr
    for i in range(len(dataset)):
        dataset[i].extend(mat[i])

    # target = []
    # for i in range(len(dataset)-1):
    #     target.append(dataset[i+1][59]/dataset[i][59])
    dataset = dataset[1:-1]
    target = target[1:]
    # print(target)
    # print(dataset[0])

    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    xtrain, xtest = dataset[:train_size], dataset[train_size:]
    ytrain, ytest = target[:train_size], target[train_size:]

    xtrain = np.array(xtrain)
    xtest = np.array(xtest)
    ytrain = np.array(ytrain)
    ytest = np.array(ytest)

    X_train_tensors = Variable(torch.Tensor(xtrain))
    X_test_tensors = Variable(torch.Tensor(xtest))

    y_train_tensors = Variable(torch.Tensor(ytrain))
    y_test_tensors = Variable(torch.Tensor(ytest))

    X_train_tensors_final = torch.reshape(
        X_train_tensors, (1, X_train_tensors.shape[0], X_train_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(
        X_test_tensors, (1, X_test_tensors.shape[0], X_test_tensors.shape[1]))
    # print(X_train_tensors_final[0,0,:])
    # print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
    # print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

    num_epochs = 1000  #epochs
    learning_rate = 0.001  #lr

    input_size = 90 + numvars  #number of features
    hidden_size = 50  #number of features in hidden state
    num_layers = 2  #number of stacked lstm layers
    dropout_prob = 0.2

    num_classes = 1  #number of output classes
    lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers,
                  dropout_prob)  #our lstm class

    # summary(lstm1)

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)#, weight_decay=0.001)

    for epoch in tqdm(range(num_epochs)):
        width = random.randint(int(train_size/4), X_train_tensors_final.shape[1])
        a, b = get_random_pair(0, X_train_tensors_final.shape[1], width)
        outputs = lstm1.forward(X_train_tensors_final[:,a:b,:])  #forward pass
        optimizer.zero_grad()  #caluclate the gradient, manually setting to 0

        # obtain the loss function
        outputs = torch.reshape(outputs, [outputs.shape[0]])
        loss = criterion(outputs, y_train_tensors[a:b])
        loss.backward()  #calculates the loss of the loss function

        optimizer.step()  #improve from loss, i.e backprop
        # if epoch % 200 == 0:
        #     print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        #     # print(width, a, b)

        # # wandb.log({"loss" :  loss.item()})

    path = "models/" + ticker + "_withsheet.pth"
    torch.save(lstm1.state_dict(), path)

    # path = "models/" + ticker + ".pth"
    # lstm1.load_state_dict(torch.load(path))
    # lstm1.eval()

    dataset = Variable(torch.Tensor(dataset))  #converting to Tensors
    #reshaping the dataset
    dataset = torch.reshape(dataset, (1, dataset.shape[0], dataset.shape[1]))

    def mse(y_true, y_pred):
        n = len(y_true)
        tot = 0
        for j in range(n):
            tot += (y_true[j] - y_pred[j])**2
        return tot / n

    predict = lstm1(dataset).detach().numpy().reshape(dataset.shape[1])
    train_predict = predict[:train_size]
    test_predict = predict[train_size:]
    print(train_predict.shape, test_predict.shape)
    print("mse predict, y", mse(train_predict, ytrain), mse(test_predict, ytest))
    print("variances", np.var(ytrain), np.var(ytest))

    return mse(test_predict, ytest)

    # plt.plot(test_predict, label='prediction')
    # plt.plot(ytest, label='data')
    # plt.legend()
    # plt.savefig('lstmwithsheet.png')
    # plt.close()

tickers = []
with open('tickersfinal2.txt', 'r') as file:
    for row in file:
        tickers.append(row.split()[0])
count = 0

for ticker in tickers:
    count += 1
    # if count > 100:
    #     break
    print('At ticker ', count, ', ', ticker)
    try:
        m = lstm_with_sheet(ticker)
        with open('lstm_with_sheet4.txt', 'a') as file:
            row = ticker + '\t' + str(m) + '\n'
            file.write(row)
    except:
        with open('lstm_with_sheet_failed4.txt', 'a') as file:
            file.write(ticker + '\n')


