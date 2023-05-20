import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, exp
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
import wandb
from tqdm import tqdm
import os
import random
import regression
from time import time
starttime = time()

def mse(y_true, y_pred):
    n = len(y_true)
    sum = 0
    for i in range(n):
        sum += (y_pred[i] - y_true[i]) * (y_pred[i] - y_true[i])
    return sum / n

class LSTM1(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers,
                 dropout_prob):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)  #lstm
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
    
def get_random_pair(start, end, width):
    a = random.randint(start, end-width)
    return (a, a + width)

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
        
def train(ticker, details=False):
    filename = 'prices/' + ticker + 'imputed.txt'
    prices = []
    with open(filename, 'r') as file:
        for row in file:
            d = row.strip().split('\t')
            prices.append(float(d[1]))

    data = prices
    forward = 90
    xvals = []
    target = []
    # for i in range(30, len(data)- forward, 30):
    #         xvals.append(phi(data[i], data[i-30]))
    #         target.append(phi(data[i+forward], data[i]))

    # for i in range(1, len(data) - forward):
    #     xvals.append(phi(data[i], data[i-1]))
    #     target.append(phi(data[i+forward], data[i]))

    for i in range(30, len(data)- forward, 30):
        temp = []
        for j in range(30):
            temp.append(phi(data[i-j], data[i-j-1]))
        xvals.append(temp)
        target.append(phi(data[i+forward], data[i]))

    trainsize = int(len(target) * 0.67)
    xtrain = np.array(xvals[:trainsize])
    xtest = np.array(xvals[trainsize:])
    ytrain = np.array(target[:trainsize])
    ytest = np.array(target[trainsize:])

    X_train_tensors = Variable(torch.Tensor(xtrain))
    X_test_tensors = Variable(torch.Tensor(xtest))

    y_train_tensors = Variable(torch.Tensor(ytrain))
    y_test_tensors = Variable(torch.Tensor(ytest))

    X_train_tensors = torch.reshape(
        X_train_tensors, (1, X_train_tensors.shape[0], X_train_tensors.shape[1]))
    X_test_tensors = torch.reshape(
        X_test_tensors, (1, X_test_tensors.shape[0], X_test_tensors.shape[1]))
    y_train_tensors = torch.reshape(
        y_train_tensors, (y_train_tensors.shape[0], 1))

    # print("Training Shape", X_train_tensors.shape, y_train_tensors.shape)
    # print("Testing Shape", X_test_tensors.shape, y_test_tensors.shape)

    num_epochs = 500  #epochs
    learning_rate = 0.001  #lr

    input_size = 30  #number of features
    hidden_size = 50  #number of features in hidden state
    num_layers = 2  #number of stacked lstm layers
    dropout_prob = 0.2

    num_classes = 1  #number of output classes
    lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, dropout_prob)  #our lstm class

    # summary(lstm1)

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate, weight_decay=0.01)


    for epoch in tqdm(range(num_epochs)):
        width = random.randint(int(trainsize/4),  X_train_tensors.shape[1])
        a, b = get_random_pair(0, X_train_tensors.shape[1], width)
        outputs = lstm1.forward(X_train_tensors[:,a:b,:])  #forward pass
        optimizer.zero_grad()  #caluclate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train_tensors[a:b,:])

        loss.backward()  #calculates the loss of the loss function

        optimizer.step()  #improve from loss, i.e backprop
        # if epoch % 20 == 0:
        #     print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    #   wandb.log({"loss" :  loss.item()})
    path = 'models/' + ticker + '_lstm.pth'
    if not os.path.exists('models'):
        os.mkdir('models')

    torch.save(lstm1.state_dict(), path)

    # lstm1.load_state_dict(torch.load(path))
    # lstm1.eval()

    dataset = np.array(xvals)
    dataset = Variable(torch.Tensor(dataset))  #converting to Tensors
    #reshaping the dataset
    dataset = torch.reshape(dataset, (1, dataset.shape[0], dataset.shape[1]))

    predict = lstm1(dataset).detach().numpy().reshape(dataset.shape[1], 1)
    train_predict = predict[:trainsize]
    test_predict = predict[trainsize:]
    meansquareerror = mse(test_predict, ytest)

    with open('lstm_results2.txt', 'a') as file:
        row = ticker + '\t' + str(meansquareerror[0]) + '\n'
        file.write(row)
    
    if details:
        plt.plot(test_predict, label='predictions')
        plt.plot(ytest, label='data')
        plt.legend()
        plt.savefig('lstmratio2.png')
        plt.close()
        predictions = predict
        predict = []
        for val in predictions:
            print(val)
        actualprice = []
        for i in range(len(ytest)):
            predict.append(psi(predictions[i], data[(i+trainsize)*30]))
            actualprice.append(data[(i+trainsize)*30+forward])
        plt.plot(actualprice, label='price')
        plt.plot(predict, label='prediction')
        plt.legend()
        plt.savefig('lstmprices2.png')
        plt.close()



tickers = []
with open('tickersfinal2.txt', 'r') as file:
    for row in file:
        tickers.append(row.split()[0])
count = 0

for ticker in tickers:
    count += 1
    print('At ticker ', count, ', ', ticker)
    try:
        train(ticker)
    except:
        with open('lstm_failed2.txt', 'a') as file:
            file.write(ticker + '\n')
    try:
        m, v = regression.autoregression(ticker)
        with open('auto_results2.txt', 'a') as file:
            row = ticker + '\t' + str(m) + '\n'
            file.write(row)
    except:
        with open('auto_failed2.txt', 'a') as file:
            file.write(ticker + '\n')
