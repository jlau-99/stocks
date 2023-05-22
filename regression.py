from collections import defaultdict
from tqdm import tqdm
import csv
import test
import pandas as pd
from datetime import date
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math


def imputedprices(ticker):
    filename = 'prices/' + ticker + '.csv'
    df = pd.read_csv(filename,
                     index_col=0,
                     header=0,
                     dtype={'Adj Close': float})
    i = 0
    dfdict = df['Adj Close'].to_dict()
    one = datetime.timedelta(days=1)
    while i+1 < df.shape[0]:
        curr = date.fromisoformat(df.index[i])
        next = date.fromisoformat(df.index[i+1])
        if curr + one != next:
            start = df['Adj Close'][i]
            end = df['Adj Close'][i+1]
            length = (next-curr).days
            diff = end - start
            d = curr + one
            for j in range(1, length):
                dfdict[d.isoformat()] = start + diff * j / length
                d += one
        i += 1
    dfdict = dict(sorted(dfdict.items(), key=lambda pair:pair[0]))
    return dfdict

def create_imputed_prices():
    tickers = []
    with open('tickersfinal.txt', 'r') as file:
        for row in file:
            tickers.append(row.split('\t')[0])
    for ticker in tqdm(tickers):
        data = imputedprices(ticker)
        filename = 'prices/' + ticker + 'imputed.txt'
        with open(filename, 'w') as file:
            for key, val in data.items():
                file.write(key)
                file.write('\t')
                file.write(str(val))
                file.write('\n')

def msee(y_true, y_pred):
    n = len(y_true)
    sum = 0
    for i in range(n):
        sum += (y_pred[i] - y_true[i]) * (y_pred[i] - y_true[i])
    return sum / n

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

def autoregression(ticker, detail=False):
    
    data = []
    filename = 'prices/' + ticker + 'imputed.txt'
    with open(filename, 'r') as file:
        for row in file:
            num = row.strip().split('\t')[1]
            data.append(float(num))
    lag = 180
    forward = 90
    xvals = []
    target = []
    for i in range(len(data) - lag - forward - 1):
        temp = []
        for j in range(lag):
            temp.append(phi(data[i+j+1], data[i+j]))
        xvals.append(temp)
        target.append(phi(data[i+lag+forward], data[i+lag]))

    trainsize = int(len(target) * 0.67)
    xtrain = np.array(xvals[:trainsize])
    xtest = np.array(xvals[trainsize:])
    ytrain = np.array(target[:trainsize])
    ytest = np.array(target[trainsize:])
    v = np.var(ytest)
    sd = math.sqrt(v)
    mean = np.mean(ytest)
    model = LinearRegression()

    model.fit(xtrain, ytrain)

    predictions = model.predict(xtest)
    mse = mean_squared_error(ytest, predictions)
    var = np.var(ytest)

    if detail:
        plt.plot(ytest, label='data')
        plt.plot(predictions, label='predictions')
        plt.axhline(y=mean, color='g', linewidth=1)
        plt.axhline(y=mean+sd, color='g', linewidth=0.3)
        plt.axhline(y=mean-sd, color='g', linewidth=0.3)
        plt.legend()
        plt.savefig('autoratio.png')
        plt.close()

        print('mse: ', mse)
        print('mean: ', mean)
        print('sd: ', sd)
        print('variance: ', var)
        print('ones: ', mean_squared_error(np.ones(ytest.shape[0]), ytest))
        sameratio = []
        for i in range(len(ytest)):
            sameratio.append(data[i+trainsize]/data[i+trainsize-forward])
        sameratio = np.array(sameratio)
        print('sameratio: ', mean_squared_error(sameratio, ytest))
        
        predict = []
        actualprice = []
        for i in range(len(ytest)):
            predict.append(psi(predictions[i], data[i+trainsize+lag]))
            actualprice.append(data[i+trainsize+lag+forward])
        plt.plot(actualprice, label='price')
        plt.plot(predict, label='prediction')
        plt.legend()
        plt.savefig('autoprices.png')
        plt.close()
    return (mse, var)

# autoregression('AAPL', detail=True)
if __name__ == '__main__':
    tickers = []
    with open('tickersfinal2.txt', 'r') as file:
        for row in file:
            tickers.append(row.split()[0])
    count = 0
    
    for ticker in tickers:
        count += 1
        print('At ticker ', count, ', ', ticker)
        try:
            m, v = autoregression(ticker)
            with open('auto_results2.txt', 'a') as file:
                row = ticker + '\t' + str(m) + '\n'
                file.write(row)
        except:
            with open('auto_failed2.txt', 'a') as file:
                file.write(ticker + '\n')
