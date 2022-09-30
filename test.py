import os
print(os.getcwd())

# unzipping downloaded files
# import shutil
# for y in range(2009,2022):
#     for x in range(1,5):
#         shutil.unpack_archive(f'{y}q{x}.zip', f'{y}q{x}')

# returns a map from tickers to cik
def gettickertocik():
    tickertocik = {}
    with open('ticker.txt','r') as file:
        for row in file:
            s = row.split('\t')
            tickertocik[s[0].upper().replace('-','/')] = s[1].strip()
    return tickertocik

# returns a list of tickers that edgar has a record of
def gettickers():
    tickers = []
    tickertocik = gettickertocik()
    with open('stocks.txt','r') as reader:
        for row in reader:
            ticker = row.strip()
            if ticker in tickertocik:
                tickers.append(ticker)
    return tickers

# returns a set of ciks of the tickers
def getcik():
    cik = set()
    tickers = gettickers()
    tickertocik = gettickertocik()
    for ticker in tickers:
        cik.add(tickertocik[ticker])
    return cik

# returns the adsh with specific filetypes
def getadsh(foldername, filetypes):
    adsh = set()
    filename = foldername + '/sub.txt'
    with open(filename, 'r') as file:
        for row in file:
            data = row.split('\t')
            # the 26th entry has the submission's filetype
            if data[25] in filetypes:
                adsh.add(data[0])
    return adsh

# returns the standard tags
def gettags(foldername):
    tags = set()
    filename = foldername + '/tag.txt'
    with open(filename, 'r') as file:
        for row in file:
            data = row.split('\t')
            # the 3rd entry is '0' for standard tags and '1' for custom tags
            if data[2] == '0':
                tags.add(data[0])
    return tags

# returns the subset of tags used by submissions in adsh
def getusedtags(foldername, tags, adsh):
    usedtags = set()
    filename = foldername + '/num.txt'
    with open(filename, 'r') as file:
        for row in file:
            data = row.split('\t')
            # the 1st entry is adsh and 2nd entry is tag
            if data[0] in adsh and data[1] in tags:
                usedtags.add(data[1])
    return usedtags 

import yfinance as yf
import csv
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

def download100prices():   
    tickers = gettickers()
    for ticker in tickers[:100]:
        data = yf.multi.download(ticker, start = '2009-01-01', end = date.today(), interval = '1d', keepna = True)
        data.to_csv(f'prices/{ticker}.csv')

df = pd.read_csv('prices/AAPL.csv', index_col = 0, header = 0, dtype={'Adj Close':float})
df.plot(y='Adj Close')
plt.show()