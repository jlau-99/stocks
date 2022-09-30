import yfinance as yf
import csv
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import numpy
import shutil
import os
print(os.getcwd())

# unzipping downloaded files
# for y in range(2009,2013):
#     for x in range(1,2):
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
def getadsh(foldername, filetypes = ['10-K','10-Q']):
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
    with open(f'{foldername}/tag.txt', 'r') as file:
        for row in file:
            data = row.split('\t')
            # the 3rd entry is '0' for standard tags and '1' for custom tags
            if data[2] == '0':
                tags.add(data[0])
    return tags

# returns the subset of tags used by submissions in adsh
def getusedtags(foldername, tags = None, adsh = None):
    if tags == None:
        tags = gettags(foldername)
    if adsh == None:
        adsh = getadsh(foldername)
    usedtags = set()
    with open(f'{foldername}/num.txt', 'r') as file:
        for row in file:
            data = row.split('\t')
            # the 1st entry is adsh and 2nd entry is tag
            if data[0] in adsh and data[1] in tags:
                usedtags.add(data[1])
    return usedtags 

# returns a map from the given adshs to their ciks
def getadshtocik(foldername, adsh):
    adshtocik = {}
    with open(f'{foldername}/sub.txt', 'r') as file:
        for row in file:
            data = row.split('\t')
            # the 1st entry is adsh and 2nd entry is cik
            if data[0] in adsh:
                adshtocik[data[0]] = data[1]
    return adshtocik


def generatecsv(foldername, tags, adshs):
    adshtovalues = {}
    for adsh in adshs:
        adshtovalues[adsh] = {}
    with open(f'{foldername}/num.txt', 'r') as file:
            for row in file:
                s = row.split('\t')
                if s[0] in adsh and s[1] in tags:
                    if s[1] in adshtovalues[s[0]] and adshtovalues[s[0]][s[1]][1] < s[4] and s[5] == 1:
                        adshtovalues[s[0]][s[1]] = (s[7],s[4])
    for adsh, values in adshtovalues.items():
        for tag in values.keys():
            values[tag] = values[tag][0]
        with open(f'{foldername}/data.csv', 'w', newline = '') as file:
            writer = csv.Dictwriter(file)
            writer.writeheader()
            for x,y in adshtovalues.items():
                # row does not contain adsh!!!
                writer.writerow(y)


# adshs = getadsh('test')
# tags = getusedtags('test', adsh = adshs)
# adshtovalues = {}
# for adsh in adshs:
#     adshtovalues[adsh] = {}
# with open('test/num.txt', 'r') as file:
#         for row in file:
#             s = row.split('\t')
#             if s[0] in adsh and s[1] in tags:
#                 if s[1] in adshtovalues[s[0]] and adshtovalues[s[0]][s[1]][1] < s[4] and s[5] == 1:
#                     adshtovalues[s[0]][s[1]] = (s[7],s[4])
# for adsh, values in adshtovalues.items():
#     for tag in values.keys():
#         values[tag] = values[tag][0]
        
# 'NetIncomeLoss'
# with open('test/data.csv', 'w', newline = '') as file:
#         writer = csv.DictWriter(file, fieldnames = tags)
#         writer.writeheader()
#         print(adsh)
#         for x,y in adshtovalues.items():
#             print(y)
#             writer.writerow(y)





def download100prices():   
    tickers = gettickers()
    for ticker in tickers[:100]:
        data = yf.multi.download(ticker, start = '2009-01-01', end = date.today(), interval = '1d', keepna = True)
        data.to_csv(f'prices/{ticker}.csv')

def plotweeklyvariance(ticker):
    df = pd.read_csv(f'prices/{ticker}.csv', index_col = 0, header = 0, dtype={'Adj Close':float})
    n = df.shape[0]-1
    arr = [0]*(n//5)
    for x in range(1,n-5,5):
        low = min([float(df.iloc[[i]]['Low']) for i in range(x, x+5)])
        high = max([float(df.iloc[[i]]['High']) for i in range(x, x+5)])
        diff = (high-low)/low
        if diff > 0.25:
            print(df.iloc[[x]])
        arr[x//5] = diff
    plt.hist(arr, bins=50)
    plt.show()

