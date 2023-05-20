from tqdm import tqdm
import yfinance as yf
from datetime import date
import os
from collections import defaultdict
from matplotlib import pyplot as plt

# number of stocks a company issues
def table_numstocks():
    ciktoticker = {}
    with open('ticker.txt','r') as file:
        for row in file:
            s = row.strip().split('\t')
            if s[1].strip() not in ciktoticker:
                ciktoticker[s[1].strip()] = 0
            ciktoticker[s[1].strip()] += 1
            if ciktoticker[s[1].strip()] > 30:
                print(s[0])
    freq = [0] * 5
    for key, val in ciktoticker.items():
        if val > 10:
            print(val)
        if val > 4:
            val = 4
        freq[val] += 1
    print(freq)

    # # returns an array of length 400, index i indicating the number of stocks such that 
    # # i is the number of variables such that the frequency that variable is nonempty is greater than frac
def table_60percent():
    tickers = []
    with open('tickersfinal.txt', 'r') as file:
        for row in file:
            tickers.append(row.split()[0])
    varcounter = [0] * 400
    varcounterall = [0] * 400
    for ticker in tqdm(tickers):
        dates = []
        path = 'data/' + ticker + '/' + ticker + '.txt'
        tagcounter = []
        with open(path, 'r') as file:
            head = file.readline().split('\t')
            if ticker == 'AAPL':
                print('apple has ', len(head), ' variables')
            tagcounter = [0] * len(head)
            for row in file:
                data = row.split('\t')
                dates.append(data[0])
                for j in range(1, len(data)):
                    if data[j].strip() != '':
                        tagcounter[j-1] += 1
        # print(tagcounter)
        count = 0
        countall = 0
        # print(len(dates))
        for val in tagcounter:
            if val/len(dates) < 0.6:
                break
            count += 1
        for val in tagcounter:
            if val < len(dates):
                break
            countall += 1
        if countall > 45:
            print(ticker, ' with ', countall, ' variables')
        if ticker == 'AAPL':
            print(count, '60 percent for apple')
            print(countall, 'all variables for apple')
        # if count < 10:
        #     print(ticker)
        #     print(tagcounter)
        # print(ticker, count, )
        varcounter[count] += 1
        varcounterall[countall] += 1

    arr = []
    arrall = []
    for i in range(0, 400, 10):
        arr.append(sum(varcounter[i:i+10]))
        arrall.append(sum(varcounterall[i:i+10]))
    print(arr[:4], sum(arr[4:]))
    print(arrall[:4], sum(arrall[4:]))

def nextq(s):
    year = s[:4]
    month = s[4:6]
    day = [-1, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31]
    day = day[int(month)]
    if int(year)%4==3 and int(month)==11:
        day += 1
    month = (int(month) + 2) % 12 + 1
    if month <= 3:
        year = int(year) + 1
    if month < 10:
        month = '0' + str(month)
    return str(year) + str(month) + str(day)

# creates the final version of the tickers to be analyzed
def create_tickersfinal2():
    freq = [0] * 52
    tickerswithsheets = []
    missingsheetscounter = [0] * 50
    tickers = []
    with open('tickersfinal.txt', 'r') as file:
        for row in file:
            tickers.append(row.split('\t')[0])
    for ticker in tqdm(tickers):
        missingsheets = 0
        count = 0
        path = 'data/' + ticker + '/' + ticker + '.txt'
        dates = []
        yearcount = defaultdict(int)
        with open(path, 'r') as file:
            file.readline()
            for row in file:
                dates.append(row[:8])
        for i in range(len(dates)-1):
            if nextq(dates[i]) != dates[i+1]:
                count += 1
        freq[count] += 1
        # if count > 4:
        #     print(ticker)
        # print(dates)
        for d in dates:
            yearcount[d[:4]] += 1
        startyear = dates[0][:4]
        endyear = dates[-1][:4]
        for year, val in yearcount.items():
            # if val > 4:
            #     print(ticker, year)
            if startyear == year or endyear == year:
                continue
            if 4 != val:
                missingsheets += 1
        missingsheetscounter[missingsheets] += 1
        if missingsheets == 0:
            tickerswithsheets.append(ticker)
        # if missingsheets > 4:
        #     print(ticker, missingsheets)
        # print(dates)

    print('missing sheet counter: ', missingsheetscounter)

    # print(freq[:3].append(sum(freq[3:])))

    with open('tickerswithsheets.txt', 'w') as file:
        for ticker in tickerswithsheets:
            file.write(ticker)
            file.write('\n')

    tickerswithprice = []
    for ticker in tickerswithsheets:
        filename = 'prices/' + ticker + 'imputed.txt'
        with open(filename, 'r') as file:
            row = file.readline()
            if row.split('\t')[0] < '2016-01-01' and row != '':
                tickerswithprice.append(ticker)

    with open('tickersfinal2.txt', 'w') as file:
        for ticker in tickerswithprice:
            file.write(ticker + '\n')

table_60percent()

# # create_tickersfinal2()
# tickers = []
# with open('tickersfinal2.txt', 'r') as file:
#     for row in file:
#         tickers.append(row.strip())
# print(len(tickers), 'lentickers')

# data = []
# c = 0
# with open('lstm_results.txt', 'r') as file:
#     for s in file:
#         arr = s.strip().split('\t')
#         if arr[0] in tickers:
#             if float(arr[1]) > 0.3:
#                 c += 1
#             else:
#                 data.append(float(arr[1]))

