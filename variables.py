import os
import test
from collections import defaultdict
from tqdm import tqdm
import yfinance as yf
from datetime import date

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists('data'):
    os.mkdir('data')

# returns a map from tickers to cik
def gettickertocik():
    tickertocik = {}
    with open('ticker.txt','r') as file:
        for row in file:
            s = row.split('\t')
            tickertocik[s[0].upper().replace('-','/')] = s[1].strip()
    tickertocik = dict(sorted(tickertocik.items(), key=lambda pair:pair[0]))
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

tickertocik = gettickertocik()
tickers = gettickers()
ciks = getcik()
ciktoticker = {}
for key, val in tickertocik.items():
    if val in ciktoticker:
        continue
    ciktoticker[val] = key 


foldernames = [f'{year}q{q}' for year in range(2009, 2022) for q in range(1, 5)]

adshs = defaultdict(dict)
print('going through subs...')
for i in tqdm(range(len(foldernames))):
    filename = foldernames[i] + '/sub.txt'
    with open(filename, 'r') as file:
        for row in file:
            data = row.split('\t')
            # index 25 has the submission's filetype
            # index 26 has the date
            if data[1] in ciks and data[25] in ['10-K','10-Q']:
                ticker = ciktoticker[data[1]]
                adshs[ticker][data[0]] = data[26]

tickersfinal = []
for ticker , val in adshs.items():
    if len(val) > 24:
        tickersfinal.append(ticker)
adshsfinal = {}
for ticker, val in adshs.items():
    if ticker in tickersfinal:
        adshsfinal[ticker] = val
adshs = adshsfinal
tickers = tickersfinal
print(len(tickers))

adshtoticker = {}
for ticker, adsh in adshs.items():
    for key, val in adsh.items():
        adshtoticker[key] = ticker


# creates files of that contain rows of the form "20200630\tAssets\t123456.789\tLiabilities\t2345.67..."
for i in tqdm(range(len(foldernames))):
    foldername = foldernames[i]
    tickerdatetag = {}
    filename = foldername + '/num.txt'
    with open(filename, 'r') as file:
        for row in file:
            data = row.split('\t')
            ddate = data[4]
            adsh = data[0]
            tag = data[1]
            coreg = data[3]
            value = data[7].strip()
            ticker = False
            if adsh in adshtoticker:
                ticker = adshtoticker[adsh]
            if ticker and ddate == adshs[ticker][adsh] and data[5] in ['0','1'] and coreg == "":
                if ticker not in tickerdatetag:
                    tickerdatetag[ticker] = {}
                if ddate not in tickerdatetag[ticker]:
                    tickerdatetag[ticker][ddate] = {}
                tickerdatetag[ticker][ddate][tag] = value
    for ticker, datetag in tickerdatetag.items():
        path = 'data/' + ticker.replace('/','-')
        if not os.path.exists(path):
            os.mkdir(path)
        path = path + '/' + foldername + '.txt'
        with open(path, 'w') as file:
            for key, val in datetag.items():
                file.write(key + '\t')
                for key2, val2 in val.items():
                    file.write(key2 + '\t' + val2 + '\t')
                file.write('\n')


with open('tickersfinal.txt', 'w') as file:
    for ticker in tickers:
        ticker2 = ticker.replace('/', '-')
        file.write(ticker2)
        file.write('\t')
        file.write(tickertocik[ticker])
        file.write('\n')
print(len(tickers))

def is_float(s):
    try:
        float(s)
        return True
    except:
        return False
    
for i in tqdm(range(len(tickers))):
    ticker = tickers[i]
    datetagval = defaultdict(dict)
    for foldername in foldernames:
        filename = 'data/' + ticker.replace('/', '-') + '/' + foldername + '.txt'
        if not os.path.exists(filename):
            continue
        with open(filename, 'r') as file:
            for row in file:
                data = row.strip().split('\t')
                ddate = data[0]
                prev = data[1]
                i = 1
                while i < len(data):
                    while i < len(data) and not is_float(data[i]):
                        prev = data[i]
                        i += 1
                    if i >= len(data):
                        break
                    datetagval[ddate][prev] = data[i]
                    i += 1
                    
    datetagval = list(sorted(datetagval.items(), key=lambda pair: pair[0]))
    tagcount = defaultdict(int)
    for ddate, tags in datetagval:
        for key, val in tags.items():
            tagcount[key] += 1
    sortedtags = list(sorted(tagcount.items(), key=lambda pair: pair[1], reverse=True))
    sortedtags = [pair[0] for pair in sortedtags]
    data = []
    dates = []
    for ddate, tagval in datetagval:
        data.append([""] * len(sortedtags))
        dates.append(ddate)
        for k, v in tagval.items():
            data[-1][sortedtags.index(k)] = v
    path = 'data/' + ticker.replace('/','-') + '/' + ticker.replace('/','-') + '.txt'
    with open(path, 'w') as f:
        f.write('\t'.join(sortedtags))
        count = 0
        for arr in data:
            f.write('\n')
            f.write(dates[count])
            count += 1
            f.write('\t')
            f.write('\t'.join(arr))


dirs = os.scandir('prices')
downloaded = set()
for dir in dirs:
    downloaded.add(dir.name.split('.')[0])

count = 1
with open('tickersfinal.txt', 'r') as file:
    for row in file:
        print(count)
        count += 1
        ticker = row.split('\t')[0]
        if ticker in downloaded:
            continue
        data = yf.multi.download(ticker, start = '2009-01-01', end = date.today(), interval = '1d', keepna = True)
        data.to_csv(f'prices/{ticker}.csv')

