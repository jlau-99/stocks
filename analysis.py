from matplotlib import pyplot as plt
from tqdm import tqdm

tickers = []
with open('tickersfinal2.txt', 'r') as file:
    for row in file:
        tickers.append(row.strip())
print(len(tickers), 'lentickers')

data = []
c = 0
mseauto = []
mselstm = []
msediff = []
auto_dict = {}
lstm_dict = {}
with open('auto_results2.txt', 'r') as file:
    for s in file:
        arr = s.strip().split('\t')
        if arr[0] in tickers:
            auto_dict[arr[0]] = float(arr[1])

with open('lstm_results2.txt', 'r') as file:
    for s in file:
        arr = s.strip().split('\t')
        if arr[0] in tickers:
            lstm_dict[arr[0]] = float(arr[1])


sheet_dict = {}
with open('lstm_with_sheet4.txt', 'r') as file:
    for s in file:
        arr = s.strip().split('\t')
        if arr[0] in tickers:
            mselstm.append(float(arr[1]))
            sheet_dict[arr[0]] = float(arr[1])

diffs = []
for k, v in auto_dict.items():
    diffs.append(v-lstm_dict[k])
count = 0
for v in diffs:
    if v > 0:
        count += 1
print('auto mse > lstm mse for ',count,'/', len(diffs), ' ', count/len(diffs))

diffs = []
for k, v in sheet_dict.items():
    diffs.append(v-lstm_dict[k])
count = 0
for v in diffs:
    if v > 0:
        count += 1
print('sheet mse > lstm mse for ',count,'/', len(diffs), ' ', count/len(diffs))

diffs = []
for k, v in sheet_dict.items():
    diffs.append(auto_dict[k]-v)
count = 0
for v in diffs:
    if v > 0:
        count += 1
print('auto mse > sheet mse for ',count,'/', len(diffs), ' ', count/len(diffs))
