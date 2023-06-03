# Predicting stock prices
## Data
### Company filings
The data is downloaded from https://www.sec.gov/dera/data/financial-statement-data-sets.html. It contains all conpany filings from 2009. The data is organized into quarters and are downloaded individually. Each folder contains 4 files: `sub.txt`, `tag.txt`, `num.txt`, and `pre.txt`. We will not use `pre.txt`. The text files are tab delimited. Next, the relevant information for each file is explained.

###### sub.txt

Each line represents a submission to sec. The first entry is the `adsh`, a unique number that identifies the submission. The second entry is `cik`, a unique number that identifies the filing company. The 26th entry is `form`, the form name. The forms we will need are `10-K` and `10-Q`, which are annual reports and quarterly reports.

e.g. `0000796343-09-000026	796343	ADOBE SYSTEMS INC	7372	US	CA	SAN JOSE	95110-2704	345 PARK AVE		4085366000	US	CA	SAN JOSE	95110-2704	345 PARK AVENUE		US	DE	770019522			1-LAF	1	1130	10-Q	20090531	2009	Q2	20090626	2009-06-25 21:17:00.0	0	0	adbe-20090529.xml	1	`

###### tag.txt
Each line represents a tag. A tag is what a number represents. For example, if a company's net income was 1 million, the tag would be net income. The first entry in each line is the `tag`. The third entry is `custom`, which is 1 if the tag is custom, and 0 if the tag is standard. We will only consider standard tags.

e.g. `AccountsPayable	us-gaap/2008	0	0	monetary	I	C	Accounts Payable (Deprecated 2009-01-31)	Carrying value as of the balance sheet date of liabilities incurred (and for which invoices have typically been received) and payable to vendors for goods and services received that are used in an entity's business. For classified balance sheets, used to reflect the current portion of the liabilities (due within one year or within the normal operating cycle if longer); for unclassified balance sheets, used to reflect the total liabilities (regardless of due date).`

###### num.txt
Each line represents a number in a filing. If a filing has 100 numbers, each number would correspond to a line in this file. The first entry is the `adsh` of the file that the number is in, the second entry is the number's `tag`, and the 8th entry is the number's `value`.

e.g. `0000891618-09-000166	AccountsPayable	us-gaap/2008		20081231	0	USD	16580000000.0000	`

### Downloading data
The `download.py` script downloads all financial statements from 2009 to 2021. The `foldernames` variable can be modified to only download the desired quarters. The files are unzipped and changed to `.csv`.

### Stock data
The data is downloaded from yahoo using the `yfinance` library. The data is downloaded as a `DataFrame`, then converted to a csv and stored in a `prices` file.

## Running the scripts
The scripts should be run in the following order:

`download.py`

`variables.py`

`tables.py`

`regression.py`

`lstm.py`

`lstm_with_sheet.py`

`analysis.py`
