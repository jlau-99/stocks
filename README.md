# Predicting stock prices
## Data
### Company filings
The data is downloaded from https://www.sec.gov/dera/data/financial-statement-data-sets.html. It contains all conpany filings from 2009. The data is organized into quarters and are downloaded individually. Each folder contains 4 files: sub.txt, tag.txt, num.txt, and pre.txt. We will not use pre.txt. The text files are tab delimited. Next, the relevant information for each file is explained.

###### sub.txt

Each line represents a submission to sec. The first entry is the adsh, a unique number that identifies the submission. The second entry is cik, a unique number that identifies the filing company. The 26th entry is form, the form name. The forms we will need are 10-K and 10-Q, which are annual reports and quarterly reports.

###### tag.txt
Each line represents a tag. A tag is what a number represents. For example, if a company's net income was 1 million, the tag would be net income. The first entry in each line is the tag. The third entry is custom, which is 1 if the tag is custom, and 0 if the tag is standard. We will only consider standard tags.

###### num.txt
Each line represents a number in a filing. If a filing has 100 numbers, each number would correspond to a line in this file. The first entry is the adsh of the file that the number is in, the second entry is the number's tag, and the 8th entry is the number's value.

### Stock data
The data is downloaded from yahoo using the yfinance library.