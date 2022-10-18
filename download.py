import requests
import shutil
import os

foldernames = [f'{year}q{q}' for year in range(2009, 2022) for q in range(1, 5)]

# download everything from 2009 to 2021
for foldername in foldernames:
    url = f'https://www.sec.gov/files/dera/data/financial-statement-data-sets/{foldername}.zip'
    response = requests.get(url, stream=True)
    with open(f'{foldername}.zip', "wb") as f:
        for chunk in response.iter_content(chunk_size=512):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# unzipping downloaded files
for foldername in foldernames:
    shutil.unpack_archive(f'{foldername}.zip', foldername)

# change from txt to csv
for foldername in foldernames:
    for filetype in ['sub', 'tag', 'num']:
            os.rename(f'{foldername}/{filetype}.txt',f'{foldername}/{filetype}.csv')