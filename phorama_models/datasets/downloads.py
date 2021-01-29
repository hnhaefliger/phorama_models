import zipfile
import os
import requests
from tqdm import tqdm

def getAndUnzip(url, download_output, unzip_output, verbose=1, name='download'):
    response = requests.get(url, stream=True) # download file from url
    length = int(response.headers.get('content-length')) # get number of bytes

    iterable = response.iter_content(chunk_size=1024)

    if verbose == 1: # progress bar
        iterable = tqdm(iterable, desc=name, leave=True, unit='KB')
    
    with open(download_output, 'wb+') as f: # save zip file
        for chunk in iterable:
            f.write(chunk)

    zip_file = zipfile.ZipFile(download_output, 'r') # open zip file

    iterable = zip_file.infolist() # get list of zipped contents

    if verbose == 1: # progress bar
        iterable = tqdm(iterable, desc=name, leave=True, unit='KB')

    for file in iterable: # extract files
        zip_file.extract(file)

    os.remove(download_output) # remove zip file
