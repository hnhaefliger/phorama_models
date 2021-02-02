import zipfile
import tarfile
import os
import requests
from tqdm import tqdm

def getAndExtract(url, download_output, extract_output, verbose=1, name='download'):
    response = requests.get(url, stream=True) # download file from url
    length = int(response.headers.get('content-length')) # get number of bytes

    iterable = response.iter_content(chunk_size=1024)

    if verbose == 1: # progress bar
        total = length // 1024
        
        if length % 1024 != 0:
            total += 1
            
        iterable = tqdm(iterable, desc=name + ' (download)', leave=True, unit='KB', total=total)

    f = open(download_output, 'wb+') # save zip file

    for chunk in iterable:
        f.write(chunk)

    f.close()

    if '.zip' in download_output:
        compressed_file = zipfile.ZipFile(download_output, 'r') # open zip file
        iterable = compressed_file.infolist() # get list of zipped contents

    elif '.tar' in download_output:
        if '.gz' in download_output:
            mode = 'r:gz'
          
        elif '.bz2' in download_output:
            mode = 'r:bz2'
            
        elif '.xz' in download_output:
            model = 'r:xz'
            
        else:
            mode = 'r'
        
        compressed_file = tarfile.open(download_output, mode) # open tar file
        iterable = compressed_file.getmembers() # get list of tar contents

    else:
        raise Exception('Unsupported compression type')

    if verbose == 1: # progress bar
        iterable = tqdm(iterable, desc=name + ' (extract)', leave=True, unit=' Files')

    for file in iterable: # extract files
        compressed_file.extract(file, path=extract_output)

    os.remove(download_output) # remove zip file
