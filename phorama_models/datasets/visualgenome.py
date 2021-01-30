from . import downloads

def get(path):
    '''
    Download and unzip the Visual Genome 2017 dataset

    http://visualgenome.org/api/v0/api_home.html
    '''

    if path != '' and path[-1] != '/':
        path += '/'

    downloads.getAndExtract('https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip', path + 'images.zip', path, verbose=1, name='part 1') # part 1
    downloads.getAndExtract('https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip', path + 'images2.zip', path, verbose=1, name='part 2') # part 2
