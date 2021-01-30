from . import downloads

def get(path):
    '''
    Download and unzip the LabelMe-12-50k dataset

    http://www.ais.uni-bonn.de/download/datasets.html
    '''

    if path != '' and path[-1] != '/':
        path += '/'

    downloads.getAndExtract('http://www.ais.uni-bonn.de/deep_learning/LabelMe-12-50k.tar.gz', path + 'LabelMe-12-50k.tar.gz', path, verbose=1, name='images') # part 1
