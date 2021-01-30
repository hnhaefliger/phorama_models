from . import downloads

def get(path):
    '''
    Download and unzip the Indoor Scene Recognition dataset

    http://web.mit.edu/torralba/www/indoor.html
    '''

    if path != '' and path[-1] != '/':
        path += '/'

    downloads.getAndExtract('http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar', path + 'indoorCVPR_09.tar', path, verbose=1, name='images') # part 1
