from . import downloads

def get(path):
    '''
    Download and unzip the COCO 2017 dataset

    https://cocodataset.org/#home
    '''

    if path != '' and path[-1] != '/':
        path += '/'

    # downloads are in this order to minimize disk usage at any given time
    downloads.getAndExtract('http://images.cocodataset.org/zips/train2017.zip', path + 'train2017.zip', path + 'train2017', verbose=1, name='training data') # training set
    downloads.getAndExtract('http://images.cocodataset.org/zips/test2017.zip', path + 'test2017.zip', path + 'test2017', verbose=1, name='testing data') # testing set
    downloads.getAndExtract('http://images.cocodataset.org/zips/val2017.zip', path + 'val2017.zip', path + 'val2017', verbose=1, name='validation data') # validation set
