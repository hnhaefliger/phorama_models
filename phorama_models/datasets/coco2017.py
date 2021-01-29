from . import downloads

def get(path):
    '''
    Download and unzip the COCO 2017 dataset
    '''

    if path != '' and path[-1] != '/':
        path += '/'

    # downloads are in this order to minimize disk usage at any given time
    downloads.getAndUnzip('http://images.cocodataset.org/zips/train2017.zip', path + 'train2017.zip', path, verbose=1, name='training data') # training set
    downloads.getAndUnzip('http://images.cocodataset.org/zips/test2017.zip', path + 'test2017.zip', path, verbose=1, name='testing data') # testing set
    downloads.getAndUnzip('http://images.cocodataset.org/zips/val2017.zip', path + 'val2017.zip', path, verbose=1, name='validation data') # validation set
