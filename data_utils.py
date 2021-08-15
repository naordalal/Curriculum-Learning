import torch
import numpy as np
import os
import pickle
import torchvision
from torchvision import transforms

def write_pickle(outfile_path, output):
    '''
    Description: Write output object to pickle file
    Input:  outfile_path - pickle file path
            output - object to write
    Output:
    '''
    with open(outfile_path, 'wb') as f:
        pickle.dump(output, f)

def read_pickle(outfile_path):
    print(outfile_path)
    '''
    Description: Read pickle and return the object
    Input: outfile_path - pickle file path
    Output: object that stored in the pickle file
    '''
    if os.path.exists(outfile_path):
        with open(outfile_path, 'rb') as f:
            return pickle.load(f)
    return None

def resize_data(data, image_size=(32,32), repeat=False):
    '''
    Description: Resize the images data
    Input:  data - data to resize
            image_size - the target data size
            repeat - indicates if to repeat the color dimension
    Output: the resized data
    '''
    transform = transforms.Compose([transforms.Resize(image_size),])                                  
    data = transform(torch.from_numpy(data))
    if repeat:
        data = data.unsqueeze_(1)
        data = data.repeat(1, 3, 1, 1)
    return data.detach().numpy()
    
def get_STL10_data(split='train'):
    '''
    Description: Create STL10 data
    Input:  split - type of data split
    Output: STL10 x,y, class_to_idx dict, epochs
    '''
    epochs = 150
    data = torchvision.datasets.STL10('STL10', download=True, split=split)
    x = data.data.astype(np.float32)
    x = resize_data(x)
    y = data.labels
    class_to_idx = {k: v for v, k in enumerate(data.classes)}
    return x, y, class_to_idx, epochs

def get_CIFAR100_data(train=True):
    '''
    Description: Create CIFAR100 data
    Input:  train - indicated if train to test data
    Output: CIFAR100 x,y, class_to_idx dict, epochs
    '''
    epochs = 150
    data = torchvision.datasets.CIFAR100('CIFAR100', download=True, train=train)
    x = np.transpose(data.data, (0, 3, 1, 2)).astype(np.float32)
    x = resize_data(x)
    y = data.targets
    class_to_idx = data.class_to_idx
    return x, y, class_to_idx, epochs

def get_CIFAR10_data(train=True):
    '''
    Description: Create CIFAR10 data
    Input:  train - indicated if train to test data
    Output: CIFAR10 x,y, class_to_idx dict, epochs
    '''
    epochs = 50
    data = torchvision.datasets.CIFAR10('CIFAR10', download=True, train=train)
    x = np.transpose(data.data, (0, 3, 1, 2)).astype(np.float32)
    x = resize_data(x)
    y = data.targets
    class_to_idx = data.class_to_idx
    return x, y, class_to_idx, epochs

def get_EMNIST_data(split='letters', train=True):
    '''
    Description: Create EMNIST data
    Input:  split - type of data split
            train - indicated if train to test data
    Output: EMNIST x,y, class_to_idx dict, epochs
    '''
    epochs = 150
    data = torchvision.datasets.EMNIST('EMNIST', split=split, download=True, train=train)
    x = data.data.detach().numpy()
    size_of_data = min(5000, len(x))
    x = x[:size_of_data]
    x = np.rot90(np.flip(x, 2), axes=(1,2)).astype(np.float32)
    x = resize_data(x, repeat=True)
    y = data.targets.detach().numpy()[:size_of_data]
    class_to_idx = data.class_to_idx
    return x, y, class_to_idx, epochs

def get_FMNIST_data(train=True):
    '''
    Description: Create Fashion-MNIST data
    Input:  train - indicated if train to test data
    Output: Fashion-MNIST x,y, class_to_idx dict, epochs
    '''
    epochs = 150
    data = torchvision.datasets.FashionMNIST("FashionMNIST", train=train, download=True)
    x = data.data.detach().numpy()
    size_of_data = min(5000, len(x))
    x = x[:size_of_data]
    x = np.rot90(np.flip(x, 2), axes=(1,2)).astype(np.float32)
    x = resize_data(x, repeat=True)
    y = data.targets.detach().numpy()[:size_of_data]
    class_to_idx = data.class_to_idx
    return x, y, class_to_idx, epochs

def get_ImageNet_data():
    '''
    Description: Create ImageNet data
    Input: 
    Output: ImageNet x,y, class_to_idx dict, epochs
    '''
    epochs = 150
    x, y = read_pickle("imagenet32")
    class_to_idx = {'orange': 950, 'banana': 954, 'cup': 965, 'red wine': 966, 'fretzel': 932, \
                    'stove': 827, 'printer': 742, 'pillow': 721, 'pijama': 697, 'missile': 657, \
                    'hammer': 587, 'dome': 538, 'balloon': 417, 'nail': 677, 'mask': 643, \
                    'golf ball': 574, 'envelope': 549, 'zebra': 340, 'poodle': 265, 'flamingo': 130, \
                    'snail': 113, 'jelly fish': 107, 'rugby ball': 768, 'mushroom': 947, 'corn': 987}

    return x, y, class_to_idx, epochs