import torch
import numpy as np
import pickle
import os
import math
import hashlib
import torchvision
from torchvision import models
from sklearn import svm
from torchvision import transforms
import random
from scipy.stats import friedmanchisquare
from data_utils import *
from scipy.stats import friedmanchisquare, rankdata
from Orange.evaluation import compute_CD, graph_ranks
import matplotlib.pyplot as plt
class MultiTransformDataset(torch.utils.data.Dataset):
    '''
    Dataset with multiple transforms
    '''
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        input, target = self.dataset[idx]
        if self.transform is None:
            return input, target
        return self.transform(input), target

    def __len__(self,):
        return len(self.dataset)

def fixed_pacing(starting_percent, increase_amount, step_length):
    '''
    Description: fixed pacing function
    Input:  starting_percent - starting percent of data
            increase_amount - increase amount of data each step_length
            step_length - number of batches witout increasing data
    Output: batching function
    '''
    def batching(x, y, current_iteration):
        '''
        Description: batching data function
        Input:  x - data x
                y - data y
                current_iteration - current batch number
        Output: current batch of data
        '''
        inc = increase_amount ** np.floor(current_iteration / step_length)
        percent = min(starting_percent * inc, 1)
        data_limit = len(x) * percent
        return x[:int(data_limit)], y[:int(data_limit)]
    return batching

def linear_pacing(starting_percent, increase_amount):
    '''
    Description: linear pacing function
    Input:  starting_percent - starting percent of data
            increase_amount - increase amount of data each batch
    Output: batching function
    '''
    def batching(x, y, current_iteration):
        '''
        Description: batching data function
        Input:  x - data x
                y - data y
                current_iteration - current batch number
        Output: current batch of data
        '''
        data_limit = starting_percent * len(x) + (increase_amount * current_iteration)
        data_limit = min(data_limit, len(x))
        return x[:int(data_limit)], y[:int(data_limit)]
    return batching

def select_params_randomly(params):
    '''
    Description: randomize params selector
    Input:  params : dict of multiple options of params
    Output: random option of param
    '''
    selected_params = {}
    for key in params.keys():
        selected_params[key] = random.choice(params[key])
    return selected_params

def replace_with_dict(labels, classes):
    '''
    Description: replace array of labels with class numbers
    Input:  labels - array of labels to replace
            classes - classes number to replace with
    Output: replaced labels array
    '''
    k = np.array(classes)
    v = np.array(list(range(len(classes))))
    sidx = k.argsort()
    return v[sidx[np.searchsorted(k, labels, sorter=sidx)]]

def stratified_order(data_, sort_func):
    '''
    Description: order the data in stratified way
    Input:  data_ - the data to sort
            sort_function - the sort function
    Output: data sorted in stratified way
    '''
    data_ = np.array(sorted(data_, key=sort_func))
    counts = np.unique(data_[:,1], return_counts=True)
    start_idx = np.cumsum(counts[1]) - counts[1]
    end_idx = np.cumsum(counts[1])
    result = []
    while True:
        result.append(data_[start_idx])
        start_idx += 1
        cont = start_idx < end_idx
        start_idx = start_idx[cont]
        end_idx = end_idx[cont]
        if len(start_idx) == 0:  
            break
    return np.concatenate(result, axis=0)

def get_super_classes_data(data_, targets, targets_dict, classes_names, batch_size = False, transform=False):
    '''
    Description: get data with specific classes
    Input:  data_ - the all data
            targets - the data labels
            targets_dict - labels to class_id dict
            classes_names - the classes to extract from data
            batch_size - the batch_size to create the data loader
            transform - the transform to do the data
    Output: data with only the desired classes
    '''
    targets = np.array(targets)
    data_mask = targets == -1
    for key in classes_names:
        data_mask = data_mask | (targets == targets_dict[key])
    x = data_[data_mask]
    y = targets[data_mask]
    y = replace_with_dict(y, np.unique(y))
    if transform:
        dataset = MultiTransformDataset(list(zip(x, y)), transform)
        return dataset, torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return [x, y]

def organize_data(x, y, kind, device, batch_size):
    '''
    Description: organize data according to algorith kind
    Input:  x - x
            y - y
            kind - the algorithm kind
            device - CPU or GPU
            batch_size - the batch_size to create the data loader
    Output: the organized data
    '''
    if 'curriculum' in kind:
        file_name = f"scoring/{hashlib.md5(str(x).encode()).hexdigest()}_scoring"
        image_size = (299, 299)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(image_size),])                                  
        dataset = MultiTransformDataset(list(zip(x, y)), transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)
        probs = transfer_scoring(dataloader, device, file_name)

    if 'curriculum' in kind:
        x, y, _ = zip(*stratified_order(np.array(list(zip(x, y, probs))), lambda x: (x[1], -x[2])))
        return [np.array(x), np.array(y)]
    elif kind == 'stratified':
        x, y = zip(*stratified_order(np.array(list(zip(x, y))), lambda x: x[1]))
        return [np.array(x), np.array(y)]
    else:
        return [x, y]

def predict(model, data_loader, device):
    '''
    Description: predict function
    Input:  model - the model
            data_loader - the data to predict
            device - CPU or GPU
    Output: the predicted labels
    '''
    feaures_vecs = []
    real_tags = []
    model.eval()
    for idx, images_batch in enumerate(data_loader):
        feaures_vecs.append(model(images_batch[0].to(device)).squeeze().detach().cpu().numpy())
        real_tags.append(images_batch[1])
    return np.concatenate(feaures_vecs, axis=0), np.concatenate(real_tags, axis=0)

def transfer_scoring(data_loader, device, file_name):
    '''
    Description: transfer scoring for specific data
    Input:  data_loader - the data to scoring
            device - CPU or GPU
            file_name - pickle path to write the scoring
    Output: the scoring
    '''
    order = read_pickle(file_name)
    if order is not None: return order
    inception = models.inception_v3(pretrained=True).to(device)
    x, y = predict(inception, data_loader, device)
    clf = svm.SVC(probability=True)
    clf.fit(x,y)
    scoring = clf.predict_proba(x)
    probs = np.array(list(scoring[np.arange(len(y)),y]))
    write_pickle(file_name, probs)
    return probs

def friedman_test(algorithms_names, control_algorithm, performances_array):
    '''
    Description: friedman test
    Input:  algorithms_names - the algorithms names
            control_algorithm - the control algorithm to compare
            performances_array - the performances array
    Output:
    '''
    #performances_array: [[Algo1,Algo2,Algo3,Algo4], [Algo1, Algo2, Algo3, Algo4], ...]
    _, pvalue = friedmanchisquare(*performances_array)
    alpha = 0.05
    if pvalue < alpha:
        print(f'Reject the null hypothesis with {pvalue} pvalue')
        posthoc(algorithms_names, control_algorithm, performances_array, alpha)
    else:
        print('Not reject the null hypothsis')

def posthoc(algorithms_names, control_algorithm, performances_array, alpha):
    '''
    Description: posthoc function
    Input:  algorithms_names - the algorithms names
            control_algorithm - the control algorithm to compare
            performances_array - the performances array
            alpha - significance level
    Output: bonferroni posthoc graph ranks
    '''
    ranks = np.array([rankdata(-p) for p in performances_array])
    average_ranks = np.mean(ranks, axis=0)
    print('\n'.join('{} average rank: {}'.format(a, r) for a, r in zip(algorithms_names, average_ranks)))
    cd = compute_CD(average_ranks, n=len(performances_array), alpha=str(alpha), test='bonferroni-dunn')
    print(f'CD = {cd}')
    cdmethod = algorithms_names.index(control_algorithm)
    graph_ranks(average_ranks, names=algorithms_names, cd=cd, cdmethod=cdmethod, width=10, textspace=1.5, reverse=True)
    plt.show()