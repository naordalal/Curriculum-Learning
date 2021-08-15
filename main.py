import sys
from utils import *
from visualization_utils import *
from data_utils import *
import data_utils as data_utils
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from Net import Net
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from torch.nn import functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime
import torchvision.datasets
from time import perf_counter
from torch import nn
import pandas as pd
import numpy as np
from configobj import ConfigObj
import warnings
import random
import torch
import math
import os

seed = 24
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def get_data_by_class(x, y, class_to_idx, super_classes, basic_hyperparams, curri_hyperparams, linear_hyperparameters):
    batch_size = 50
    sets = []
    sets.append(['curriculum'] + get_super_classes_data(x, y, class_to_idx, super_classes) + [curri_hyperparams, True])
    sets.append(['linear_curriculum'] + get_super_classes_data(x, y, class_to_idx, super_classes) + [linear_hyperparameters, True])
    sets.append(['stratified'] + get_super_classes_data(x, y, class_to_idx, super_classes) + [basic_hyperparams, False])
    sets.append(['vanilla'] + get_super_classes_data(x, y, class_to_idx, super_classes) + [basic_hyperparams, False])
    return sets

def cv_training(x, y, hyperparameters, dataset_name, kind, optmizer, loss_fn, device, random_search=False, 
                results_dir_path = 'Results'):
    result = read_pickle(os.path.join(results_dir_path, dataset_name + '_' + kind))
    if result is not None:
       return list(result)
    best_model = [0]
    metrics = []
    scores = []

    n_classes = len(np.unique(y))
    in_channels = x.shape[1]

    outer_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    inner_skf = StratifiedKFold(n_splits=3, random_state=0)

    for idx, (train_val_index, test_index) in enumerate(outer_skf.split(x, y)):
        x_train_val, x_test = x[train_val_index], x[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]

        if random_search:
            best = [0]
            for iter_i in range(50):
                random_hyperparams = select_params_randomly(hyperparameters)
                for train_index, val_index in inner_skf.split(x_train_val, y_train_val):
                    x_train, x_val = x_train_val[train_index], x_train_val[val_index]
                    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
                    model = Net(in_channels, n_classes, loss_fn, optimizer, device, kind, **random_hyperparams)
                    score, _ = model.fit(x_train, y_train, x_val, y_val)
                    if score > best[0]:
                        best = [score, random_hyperparams]
            best_params = best[1]
        else:
            best_params = hyperparameters

        
        model = Net(in_channels, n_classes,loss_fn, optimizer, device, kind, **best_params)

        start = perf_counter()
        score, history = model.fit(x_train_val, y_train_val, x_test, y_test)
        end = perf_counter()
        training_time = end - start
        print(f'fit took {training_time} seconds with score of', score)

        start = perf_counter()
        test_dl = model.build_dataloader(x_test, y_test)
        y_pred = model.predict_proba(test_dl)
        end = perf_counter()
        inference_time = (end - start) / (len(x_test) / 1000)
        
        metrics.append([dataset_name, kind, idx, best_params] + calc_metrics(y_pred, y_test, np.unique(y_test)) + [training_time, inference_time])
        scores.append(score)
        if score > best_model[0]:
            best_model = [score, history]
    result = [best_model[1], np.mean(scores), np.std(scores), metrics]
    write_pickle(os.path.join(results_dir_path, dataset_name + '_' + kind), result)
    return result


def get_data(dataset_name, **params):
  get_data_function = eval(f"get_{dataset_name.split('_')[0]}_data")
  return get_data_function(**params)

def run_on_dataset(data, dataset_name):
    total_metrics_ = []
    models_history = []
    mean_std = []
    for kind, x_, y_, params, cv_apply in data:
        history, mean, std, metrics = cv_training(x_, y_, params, dataset_name, kind, optimizer, loss_fn, device, random_search=cv_apply)
        total_metrics_ += metrics
        models_history.append((kind, history))
        mean_std.append((kind, mean, std))
        print(f"{kind} done training")
    top_1_acc_graph(models_history)
    kinds, means, stds = zip(*mean_std)
    print_bar_plot(kinds, means, stds)
    return total_metrics_

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD
curriculum_hyperparams = {'batch_size': [100], 'epochs': [150], 'pacing_func':[fixed_pacing], 'starting_percent':[0.05, 0.1, 0.15, 0.2],
                        'increase_amount':[1.5, 2, 3], 'step_length':[50, 100, 200, 400], 'lr_step_length': [200, 400, 600, 800],
                        'initial_lr':[0.035, 0.05, 0.01], 'decay_lr':[1.5, 1.3, 1.1]}
curri_hyps = {'batch_size': [100], 'epochs': [25], 'pacing_func':[fixed_pacing], 'starting_percent':[0.1, 0.2],
                        'increase_amount':[1.5, 2], 'step_length':[100, 200], 'lr_step_length': [200, 400, 600, 800],
                        'initial_lr':[0.035, 0.05, 0.01], 'decay_lr':[1.5, 1.3, 1.1]}
linear_curri_hyps = {'batch_size': [100], 'epochs': [25], 'starting_percent':[0.1, 0.2], 'pacing_func':[linear_pacing],
                        'increase_amount':[1.2, 1.1, 1.3], 'step_length':[100, 200], 'lr_step_length': [200, 400, 600, 800],
                        'initial_lr':[0.035, 0.05, 0.01], 'decay_lr':[1.5, 1.3, 1.1]}

basic_hyperparams = {'batch_size': 100, 'epochs': 10, 'initial_lr':0.035}

config = ConfigObj('config_file')

total_metrics = []
for idx, dataset_name in enumerate(config):
    params = {}
    if 'Params' in config[dataset_name]:
      params = config[dataset_name]['Params']
    x, y, class_to_idx, epochs = get_data(dataset_name, **params)
    basic_hyperparams['epochs'], curri_hyps['epochs'], linear_curri_hyps['epochs'] = epochs, [epochs], [epochs]
    classes = config[dataset_name]['Classes']
    for idx, category in enumerate(classes):
        data = get_data_by_class(x, y, class_to_idx, classes[category], basic_hyperparams, curri_hyps, linear_curri_hyps)
        total_metrics += run_on_dataset(data, dataset_name + '_' + category)

df = pd.read_csv('scores_for_report.csv')
scores = df.Accuracy.values.reshape((-1, 4))

algorithm_names = ['curriculum', 'linear curriculum', 'stratified', 'vanilla']
control_algorithm = 'linear curriculum'
friedman_test(algorithm_names, control_algorithm, scores)


