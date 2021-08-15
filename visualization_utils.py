import torch
import numpy as np
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, precision_score, recall_score, confusion_matrix

def smooth_func(y, box_pts = 10):
    '''
    Description: Test accuracy weighted Smoothing
    Input: y - accuracy scores
           box_pts - # points for smoothing
    Output: smoothed test accuracy
    '''
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth[:int(-box_pts / 2)]

def print_bar_plot(algo_labels, means, stds):
    '''
    Description: Plotting bar (with error) graph for the examined algorithms
    Input: algo_labels - algorithms names
           means - external CV average per algorithm
           stds - external CV standard deviation per algorithm
    Output: -
    '''
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(algo_labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()

def top_1_acc_graph(histories, smooth=True):
    '''
    Description: Plotting top1 accuracy graph for the examined algorithms
    Input: histories - test accuracy on the best model from the external CV
           smooth - applying smoothing flag
    Output: -
    '''    
    for name, history in histories:
        if smooth:
            history = smooth_func(history)
        plt.plot(np.array(list(range(len(history)))) * 20, history, label= name)
    plt.legend(loc = 'lower right')
    plt.xlabel('Batch Number')
    plt.ylabel('Accuracy')
    plt.show()

def calc_metrics(predictions_proba, actual, classes):
    '''
    Description: Calculating metrics each algorithm and dataset.
    Input: predictions_proba - model predicted labels (softmax vectors)
           actual - the actual labels
           classes - classes labels (numbers)
    Output: -
    '''    
    predictions = np.argmax(predictions_proba, axis=1)
    cm = confusion_matrix(actual, predictions)
    accuracy = accuracy_score(actual, predictions)
    FP = (cm.sum(axis=0) - np.diag(cm)).astype(float)
    FN = (cm.sum(axis=1) - np.diag(cm)).astype(float)
    TP = np.diag(cm).astype(float)
    TN = (cm.sum() - (FP + FN + TP)).astype(float)
    tpr = np.mean(TP/(TP+FN))
    fpr = np.mean(FP/(FP+TN))
    precision = precision_score(actual, predictions, average='macro')
    
    auc_classes = []
    area_under_precision_recall_curve_classes = []
    for class_id in classes:
      class_predictions_proba = predictions_proba[:,class_id]
      f, t, thresholds = roc_curve(actual, class_predictions_proba, pos_label=class_id)
      pr, rec, thresholds = precision_recall_curve(actual, class_predictions_proba, pos_label=class_id)
      auc_classes.append(auc(f, t))
      area_under_precision_recall_curve_classes.append(auc(rec, pr))      

    auc_score = np.array(auc_classes).mean()
    area_under_precision_recall_curve = np.array(area_under_precision_recall_curve_classes).mean()
    return [accuracy, tpr, fpr, precision, auc_score, area_under_precision_recall_curve]

def scores_to_csv(total_metrics):
    '''
    Description: Creating the results including the mterics, times, and data descriptions
    Input: total_metrics - full description each external CV
    Output: full and lighter scores detailed csv's 
    '''
    columns = ['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]', 'Hyperparameters Values', 'Accuracy', 'TPR', 'FPR', 'Precision',
            'AUC', 'PR-Curve', 'Training Time', 'Inference Time']
    df = pd.DataFrame(total_metrics, columns=columns)
    df.to_csv('scores.csv')

    idx = df.groupby(['Dataset Name', 'Algorithm Name'])['Accuracy'].transform(max) == df['Accuracy']
    sub_df = df[idx].drop(['Unnamed: 0', 'Cross Validation [1-10]', 'Hyperparameters Values', 'PR-Curve', 'Training Time', 'Inference Time'], axis=1).reset_index()
    sub_df.drop(['index'], axis=1, inplace=True)
    sub_df.to_csv('scores_for_report.csv')