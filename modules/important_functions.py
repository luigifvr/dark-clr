# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# load standard python modules
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
print( "device: " + str( device )    )




def get_representations(droprate,kind = None,special_path = None):

    standard_path = "/remote/gpu05/rueschkamp/Representations"

    if kind == None:
        qcd_path = standard_path +"/Run" + droprate +"/qcd_representations.npy"
        aachen_path = standard_path +"/Run" + droprate +"/aachen_representations.npy"
        heidelberg_path = standard_path +"/Run" + droprate +"/heidelberg_representations.npy"
    elif kind =="special":
        qcd_path = standard_path +"/" + special_path +"/qcd_representations.npy"
        aachen_path = standard_path +"/" + special_path +"/aachen_representations.npy"
    elif kind=="full_path":
        qcd_path =  special_path +"/qcd_representations.npy"
        aachen_path = special_path +"/aachen_representations.npy"

    qcd_representations = torch.Tensor(np.load(qcd_path))
    aachen_representations = torch.Tensor(np.load(aachen_path))
    #heidelberg_representation = torch.Tensor(np.load(heidelberg_path))

    return qcd_representations , aachen_representations #, heidelberg_representation
    #return qcd_representations[80000:,:] , aachen_representations[80000:,:]




# for plotting

def plot_feature_values(representation,second_representation = None):
    features = torch.mean(qcd_reps,dim=0)
    features_std = torch.std(qcd_reps,dim=0)
    sorted_tensor, indices = torch.sort(features, descending=True)
    sorted_tensor_err = features_std[indices]
    # Convert the tensors to numpy arrays
    sorted_tensor_np = sorted_tensor.detach().numpy()
    sorted_tensor_err_np = sorted_tensor_err.detach().numpy()

    # Plot the tensor values with error bars
    #plt.errorbar(range(len(sorted_tensor_np)), sorted_tensor_np, yerr=sorted_tensor_err_np)
    plt.plot(range(len(sorted_tensor_np)), sorted_tensor_np)
    plt.fill_between(range(len(sorted_tensor_np)), sorted_tensor_np - sorted_tensor_err_np,sorted_tensor_np + sorted_tensor_err_np,alpha=0.5)
    plt.ylabel('Mean value of features', fontsize=12)
    plt.xlabel('feature', fontsize=12)
    # Display the plot
    #if second_representation != None:
    #    plot_feature_values(second_representation)

    plt.show()

def plot_feature_values_differences(representation,second_representation):

    representation_difference = representation - second_representation

    #similar plotting to just values
    features = torch.mean(representation_difference,dim=0)
    features_std = torch.std(representation_difference,dim=0)
    sorted_tensor, indices = torch.sort(features, descending=True)
    sorted_tensor_err = features_std[indices]
    # Convert the tensors to numpy arrays
    sorted_tensor_np = sorted_tensor.detach().numpy()
    sorted_tensor_err_np = sorted_tensor_err.detach().numpy()

    # Plot the tensor values with error bars
    #plt.errorbar(range(len(sorted_tensor_np)), sorted_tensor_np, yerr=sorted_tensor_err_np)
    plt.plot(range(len(sorted_tensor_np)), sorted_tensor_np)
    plt.fill_between(range(len(sorted_tensor_np)), sorted_tensor_np - sorted_tensor_err_np,sorted_tensor_np + sorted_tensor_err_np,alpha=0.5)
    plt.ylabel('Mean value of features_differences', fontsize=12)
    plt.xlabel('feature', fontsize=12)
    # Display the plot
    #if second_representation != None:
    #    plot_feature_values(second_representation)

    plt.show()