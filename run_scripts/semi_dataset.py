import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

class SemiV_Dataset(Dataset):
    def __init__(self, data_path,signal_origin, usage, number_constit,ratio, number_of_jets = None, contamination_fraction=0.0 ):
        #getting data
        num_jets = number_of_jets
        if number_of_jets == None:
            num_jets = 1e3
        
        background_path = data_path + "qcd" + "_constit.h5"
        signal_path = data_path + signal_origin + "_constit.h5" # if qcd is also given as input this outputs solely qcd.

        train_data , train_labels = converter_plus(background_path,int( num_jets) ,number_constit,usage,ratio)
        train_data_signal, train_labels_signal = converter_plus(signal_path, int(num_jets*contamination_fraction), number_constit, usage, ratio)
        if contamination_fraction != 0.0:
            train_data = torch.cat((train_data, train_data_signal))
            train_labels = torch.cat((train_labels, train_labels_signal))

        test_data_background , test_labels_background = converter_plus(background_path,int(num_jets *ratio),number_constit,usage,ratio)
        test_data_signal , test_labels_signal = converter_plus(signal_path,int(num_jets *ratio),number_constit,usage,ratio)

        test_data = torch.cat((test_data_background,test_data_signal))
        test_labels = torch.cat((test_labels_background,test_labels_signal))

        # re-scale test data, for the training data this will be done on the fly due to the augmentations
        test_data = rescale_pts( test_data )

        if usage== "training" :
            self.labels = train_labels
            self.data = train_data
        elif usage=="testing":
            self.labels = test_labels
            self.data = test_data
        else:
            print("check usage!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        return data, label

#doing the convertions to pt,eta,phi and putting them to gpu
def converter_plus(data_path,num_jets,number_constit,usage,ratio):
    if usage== "training" :
        data_frame = pd.read_hdf(data_path, key='table', start=0, stop=num_jets)
    elif usage=="testing":
        data_frame = pd.read_hdf(data_path, key='table', start=int(num_jets*(1/ratio)), stop=int(num_jets+num_jets*(1/ratio)))
    else:
        print("check usage!")

    
    #give names
    max_const = number_constit
    feat_list =  ["E","PX","PY","PZ"] 
    cols = ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(max_const)]
    #reshape
    vec4 = np.expand_dims(data_frame[cols],axis=-1).reshape(-1, len(feat_list), max_const)
    #getting p_vec and E
    E  = vec4[:,0,:]
    pxs   = vec4[:,1,:]
    pys   = vec4[:,2,:]
    pzs   = vec4[:,3,:]
    
    #calculate pT, eta, phi
    pTs = pT(pxs,pys)
    etas = eta(pTs,pzs)
    phis = phi(pxs,pys)

    for i in range(etas.shape[0]):
        etas[i,:], phis[i,:] = centre_jet( etas[i,:], phis[i,:], pTs[i,:] )
    pTs, indices = torch.sort(torch.Tensor(pTs), dim=1, descending=True) # Ordering pTs
    etas = torch.Tensor(etas).gather(dim=1,index=indices)
    phis = torch.Tensor(phis).gather(dim=1,index=indices)
    
    jet_data = torch.cat((pTs.unsqueeze(1),etas.unsqueeze(1),phis.unsqueeze(1)),axis = 1)
    labels = data_frame["is_signal_new"].to_numpy()
    return (jet_data).to(device) , torch.Tensor(labels).to(device)


# returns momenta for the centroid and principal axis
def input_mom (x, y, weights, x_power, y_power):
    return ((x**x_power)*(y**y_power)*weights).sum()

def centre_jet(x, y, weights):
    x_centroid = input_mom(x, y, weights, 1, 0) / weights.sum()
    y_centroid = input_mom(x, y, weights, 0, 1)/ weights.sum()
    x = x - x_centroid
    y = y - y_centroid
    return x, y

def pT(px,py):
    pT = np.sqrt( px**2 + py**2 )
    return pT

# Calculate pseudorapidity of pixel entries
def eta(pT, pz):
    small = 1e-10
    small_pT = (np.abs(pT) < small)
    small_pz = (np.abs(pz) < small)
    not_small = ~(small_pT | small_pz)
    theta = np.arctan(pT[not_small]/pz[not_small])
    theta[theta < 0] += np.pi
    etas = np.zeros_like(pT)
    etas[small_pz] = 0
    etas[small_pT] = 1e-10
    etas[not_small] = np.log(np.tan(theta/2))
    return etas

# Calculate phi of the pixel entries (in range [-pi,pi])
# phis are returned in radians, np.arctan(0,0)=0 -> zero constituents set to -np.pi
def phi (px, py):
    phis = np.arctan2(py,px)
    phis[phis < 0] += 2*np.pi
    phis[phis > 2*np.pi] -= 2*np.pi
    phis = phis - np.pi 
    return phis
    
def rescale_pts(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    '''
    batch_rscl = batch.clone()
    batch_rscl[:,0,:] = torch.nan_to_num(batch_rscl[:,0,:]/300, nan=0.0, posinf=0.0, neginf=0.0)
    return batch_rscl
