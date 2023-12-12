#!/bin/env python3.7

# load custom modules required for CLR training
from modules.TransformerEncoder import Transformer # new relu here
from modules.contrastive_losses import clr_loss,anomclr_loss,anomclr_plus_loss,anomclr_plus_loss_bonus_scalars
from modules.jet_augmentations import rotate_jets, distort_jets, rescale_pts, crop_jets, translate_jets, collinear_fill_jets, collinear_fill_jets_fast, shift_pT, pt_reweight_jet, drop_constits_jet, rescale_reweight_pts, drop_constits_jet_sampled

# import args from extargs.py file
import DarkCLR_extargs as args

# load standard python modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

import shutil
#starting counter
t0 = time.time()


#loading in data ------------------------------------------------------------

from semi_dataset import SemiV_Dataset
from torch.utils.data import DataLoader


# set up results directory---------------------------------------------------

dropchance = args.dropchance
contamination_fraction = 0.0

expt_dir = args.resdir

# check if experiment already exists
if os.path.isdir(expt_dir):
    sys.exit("ERROR: experiment already exists")
else:
    os.makedirs(expt_dir)
    # initialise logfile
    logfile = open(expt_dir + "/my_logfile.txt", "a" )
    print( "logfile initialised" , file=logfile, flush=True  )
print("experiment: "+str(expt_dir), file=logfile, flush=True)

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
print( "device: " + str( device )  , file=logfile, flush=True  )

# dataloader and conversion to tensor
training_set = SemiV_Dataset(
                                    data_path =args.data_path,
                                    signal_origin= "aachen",
                                    usage= "training",
                                    number_constit= args.n_constit,
                                    number_of_jets= args.n_jets,
                                    ratio = args.ratio,
                                    contamination_fraction=contamination_fraction,
                                    )

dl_training = DataLoader(training_set,batch_size=args.batch_size, shuffle=True)

t1 = time.time()
print( "time taken to load and preprocess data: "+str( np.round( t1-t0, 2 ) ) + " seconds" , file=logfile, flush=True   )


#SAVE THE EXTARGS ------------------------------------------------------------------
# Get the path of the current script
script_path = __file__
# Define the destination file path to save the script
destination_path = expt_dir + "script.py"
# Copy the contents of the script to the destination file
shutil.copyfile(script_path, destination_path)
# Define the source script file path
extargs_script_path = "/remote/gpu05/rueschkamp/DarkCLR/DarkCLR_extargs.py"
# Define the destination file path to save the script
extargs_destination_path = expt_dir + "extargs.py"
# Copy the contents of the script to the destination file
shutil.copyfile(extargs_script_path, extargs_destination_path)

#initializing the network  ------------------------------------------------------------------
input_dim = 3 

net = Transformer( input_dim, args.model_dim, args.output_dim, args.n_heads, args.dim_feedforward, args.n_layers, args.learning_rate, args.n_head_network_hidden_layers,hidden_head_network_dim=args.hidden_head_network_dim,head_norm=False, dropout=0.1, opt=args.opt )
net.to( device );

# Plot the training loss
def plotting(epoch,stars,dashes):
    x = np.linspace(0, epoch - 1, epoch)

    fig, ax1 = plt.subplots()
    ax1.plot(x, losses, label="loss")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f"AnomCLR Loss with {args.n_jets:.0e} jets")
    ax1.legend()
    
    plt.savefig((expt_dir+f"CLR-Loss_{epoch}epochs_{args.n_jets:.0e}Jets.pdf"), format="pdf")

    # Create a new figure and axes for the second plot
    fig, ax2 = plt.subplots()
    ax2.plot(x, stars, label="Anomaly Similarity")
    ax2.plot(x, dashes, label="Physical Similarity")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Similarity')
    ax2.set_title('Similarity of the Transformer Network')
    ax2.legend()
    plt.savefig((expt_dir+f"Similarities_{epoch}epochs_{args.n_jets:.0e}Jets.pdf"), format="pdf")

print( "starting training loop, running for " + str( args.n_epochs ) + " epochs" +  str( args.n_jets )+"Jets" , file=logfile, flush=True  )
print( "---"  , file=logfile, flush=True  )

losses = []
stars = []
dashes = []

best_loss = 1e3
#training loop
for epoch in range( args.n_epochs +1):
    # initialise timing stats
    losses_e = []
    stars_e = [] #anomaly
    dashes_e = [] # physical

    if (epoch%20) ==0 :

        tms0 = time.time()

        filename = expt_dir+ f"Model_{epoch}epochs_{args.n_jets:.0e}Jets.pt"
        torch.save(net.state_dict(), filename)

        plotting(epoch,stars,dashes)

        tms1 = time.time()
        print( f"time taken to plot and save model: {round( tms1-tms0, 1 )}s", file=logfile, flush=True  )

    print("epoch: ",epoch, file=logfile, flush=True)
    for i, (data, labels) in enumerate(dl_training):
        net.optimizer.zero_grad()
        x_i = data
        
        x_i = rotate_jets( x_i ) # undo the centering
        x_j = x_i.clone()
        x_k = x_i.clone()

        # pos augmentation
        x_j = rotate_jets( x_j ) 
        x_i = translate_jets( x_i, width=args.trsw )
        x_j = translate_jets( x_j, width=args.trsw )
        x_k = translate_jets( x_k, width=args.trsw )

        # neg augmentation
        x_k = drop_constits_jet(x_k, dropchance)

        x_i = rescale_pts( x_i )
        x_j = rescale_pts( x_j )
        x_k = rescale_pts( x_k )

        x_i = x_i.transpose(1,2)
        x_j = x_j.transpose(1,2)
        x_k = x_k.transpose(1,2)

        z_i , h_i = net(x_i, use_mask=args.mask, use_continuous_mask=args.cmask) #dim: x_i = torch.Size([128, 50, 3]) and z_i = torch.Size([128, 1000])
        z_j ,_= net(x_j, use_mask=args.mask, use_continuous_mask=args.cmask)
        z_k ,_ = net(x_k,use_mask = args.mask, use_continuous_mask = args.cmask)
        # compute the loss, back-propagate, and update scheduler if required
        loss , star , dash = anomclr_plus_loss_bonus_scalars( z_i, z_j, z_k,args.temperature, scalar_positives= 1 ,scalar_augmented=1 )

        loss = loss +  1.e-3*torch.mean(torch.linalg.vector_norm(z_i, dim=-1))
        loss = loss.to( device )

        loss.backward()
        net.optimizer.step()
        losses_e.append( loss.detach().cpu().numpy() )
        stars_e.append(np.mean(star.detach().cpu().numpy()))
        dashes_e.append(np.mean(dash.detach().cpu().numpy()))
        
    
    loss_e = np.mean( np.array( losses_e ) )
    losses.append( loss_e )

    if loss_e < best_loss:
        best_loss = loss_e
        print(f"new best loss: {best_loss}", file=logfile, flush=True)
        filename = expt_dir+ f"Model_bestepochs_{args.n_jets:.0e}Jets.pt"
        torch.save(net.state_dict(), filename)

    star_e = np.mean(np.array(stars_e))
    stars.append(star_e)

    dash_e = np.mean(np.array(dashes_e))
    dashes.append(dash_e)


t2 = time.time()

print( f"Training done. Time taken : {round( t2-t1, 1 )}s"  , file=logfile, flush=True )

# save final model
filename = expt_dir+ f"Model_finalepochs_{args.n_jets:.0e}Jets.pt"
torch.save(net.state_dict(), filename)
