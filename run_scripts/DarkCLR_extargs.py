#//////-------------------------JetCLR extargs
#!/bin/env python3
logfile = "/remote/gpu06/favaro/darkclr/results/my_logfile.txt"
data_path = "/remote/gpu06/favaro/datasets/semivisible_jet_constit/"
ratio = 0.2
n_jets = 1e5
n_constit = 50
n_epochs = 150
batch_size = 256

#Transformer parameters
resdir = "/remote/gpu06/favaro/darkclr/results/minimal_setup_dim16/Run0.5_TestA3_0.0_4/"
dropchance = 0.5

dim = 512

model_dim = 128
output_dim = dim
n_heads = 4
dim_feedforward = 512
n_layers = 4
n_head_network_hidden_layers = 0
hidden_head_network_dim = dim

opt = "adam"
learning_rate = 0.00005

mask = True
cmask = False

ptcm = 0.1 # distort_jets
ptst = 0.1 #
trsw = 1.0 # translate_jets

#loss function params
temperature = 0.10
