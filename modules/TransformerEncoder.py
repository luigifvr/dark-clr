import numpy as np
import torch
import torch.nn as nn

#################################################
###  TRANSFORMER CLASS
#################################################

'''
Here we define the transformer class used in obtaining the AnomalyCLR representations.
It is based on a standard pytorch TransformerEncoder set-up, with masking.
'''

class Transformer( nn.Module ):
    
    # define and intialize the structure of the neural network

    def __init__( self, input_dim, model_dim, output_dim, n_heads, dim_feedforward, n_layers, learning_rate,  n_head_network_hidden_layers=2,hidden_head_network_dim=1000, head_norm=True, dropout=0.1, opt="adam" ):
        super().__init__()

        # define hyperparameters
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_head_network_hidden_layers = n_head_network_hidden_layers
        self.hidden_head_network_dim = hidden_head_network_dim#8192
        self.head_norm = head_norm
        self.dropout = dropout



        # define subnetworks
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, n_heads, dim_feedforward=dim_feedforward, dropout=dropout, activation = "relu"), n_layers)

        # define the head network using information from the input
        if n_head_network_hidden_layers == 0: # CHANGED IN THIS JR VERSION
            if head_norm: self.norm_layers = nn.ModuleList([nn.LayerNorm(self.hidden_head_network_dim)])
            self.head_layers = nn.ModuleList([nn.Linear(model_dim, self.hidden_head_network_dim)])
            self.head_layers.append(nn.Linear(self.hidden_head_network_dim,output_dim))
        else:
            if head_norm: self.norm_layers = nn.ModuleList([nn.LayerNorm(self.hidden_head_network_dim)])
            self.head_layers = nn.ModuleList([nn.Linear(model_dim, self.hidden_head_network_dim)])
            for i in range(n_head_network_hidden_layers): # adding hidden, square matrices
                if head_norm: self.norm_layers.append(nn.LayerNorm(self.hidden_head_network_dim)) 
                self.head_layers.append(nn.Linear(self.hidden_head_network_dim, self.hidden_head_network_dim))
            self.head_layers.append(nn.Linear(self.hidden_head_network_dim,output_dim))


        # set-up the optimiser used during training
        if opt == "adam":
            self.optimizer = torch.optim.Adam( self.parameters(), lr=self.learning_rate,eps=0.0001 )
        if opt == "sgdca" or opt == "sgdslr" or opt == "sgd":
            self.optimizer = torch.optim.SGD( self.parameters(), lr=self.learning_rate)#, momentum=0.9 )

    #################################################
    #################################################

    # the forward function determines how data is passed through the network

    def forward(self, inpt, mask=None, use_mask=False, use_continuous_mask=False, mult_reps=False):
        '''
        input dims : (batch_size, n_constit, 7)
        but transformer expects (n_constit, batch_size, 7) so we need to transpose
        if use_mask is True, will mask out all inputs with pT=0
        '''

        # check that only one masking is selected in the input
        assert not (use_mask and use_continuous_mask)

        # make a copy of the inputs
        x = inpt + 0.
        
        # make a mask and send it to the device
        if use_mask: pT_zero = x[:,:,0] == 0    # shape: (batch_size,n_constit)
        if use_continuous_mask: pT = x[:,:,0]   # shape: (batch_size,n_constit)
        if use_mask:
            mask = self.make_mask(pT_zero).to(x.device)
        elif use_continuous_mask:
            mask = self.make_continuous_mask(pT).to(x.device)
        else:
            mask = None

        # transpose data for input to the transformer
        x = torch.transpose(x, 0, 1)        # shape: (n_constit,batch_size,7) --> (n_constit,batch_size,3) for DarkJets
        
        # pass data through the embedding layer
        x = self.embedding(x)
        #x = self.embedding1( self.embedding(x) )           # shape: (n_constit,batch_size,model_dim)

        # pass data through the transformer, including the mask
        x = self.transformer(x, mask=mask)  # shape: (n_constit,batch_size,output_dim)

        # we need to again implement the masking on the outputs of the transformer
        if use_mask:
            x[torch.transpose(pT_zero, 0, 1)] = 0
        elif use_continuous_mask:
            x *= torch.transpose(pT, 0, 1)[:,:,None]
        
        # sum over the representation vector for each constituent
        x = x.sum(0)                         # (batch_size, model_dim)

        # pass data through the head network and return the output
        return self.head(x, mult_reps) , x # first outout is z and second is h

    #################################################
    #################################################
    
    # the head network comes after the transformer network, and is where we obtain the representations from

    def head(self, x, mult_reps):
        '''
        if mult_reps=True the head returns a representation vector for each layer of the head networke
        input:  x shape : (batchsize,model_dim)
                mult_reps : boolean
        output: reps shape : (batchsize,output_dim)                  for mult_reps=False
                reps shape : (batchsize,number_of_reps,output_dim)  for mult_reps=True
        '''
        
        # define the activation function
        relu = nn.ReLU()
        
        # return representations from multiple layers for evaluation --------OFF in our case 
        if mult_reps == True:

            # if the head has multiple layers, we loop through them
            if self.n_head_network_hidden_layers > 0:

                # define the reps object with the correct shape
                reps = torch.empty(x.shape[0], self.n_head_network_hidden_layers+1, self.output_dim)

                # the first representation is the one coming straight from the transformer
                reps[:, 0] = x

                # loop through the head layers
                for i, layer in enumerate(self.head_layers):

                    # only apply layer norm on head if head_norm=True
                    if self.head_norm: x = self.norm_layers[i](x)

                    # pass data through the activation function and each layer
                    x = relu(x) 
                    x = layer(x)

                    # append the output of each layer to the reps object we created
                    reps[:, i+1] = x    # shape (n_head_network_hidden_layers, output_dim)

                # return the representations
                return reps  

            # if there are no head layers, i.e. no head network -> just return x in a list with dimension 1
            else:  
                reps = x[:, None, :]    # shape (batchsize, 1, model_dim)

                # return the representations
                return reps  

        # while training the CLR network the loss is computed on the whole network outputs
        # so when mult_reps=False we return only last representation from the head network
        else:  
            x = self.head_layers[0](x)
            if self.head_norm: x = self.norm_layers[0](x)
            # pass the data through the head network, this will do nothing if n_head_network_hidden_layers=0
            for i, layer in enumerate(self.head_layers[1:]):
                if self.head_norm and i < self.n_head_network_hidden_layers  : x = self.norm_layers[i+1](x)
                x = relu(x)
                x = layer(x)    # shape either (model_dim) if no head, or (output_dim) if head exists


            # return the final representation only
            return x  

    #################################################
    #################################################

    # when we want to pass a large amount of data through the transformer to get representations we need to do it in a batch-wise way
    # this function does that

    def forward_batchwise( self, x, batch_size, use_mask=False, use_continuous_mask=False):
        
        # set the device
        device = next(self.parameters()).device
        
        # turn off gradient calculation to save computation
        with torch.no_grad():
        
            # set number of reps based on n_head_network_hidden_layers
            if self.n_head_network_hidden_layers == 0:
                rep_dim = self.model_dim
                number_of_reps = 1
            elif self.n_head_network_hidden_layers > 0:
                rep_dim = self.output_dim
                number_of_reps = self.n_head_network_hidden_layers+1
            
            # initiate the output array
            out = torch.empty( x.size(0), number_of_reps, rep_dim )

            # get idx lists of different batches
            idx_list = torch.split( torch.arange( x.size(0) ), batch_size )
            
            # iterate through batch idx lists
            for idx in idx_list:

                # pass each batch through the network
                output = self(x[idx].to(device), use_mask=use_mask, use_continuous_mask=use_continuous_mask, mult_reps=True).detach().cpu()
                
                # update the output array with the representations
                out[idx] = output

        # return the representations
        return out

    #################################################
    #################################################

    # we define a binary mask for the transformer

    def make_mask(self, pT_zero):
        '''
        Input : batch of bools of whether pT=0, shape (batchsize, n_constit)
        Output : mask for transformer model which masks out constituents with pT=0, shape (batchsize*n_transformer_heads, n_constit, n_constit)
        mask is added to attention output before softmax: 0 means value is unchanged, -inf means it will be masked
        '''

        # get number of constituents
        n_constit = pT_zero.size(1)

        # produce the mask with the correct shape for the transformer architecture
        pT_zero = torch.repeat_interleave(pT_zero, self.n_heads, axis=0)
        pT_zero = torch.repeat_interleave(pT_zero[:,None], n_constit, axis=1)
        mask = torch.zeros(pT_zero.size(0), n_constit, n_constit)

        # the mass is additive in the attention softmax, so masked requires an addition of -np.inf
        mask[pT_zero] = -np.inf

        # return the mask
        return mask
    
    #################################################
    #################################################
    
    # we define a continuous pT dependent masking

    def make_continuous_mask(self, pT):
        '''
        Input: batch of pT values, shape (batchsize, n_constit)
        Output: mask for transformer model: -1/pT, shape (batchsize*n_transformer_heads, n_constit, n_constit)
        mask is added to attention output before softmax: 0 means value is unchanged, -inf means it will be masked
        intermediate values mean it is partly masked
        this function implements IR safety in the transformer
        '''
        
        # get the number of constituents
        n_constit = pT.size(1)

        # create mask with the correct shape 
        pT_reshape = torch.repeat_interleave(pT, self.n_heads, axis=0)
        pT_reshape = torch.repeat_interleave(pT_reshape[:,None], n_constit, axis=1)
        
        # the mask affects all consituents proportional to their pT^0.5, in this case, can be modified
        mask = 0.5*torch.log( pT_reshape ) # Was 0.5

        # return the mask
        return mask
