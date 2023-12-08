import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

#peak to peak torch
def ptp(input, dim=None, keepdim=False):
    if dim is None:
        return input.max() - input.min()
    return input.max(dim, keepdim).values - input.min(dim, keepdim).values



#-------------------------------------_Positive Augmentations_---------------------------------------#

def translate_jets(batch, width=1.0):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of eta-phi translated jets, same shape as input
    '''

    device = batch.device
    mask = (batch[:, 0] > 0)  # 1 for constituents with non-zero pT, 0 otherwise
    ptp_eta = ptp(batch[:, 1, :], dim=-1, keepdim=True)  # ptp = 'peak to peak' = max - min
    ptp_phi = ptp(batch[:, 2, :], dim=-1, keepdim=True)  # ptp = 'peak to peak' = max - min
    low_eta = -width * ptp_eta
    high_eta = +width * ptp_eta
    low_phi = torch.max(-width * ptp_phi, -torch.tensor(np.pi, device=device) - torch.amin(batch[:, 2, :], dim=-1, keepdim=True))
    high_phi = torch.min(+width * ptp_phi, +torch.tensor(np.pi, device=device) - torch.amax(batch[:, 2, :], dim=-1, keepdim=True))
    shift_eta = mask * torch.rand((batch.shape[0], 1), device=device) * (high_eta - low_eta) + low_eta
    shift_phi = mask * torch.rand((batch.shape[0], 1), device=device) * (high_phi - low_phi) + low_phi
    shift = torch.stack([torch.zeros((batch.shape[0], batch.shape[2]), device=device), shift_eta, shift_phi], dim=1)
    shifted_batch = batch + shift
    return shifted_batch

def rotate_jets(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets rotated independently in eta-phi, same shape as input
    '''
    device = batch.device
    batch_size = batch.size(0)
    
    rot_angle = torch.rand(batch_size, device=device) * 2 * np.pi
    c = torch.cos(rot_angle)
    s = torch.sin(rot_angle)
    o = torch.ones_like(rot_angle)
    z = torch.zeros_like(rot_angle)

    #print(o.shape)

    rot_matrix = torch.stack([
    torch.stack([o, z, z], dim=0),
    torch.stack([z, c, s], dim=0),
    torch.stack([z, -s, c], dim=0)], dim=1) # (3, 3, batch_size]

    #print(rot_matrix[:,:,0])

    return torch.einsum('ijk,lji->ilk', batch, rot_matrix)

def normalise_pts(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-normalised jets, pT in each jet sums to 1, same shape as input
    '''
    batch_norm = batch.clone()
    batch_norm[:, 0, :] = torch.nan_to_num(batch_norm[:, 0, :] / torch.sum(batch_norm[:, 0, :], dim=1)[:, None], posinf=0.0, neginf=0.0)
    return batch_norm

def rescale_pts(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    '''
    batch_rscl = batch.clone()
    #batch_rscl[:,0,:] = torch.nan_to_num(batch_rscl[:,0,:]/600, nan=0.0, posinf=0.0, neginf=0.0) #previos one
    batch_rscl[:,0,:] = torch.nan_to_num(batch_rscl[:,0,:]/300, nan=0.0, posinf=0.0, neginf=0.0)
    #batch_rscl[:,0,:] = torch.nan_to_num((batch_rscl[:,0,:]/300)**0.1, nan=0.0, posinf=0.0, neginf=0.0)
    return batch_rscl


def rescale_reweight_pts(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    '''
    batch_rscl = batch.clone()
    #batch_rscl[:,0,:] = torch.nan_to_num(batch_rscl[:,0,:]/600, nan=0.0, posinf=0.0, neginf=0.0) #previos one
    #batch_rscl[:,0,:] = torch.nan_to_num(batch_rscl[:,0,:]/300, nan=0.0, posinf=0.0, neginf=0.0)
    batch_rscl[:,0,:] = torch.nan_to_num((batch_rscl[:,0,:]/300)**0.1, nan=0.0, posinf=0.0, neginf=0.0)
    return batch_rscl



def rescale_batchwise_reweight_pts(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    '''
    batch_rscl = batch.clone()
    total_pt = torch.sum(batch_rscl[:,0,:],dim=1)
    #print(total_pt.unsqueeze(1).size())
    rescaled_pts=(batch_rscl[:,0,:]/total_pt.unsqueeze(1) )**0.5
    total_rescaled_pt = torch.sum(rescaled_pts,dim=1)

    batch_rscl[:,0,:] = torch.nan_to_num(rescaled_pts/total_rescaled_pt.unsqueeze(1), nan=0.0, posinf=0.0, neginf=0.0)
    return batch_rscl

def crop_jets( batch, nc ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of cropped jets, each jet is cropped to nc constituents, shape (batchsize, 3, nc)
    '''
    batch_crop = batch.clone()
    return batch_crop[:,:,0:nc]

def distort_jets(batch, strength=0.1, pT_clip_min=0.1):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with each constituents position shifted independently, shifts drawn from normal with mean 0, std strength/pT, same shape as input
    '''
    pT = batch[:, 0].to(device)  # (batchsize, n_constit)
    
    #print(pT.device)
    shift_eta = torch.nan_to_num(
        strength * torch.randn(batch.shape[0], batch.shape[2]).to(device) / pT.clip(min=pT_clip_min),
        posinf=0.0,
        neginf=0.0,
    )  # * mask
    shift_phi = torch.nan_to_num(
        strength * torch.randn(batch.shape[0], batch.shape[2]).to(device) / pT.clip(min=pT_clip_min),
        posinf=0.0,
        neginf=0.0,
    )  # * mask
    shift = torch.stack([torch.zeros((batch.shape[0], batch.shape[2])).to(device), shift_eta, shift_phi], 1)
    #print(shift.device)
    shift.to(device)
    return batch + shift

def collinear_fill_jets(batch):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with collinear splittings, the function attempts to fill as many of the zero-padded args.nconstit
    entries with collinear splittings of the constituents by splitting each constituent at most once, same shape as input
    '''
    #device = batch.device
    batch_size = batch.size(0)
    n_constit = batch.size(2)
    
    batchb = batch.clone()
    nzs = torch.sum(batch[:,:,0]>0, dim=1)
    nzs1 = torch.max(torch.cat([nzs.view(batch_size, 1), torch.full((batch_size, 1), int(n_constit/2), dtype=torch.int64, device=device)], dim=1), dim=1).values
    zs1 = n_constit - nzs1
    
    for k in range(batch_size):
        els = torch.randperm(nzs1[k], device=device)[:zs1[k]]
        rs = torch.rand(zs1[k], device=device)
        for j in range(zs1[k]):
            batchb[k,0,els[j]] = rs[j]*batch[k,0,els[j]]
            batchb[k,0,nzs[k]+j] = (1-rs[j])*batch[k,0,els[j]]
            batchb[k,1,nzs[k]+j] = batch[k,1,els[j]]
            batchb[k,2,nzs[k]+j] = batch[k,2,els[j]]
            
    return batchb

def collinear_fill_jets_fast(batch):
    '''
    Fill as many of the zero-padded entries with collinear splittings
    of the constituents by splitting each constituent at most once.
    Parameters
    ----------
    batch : torch.Tensor
        batch of jets with zero-padding
    Returns
    -------
    batch_filled : torch.Tensor
        batch of jets with collinear splittings
    '''
    batch_filled = batch.clone()
    n_constit = batch_filled.shape[2]
    n_nonzero = torch.sum(batch_filled[:,0,:]>0, dim=1)
    
    n_split = torch.min(torch.stack([n_nonzero, n_constit-n_nonzero], dim=1), dim=1).values
    idx_flip = torch.where(n_nonzero != n_split)[0]
    mask_split = (batch_filled[:,0,:] != 0)
    
    mask_split[idx_flip] = torch.flip(mask_split[idx_flip].float(), dims=[1]).bool()

    #print(mask_split)
    mask_split[idx_flip] = ~mask_split[idx_flip]
    r_split = torch.rand(size=mask_split.shape, device=batch.device)
    
    a = r_split*mask_split*batch_filled[:,0,:]
    b = (1-r_split)*mask_split*batch_filled[:,0,:]
    c = ~mask_split*batch_filled[:,0,:]
    batch_filled[:,0,:] = a+c+torch.flip(b, dims=[1])
    batch_filled[:,1,:] += torch.flip(mask_split*batch_filled[:,1,:], dims=[1])
    batch_filled[:,2,:] += torch.flip(mask_split*batch_filled[:,2,:], dims=[1])
    return batch_filled

#-------------------------------------_Negative Augmentations_---------------------------------------#

def recentre_jet(batch):
    batchc = batch.clone()
    pts = batch[:,0,:]
    etas = batch[:,1,:]
    phis = batch[:,2,:]
    if torch.sum( pts ) != 0:
        eta_shift = torch.sum(  pts*etas  ) / torch.sum( pts )
        phi_shift = torch.sum(  pts*phis ) / torch.sum( pts )
        etas = etas - eta_shift
        phis = phis - phi_shift

    pTs, indices = torch.sort(pts, dim=1, descending=True) # Ordering pTs
    etas = etas.gather(dim=1,index=indices)
    phis = phis.gather(dim=1,index=indices)

    batch_recentred = torch.cat((pTs.unsqueeze(1),etas.unsqueeze(1),phis.unsqueeze(1)),axis = 1)

    #print(batch_dropped.size())
    return batch_recentred



def recentre_jet_old(batch):
    batchc = batch.clone()
    pts = batch[:,0,:]
    etas = batch[:,1,:]
    phis = batch[:,2,:]
    eta_shift = torch.sum(  pts*etas  ) / torch.sum( pts )
    phi_shift = torch.sum(  pts*phis ) / torch.sum( pts )
    batchc[:,1,:] = batch[:,1,:] - eta_shift
    batchc[:,2,:] = batch[:,2,:] - phi_shift
    return batchc

def shift_pT(batch):
    batch_shifted = batch.clone()
    shifts = 1 + torch.rand(batch_shifted.shape[0], 1) * 4 
    shifts = shifts.to(device)
    batch_shifted[:,0] *= shifts

    return batch_shifted

def pt_reweight_jet( batch, beta=1.5 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    Dim 1 ordering: (pT, eta, phi)
    Output: batch of jets where the pt of the constituents in each jet has has been re-weighted by some power
    Note: rescale pts so that the augmented jet pt matches the original
    '''
    batchc = batch.clone()

    etas = batchc[:,1,:]
    phis = batchc[:,2,:]
    batchc = batchc[:,0,:]**beta
    pts = torch.sum( batch[:,0,:], axis=1 )
    pts_aug = torch.sum( batchc, axis=1 )

    #pts_aug[pts_aug == 0] = 1*e-7
    pt_rescale =  pts/pts_aug
    pTs = pt_rescale.unsqueeze(-1)* batchc
    pTs, indices = torch.sort(pTs, dim=1, descending=True) # Ordering pTs
    etas = etas.gather(dim=1,index=indices)
    phis = phis.gather(dim=1,index=indices)
    #print(pTs)

    jet = torch.cat((pTs.unsqueeze(1),etas.unsqueeze(1),phis.unsqueeze(1)),axis = 1)
    return recentre_jet( jet.float() )

def drop_constits_jet( batch, prob=0.1 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    Dim 1 ordering: (pT, eta, phi)
    Output: batch of jets where each jet has some fraction of missing constituents
    Note: rescale pts so that the augmented jet pt matches the original
    '''
    batch_dropped = batch.clone()
    #n_nonzero = torch.sum(batch_dropped[:,0,:]>0, dim=1)
    nj = batch_dropped.shape[0]
    nc = batch_dropped.shape[2]

    mask = torch.rand((nj, nc)) > prob
    mask = mask.int().to(device)
    num_zeros_tensor = (mask == 0).sum().item()
    #print(num_zeros_tensor)
    batch_dropped = batch_dropped * mask.unsqueeze(1)

    #print(batch_dropped)
    pts = torch.sum( batch[:,0,:], axis=1 )
    pts_aug = torch.sum( batch_dropped[:,0,:], axis=1 )

    pTs= batch_dropped[:,0,:]
    
    if torch.any(pts_aug != 0):
        pt_rescale = torch.where(pts_aug != 0, pts / pts_aug, torch.ones_like(pts))
        #print(pt_rescale)
        pTs *= pt_rescale.unsqueeze(1)

    pTs, indices = torch.sort(pTs, dim=1, descending=True) # Ordering pTs
    etas = batch_dropped[:,1,:]
    etas = etas.gather(dim=1,index=indices)
    phis = batch_dropped[:,2,:]
    phis = phis.gather(dim=1,index=indices)

    
    batch_dropped = torch.cat((pTs.unsqueeze(1),etas.unsqueeze(1),phis.unsqueeze(1)),axis = 1)

    non_zero_count = torch.sum(pTs != 0, dim=-1, keepdim=True)
    #print(batch_dropped)
    if torch.min(non_zero_count)==0:
        return drop_constits_jet(batch,prob)
    else:
        return recentre_jet( batch_dropped )

def drop_constits_jet_safe( batch, prob=0.5 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    Dim 1 ordering: (pT, eta, phi)
    Output: batch of jets where each jet has some fraction of missing constituents
    Note: rescale pts so that the augmented jet pt matches the original
    '''
    batch_dropped = batch.clone()

    pTs= batch_dropped[:,0,:]
    non_zero_count = torch.sum(pTs != 0, dim=-1, keepdim=True)
    dropping_numbers = torch.round( non_zero_count * prob)
    #print(dropping_numbers)
    total_length_mask = pTs.size(1)

    mask_safe = []
    # Create a mask with zeros distributed between ones
    for i in range(len(dropping_numbers)):
        length_of_nonzero_pTs = int(non_zero_count[i])
        #print(dropping_numbers[i])
        n_drop = int(dropping_numbers[i])

        non_zero_mask = torch.cat((torch.zeros(n_drop), torch.ones(length_of_nonzero_pTs - n_drop))) #creating mask for non zero entries
        shuffled_non_zero_mask = non_zero_mask[torch.randperm(non_zero_mask.size(0))]  # Generate random permutation of non-zero mask

        #print(shuffled_non_zero_mask)

        jet_mask = torch.cat((shuffled_non_zero_mask,torch.zeros(total_length_mask-length_of_nonzero_pTs))) #Creating mask for whole jet

        mask_safe.append(jet_mask)

    mask = torch.stack(mask_safe).to(device)

    result = batch_dropped * mask.unsqueeze(1)

    

    #From here on just rescaling and reordering --> Masking is done here. 

    #print(batch_dropped)
    pts = torch.sum( result[:,0,:], axis=1 )
    pts_aug = torch.sum( result[:,0,:], axis=1 )


    if torch.any(pts_aug != 0):
        pt_rescale = torch.where(pts_aug != 0, pts / pts_aug, torch.ones_like(pts))
        #print(pt_rescale)
        pTs *= pt_rescale.unsqueeze(1)

    pTs_dropped =result[:,0,:]

    pTs, indices = torch.sort(pTs_dropped, dim=1, descending=True) # Ordering pTs
    #print("pts:", pTs)
    etas = result[:,1,:]
    etas = etas.gather(dim=1,index=indices)
    phis = result[:,2,:]
    phis = phis.gather(dim=1,index=indices)

    
    batch_dropped = torch.cat((pTs.unsqueeze(1),etas.unsqueeze(1),phis.unsqueeze(1)),axis = 1)

    #print(batch_dropped)
    return recentre_jet( batch_dropped )

def drop_constits_jet_sampled( batch, a=0.1, b=0.6 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    Dim 1 ordering: (pT, eta, phi)
    Output: batch of jets where each jet has some fraction of missing constituents
    Note: rescale pts so that the augmented jet pt matches the original
    '''
    batch_dropped = batch.clone()
    #n_nonzero = torch.sum(batch_dropped[:,0,:]>0, dim=1)
    nj = batch_dropped.shape[0]
    nc = batch_dropped.shape[2]

    # sample a dropping probability
    prob = torch.rand(1)*(b - a) + a

    mask = torch.rand((nj, nc)) > prob
    mask = mask.int().to(device)
    num_zeros_tensor = (mask == 0).sum().item()
    #print(num_zeros_tensor)
    batch_dropped = batch_dropped * mask.unsqueeze(1)

    #print(batch_dropped)
    pts = torch.sum( batch[:,0,:], axis=1 )
    pts_aug = torch.sum( batch_dropped[:,0,:], axis=1 )

    pTs= batch_dropped[:,0,:]
    
    if torch.any(pts_aug != 0):
        pt_rescale = torch.where(pts_aug != 0, pts / pts_aug, torch.ones_like(pts))
        #print(pt_rescale)
        pTs *= pt_rescale.unsqueeze(1)

    pTs, indices = torch.sort(pTs, dim=1, descending=True) # Ordering pTs
    etas = batch_dropped[:,1,:]
    etas = etas.gather(dim=1,index=indices)
    phis = batch_dropped[:,2,:]
    phis = phis.gather(dim=1,index=indices)

    
    batch_dropped = torch.cat((pTs.unsqueeze(1),etas.unsqueeze(1),phis.unsqueeze(1)),axis = 1)

    non_zero_count = torch.sum(pTs != 0, dim=-1, keepdim=True)
    #print(batch_dropped)
    if torch.min(non_zero_count)==0:
        return drop_constits_jet(batch,prob)
    else:
        return recentre_jet( batch_dropped )


def add_jets(batch):
    a = 1














#-------------------------------------_ExperimentCorner_---------------------------------------#

"""

def phi_rotate_jets(batch):

    rot_batch = batch # is this right when it should stay on gpu?
    rot_batch = rot_batch.to(device)
    batch_size = batch.size(0)
    constit = batch.size(2)

    rotate_tensor = torch.rand([batch_size,constit]) * 2 * np.pi #creating the array of random rotations
    rotate_tensor = rotate_tensor.to(device)

    rot_batch[:,2,:] =+ np.pi # shifting the phi tensor to make use of the % function
    rot_batch[:,2,:] += rotate_tensor
    rot_batch[:,2,:] %= 2 * np.pi # getting it in the same output range
    rot_batch[:,2,:] =- np.pi # shifting back

    return rot_batch
    
"""
def subjet_shift_jet( batch, nsubs=1, probs=[0.5], R=0.8 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    Dim 1 ordering: (pT, eta, phi)
    Output: batch of jets where a subset of constituents have been shifted
    Note: probs should be a vector of length nsubs, drop anything that falls outside the jet radius 
        and re-scale pts so that augmented jets have the same pTs as the originals,
        so choose sensible parameters in the function.
    '''
    device = batch.device
    batchc = batch.clone()
    nj = batchc.shape[0]
    nc = batchc.shape[2]
    eta_shifts = torch.stack( [ ( torch.rand( nj ) - 0.5 )/(1*R) for i in range(nsubs) ]).to(device)
    phi_shifts = torch.stack( [ ( torch.rand( nj ) - 0.5 )/(1*R) for i in range(nsubs) ]).to(device)
    masks = torch.stack( [ torch.tensor( torch.rand( nj, nc ) > probs[i], dtype=torch.int, device=device ) for i in range(nsubs) ]).to(device)
    for i in range( nj ):
        for j in range( nc ):
            for k in range( nsubs ):
                if torch.all( torch.tensor( [ masks[l][i][j] for l in range(k+1) ], device=device ) == 0 ):
                    batchc[i,1,j] = batchc[i,1,j] + eta_shifts[k][i]
                    batchc[i,2,j] = batchc[i,2,j] + phi_shifts[k][i]
    batchc = recentre_jet( batchc )
    for i in range( nj ):
        for j in range( nc ):
            if torch.sqrt( batchc[i,1,j]**2 + batchc[i,2,j]**2 ) > R:
                batchc[i,:,j] = torch.tensor( [0.0,0.0,0.0], device=device )
    pts = torch.sum( batch[:,0,:], dim=1 )
    pts_aug = torch.sum( batchc[:,0,:], dim=1 )
    pt_rescale = [ pts[i]/pts_aug[i] for i in range(nj) ]
    for i in range(nj):
        batchc[i,0,:] = batchc[i,0,:]*pt_rescale[i]
    batchc = recentre_jet( batchc )
    return batchc
