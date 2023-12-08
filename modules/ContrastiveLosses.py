import torch
import torch.nn.functional as F


def clr_loss( x_i, x_j, temperature ):
    xdevice = x_i.get_device()
    batch_size = x_i.shape[0]
    z_i = F.normalize( x_i, dim=1 ) # no augs
    z_j = F.normalize( x_j, dim=1 ) # phys augs
    z   = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives = torch.cat( [sim_ij, sim_ji], dim=0 )
    nominator = torch.exp( positives / temperature )
    negatives_mask = ( ~( torch.diag( torch.ones(2*batch_size) ) + torch.diag( torch.ones(batch_size),batch_size ) + torch.diag( torch.ones(batch_size),-batch_size ) > 0 ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )
    loss_partial = -torch.log( nominator / torch.sum( denominator, dim=1 ) )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss, nominator, denominator


def anomclr_loss( x_i, x_j, x_k, temperature ):
    xdevice = x_i.get_device()
    batch_size = x_i.shape[0]
    z_i = F.normalize( x_i, dim=1 ) # no augs
    z_j = F.normalize( x_j, dim=1 ) # phys augs
    z_k = F.normalize( x_k, dim=1 ) # anom augs
    z   = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives = torch.cat( [sim_ij, sim_ji], dim=0 ).clone()
    negatives_mask = ( ~( torch.diag( torch.ones(2*batch_size) ) + torch.diag( torch.ones(batch_size),batch_size ) + torch.diag( torch.ones(batch_size),-batch_size ) > 0 ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )
    zn = torch.cat( [z_i,z_k], dim=0 )
    similarity_matrix = F.cosine_similarity( zn.unsqueeze(1), zn.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives_n = torch.cat( [sim_ij, sim_ji], dim=0 ).clone()
    nominator = torch.exp( (positives-positives_n) / temperature )
    loss_partial = -torch.log( nominator / ( torch.sum( denominator, dim=1 ) ) )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss

def anomclr_plus_loss( x_i, x_j, x_k, temperature ):
    xdevice = x_i.get_device()
    batch_size = x_i.shape[0]
    z_i = F.normalize( x_i, dim=1 ) # no augs
    z_j = F.normalize( x_j, dim=1 ) # phys augs
    z_k = F.normalize( x_k, dim=1 ) # anom augs
    z   = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives = torch.cat( [sim_ij, sim_ji], dim=0 ).clone()
    negatives_mask = ( ~( torch.diag( torch.ones(2*batch_size) ) + torch.diag( torch.ones(batch_size),batch_size ) + torch.diag( torch.ones(batch_size),-batch_size ) > 0 ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    zn = torch.cat( [z_i,z_k], dim=0 )
    similarity_matrix = F.cosine_similarity( zn.unsqueeze(1), zn.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives_n = torch.cat( [sim_ij, sim_ji], dim=0 ).clone()
    nominator = torch.exp( (positives-positives_n) / temperature )
    loss_partial = -torch.log( nominator )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss

def anomclr_plus_loss_bonus( x_i, x_j, x_k, temperature ):
    xdevice = x_i.get_device()
    batch_size = x_i.shape[0]
    z_i = F.normalize( x_i, dim=1 ) # no augs
    z_j = F.normalize( x_j, dim=1 ) # phys augs
    z_k = F.normalize( x_k, dim=1 ) # anom augs
    z   = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives = torch.cat( [sim_ij, sim_ji], dim=0 ).clone()
    negatives_mask = ( ~( torch.diag( torch.ones(2*batch_size) ) + torch.diag( torch.ones(batch_size),batch_size ) + torch.diag( torch.ones(batch_size),-batch_size ) > 0 ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    zn = torch.cat( [z_i,z_k], dim=0 )
    similarity_matrix = F.cosine_similarity( zn.unsqueeze(1), zn.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives_n = torch.cat( [sim_ij, sim_ji], dim=0 ).clone()
    nominator = torch.exp( (positives-positives_n) / temperature )
    loss_partial = -torch.log( nominator )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss , positives_n , positives

def anomclr_plus_loss_bonus_scalars( x_i, x_j, x_k, temperature, scalar_positives =1, scalar_augmented = 1 ):
    xdevice = x_i.get_device()
    batch_size = x_i.shape[0]
    z_i = F.normalize( x_i, dim=1 ) # no augs
    z_j = F.normalize( x_j, dim=1 ) # phys augs
    z_k = F.normalize( x_k, dim=1 ) # anom augs
    z   = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives = torch.cat( [sim_ij, sim_ji], dim=0 ).clone()
    negatives_mask = ( ~( torch.diag( torch.ones(2*batch_size) ) + torch.diag( torch.ones(batch_size),batch_size ) + torch.diag( torch.ones(batch_size),-batch_size ) > 0 ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    zn = torch.cat( [z_i,z_k], dim=0 )
    similarity_matrix = F.cosine_similarity( zn.unsqueeze(1), zn.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives_n = torch.cat( [sim_ij, sim_ji], dim=0 ).clone()
    # new implemented Try
    positives_n = positives_n *scalar_augmented
    positives = positives * scalar_positives
    # done 
    nominator = torch.exp( (positives-positives_n) / temperature )
    loss_partial = -torch.log( nominator )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss , positives_n , positives


def anomclr_loss_bonus( x_i, x_j, x_k, temperature ):
    xdevice = x_i.get_device()
    batch_size = x_i.shape[0]
    z_i = F.normalize( x_i, dim=1 ) # no augs
    z_j = F.normalize( x_j, dim=1 ) # phys augs
    z_k = F.normalize( x_k, dim=1 ) # anom augs
    z   = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives = torch.cat( [sim_ij, sim_ji], dim=0 ).clone()
    negatives_mask = ( ~( torch.diag( torch.ones(2*batch_size) ) + torch.diag( torch.ones(batch_size),batch_size ) + torch.diag( torch.ones(batch_size),-batch_size ) > 0 ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )
    zn = torch.cat( [z_i,z_k], dim=0 )
    similarity_matrix = F.cosine_similarity( zn.unsqueeze(1), zn.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives_n = torch.cat( [sim_ij, sim_ji], dim=0 ).clone() #non physical
    nominator = torch.exp( (positives-positives_n) / temperature )
    loss_partial = -torch.log( nominator / ( torch.sum( denominator, dim=1 ) ) )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss , positives_n , positives , denominator
