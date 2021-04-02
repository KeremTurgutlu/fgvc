from fastai.vision.all import *

def generate_batch_attention_maps(attn_wgts, targ_sz, mode='bilinear'):
    "Generate attention flow maps with shape (targ_sz,targ_sz) from L layer attetion weights of transformer model"
    # Stack for all layers - BS x L x K x gx x gy
    att_mat = torch.stack(attn_wgts, dim=1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
    # To account for residual connections, we add an identity matrix to the
    aug_att_mat = att_mat + torch.eye(att_mat.size(-1))[None,None,...].to(att_mat.device)
    # Re-normalize the weights.
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = aug_att_mat[:,0]
    for n in range(1, aug_att_mat.size(1)): joint_attentions = torch.bmm(aug_att_mat[:,n], joint_attentions)

    # BS x (num_patches+1) -> BS x gx x gy
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    joint_attentions = joint_attentions[:,0,1:].view(joint_attentions.size(0),grid_size,grid_size)
    joint_attentions /= torch.amax(joint_attentions, dim=(-2,-1), keepdim=True)

    # Bilinear interpolation to target size
    if mode == 'bilinear':
        joint_attentions = F.interpolate(joint_attentions[None,...], 
                                         (targ_sz,targ_sz), 
                                         mode=mode, align_corners=True)[0].detach().cpu().numpy()
    elif mode == 'nearest':
        joint_attentions = F.interpolate(joint_attentions[None,...], 
                                         (targ_sz,targ_sz), 
                                         mode=mode)[0].detach().cpu().numpy()
    elif mode is None:
        joint_attentions = joint_attentions.detach().cpu().numpy()
    
    return joint_attentions