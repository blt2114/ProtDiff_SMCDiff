import torch
import tree
import numpy as np
import copy

def diffuse_motif(motif_ca_xyz, prot_diffuser, T):
    motif_forward_diffusion = [motif_ca_xyz]
    for t in range(T):
        x_t_1 = copy.copy(motif_forward_diffusion[-1])
        x_t, noise_t = prot_diffuser.ar_forward_diffusion(x_t_1, t)
        motif_forward_diffusion.append(x_t.float())  # [-1] is diffused, [0] is original structure
    return motif_forward_diffusion


def next_step_pred_with_motif(
    x_t, t, exp, motif_forward_diffusion,
    motif_idcs, inference_feats):

    B = x_t.shape[0]
    # force in diffused motif
    x_t = copy.copy(x_t)
    #import pdb; pdb.set_trace()
    x_t[:, motif_idcs] = motif_forward_diffusion[t][None]

    inference_feats['bb_corrupted'] = copy.copy(x_t)
    inference_feats['t'] = torch.tensor([t for _ in range(B)])
    inference_feats = tree.map_structure(lambda x: x.to(exp.device), inference_feats)
    inference_feats['bb_mask'] = torch.tile(
        inference_feats['bb_mask'][0],[B, 1])
    e_t = exp.model(inference_feats).cpu().detach().numpy()
    return e_t

def inpaint(
    N_samples, exp, target_len, motif_forward_diffusion,
    motif_idcs, prot_diffuser):
    T = exp.cfg.experiment.T
    inference_feats= {
            'bb_mask':torch.zeros([N_samples, target_len]),
            'residue_index':torch.tile(torch.arange(target_len),[N_samples,1])
            }
    inference_feats['bb_mask'][:, :target_len] = 1.

    # Sample x_T as random noise
    x_T = torch.Tensor(np.random.normal(
        size=list(inference_feats['bb_mask'].shape) + [3])).type(torch.float32)

    sampled_diffusion = [x_T]
    log_freq = 100
    for t in reversed(range(T)):
        x_t = sampled_diffusion[-1]

        e_t = next_step_pred_with_motif(
            x_t, t, exp, motif_forward_diffusion,
            motif_idcs, inference_feats)

        x_t_1 = prot_diffuser.ar_reverse_diffusion(x_t, e_t, t).type(torch.float32)
        x_t_1 *= inference_feats['bb_mask'][..., None].cpu()

        sampled_diffusion.append(x_t_1)
        if t % log_freq == (log_freq-1):
            print(f'On {t}')
    return sampled_diffusion

def motif_scaffolding_params(feats, motif_start, motif_end, target_len, pdb_name):
    """motif_scaffolding_params assembles key parameters for an inpainting task

    Args:
        feats: target features of the full scaffold from which to extract the motif
        motif_start, motif_end: indices of the start and end of the ranges of
            the residues attributed as the motif
        target_len: length_of_full proteins to generate
        pdb_name: name of the pdbs from which motif is extracted

    """
    # Define and extract motif
    motif_idcs = list(range(motif_start, motif_end+1))# + list(range(gap_end, target_len))
    full_ca_xyz_true = copy.deepcopy(feats['bb_positions'])

    # Set name of motif scaffolding task
    inpainting_task_name = pdb_name
    inpainting_task_name += "_length_%d"%target_len
    inpainting_task_name += "_motifRange%d_%d"%(motif_start, motif_end)

    return full_ca_xyz_true, motif_idcs, inpainting_task_name
