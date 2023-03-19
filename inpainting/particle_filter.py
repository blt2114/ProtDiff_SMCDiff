import torch
import numpy as np
from inpainting import inpaint


def residual_resample(weights):
    """residual_resample samples from discrete distribution with probabilities `weights'
    trying to maintain diversity.

    Args:
        weights: simplex variable weights of shape [B]

    Returns:
        idcs of samples
    """
    B = len(weights)
    weights *= B/sum(weights)

    weights_floor = np.floor(weights)
    weights_remainder = weights - weights_floor
    idcs_no_replace = sum([[i]*int(w) for i, w in enumerate(weights_floor)], [])

    N_replace = sum(weights_remainder)
    N_replace = int(np.round(N_replace))
    idcs_replace = np.random.choice(B, size=N_replace, p=weights_remainder/sum(weights_remainder))
    idcs = idcs_no_replace + list(idcs_replace)
    return idcs, N_replace


def log_impt_weights(
    prot_diffuser, x_t, e_t, t, motif_forward_diffusion,
    motif_idcs):
    """computes log importance weights
    """
    if t ==0: return torch.zeros(x_t.shape[0])
    mu, sd = prot_diffuser.ar_reverse_diffusion_distribution(
        x_t, e_t, t)
    mu_M = mu[:, motif_idcs]
    x_t_1_m = motif_forward_diffusion[t-1]

    # compute un-normalized weighting factor for importance resampling step
    log_w = -(1./2)*(x_t_1_m-mu_M)**2/(sd**2)
    log_w = torch.sum(log_w, axis=[1,2])
    log_w -= torch.logsumexp(log_w, 0)
    return log_w

def inpaint_particle_filter(
    N_particles, exp, target_len, motif_forward_diffusion,
    motif_idcs, prot_diffuser, log_freq = 100):
    T = exp.cfg.experiment.T

    # characteristics of particle filtering trajectory & resampling
    ws, N_replace, resample_times = [], [], []

    # set mask to have correct batch dimension
    inference_feats= {
            'bb_mask':torch.zeros([N_particles, target_len]),
            'residue_index':torch.tile(torch.arange(target_len),[N_particles,1])
            }
    inference_feats['bb_mask'][:, :target_len] = 1.

    # initialize weights to ones.
    weights = np.ones([N_particles])

    # Sample x_T as random noise
    x_T = torch.Tensor(np.random.normal(
        size=list(inference_feats['bb_mask'].shape) + [3])).type(torch.float32)

    sampled_diffusion = [x_T]
    for t in reversed(range(T)):
        if t % log_freq == (log_freq-1): print(f'On {t}')

        # predict error with diffusion model
        x_t = sampled_diffusion[-1]
        e_t = inpaint.next_step_pred_with_motif(
            x_t, t, exp, motif_forward_diffusion,
            motif_idcs, inference_feats)

        # compute importance weights
        log_w = log_impt_weights(
            prot_diffuser, x_t, e_t, t, motif_forward_diffusion,
            motif_idcs)

        # Update Self-normalized importance weights
        weights = weights*torch.exp(log_w).cpu().detach().numpy()
        weights /= sum(weights) # Re-normalize
        ws.append(weights)

        # Residual resample, but only if
        #   (1) weights are sufficiently non-uniform, and
        #   (2)(optionally) not too close to end of the trajectory
        departure_from_uniform = np.sum(abs(N_particles*weights-1))
        #if (departure_from_uniform > 0.75*N_particles) and t > T//20: ()
        if departure_from_uniform > 0.75*N_particles:
            print(t, "resampling, departure=%0.02f"%departure_from_uniform)
            idcs, N_replace_t = residual_resample(weights)
            resample_times.append(t)
            N_replace.append(N_replace_t)

            # Apply resampling
            x_t, e_t = x_t[idcs], e_t[idcs]

            # Reset weights to uniform
            weights = np.ones_like(weights)/N_particles

        x_t_1 = prot_diffuser.ar_reverse_diffusion(x_t, e_t, t).type(torch.float32)
        x_t_1 *= inference_feats['bb_mask'][..., None].cpu()
        sampled_diffusion.append(x_t_1)

    return sampled_diffusion, N_replace, ws, resample_times
