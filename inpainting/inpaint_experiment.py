from data import utils as du
import numpy as np
from inpainting import inpaint, particle_filter

def run_inpainting(exp, target_len, motif_ca_xyz, motif_idcs, prot_diffuser, T, N_samples_per_diffusion,
                   inpainting_task_name, output_dir, inpaint_method, num_save=None):
    motif_forward_diffusion = inpaint.diffuse_motif(motif_ca_xyz, prot_diffuser, T)

    if num_save is None: num_save = N_samples_per_diffusion

    fn_base = output_dir + inpainting_task_name + "_" + inpaint_method
    if inpaint_method == "replacement":
        # independent inpaint with replacement method
        sampled_diffusion =  inpaint.inpaint(
            N_samples_per_diffusion, exp, target_len, motif_forward_diffusion,
            motif_idcs, prot_diffuser)
    elif inpaint_method == "particle":
        N_particles = N_samples_per_diffusion
        sampled_diffusion, N_replace, ws, resample_times = particle_filter.inpaint_particle_filter(
            N_particles, exp, target_len, motif_forward_diffusion,
            motif_idcs, prot_diffuser)
    else:
        assert inpaint_method == "fixed", "must be one of: " + ",".join(["replacement", "particle", "fixed"])
        motif_fixed = [motif_forward_diffusion[0]]*len(motif_forward_diffusion)
        # Independent inpaint with motif fixed at all t
        sampled_diffusion =  inpaint.inpaint(
            N_samples_per_diffusion, exp, target_len, motif_fixed,
            motif_idcs, prot_diffuser)

    # take last time step (t=0)
    sample_t0 = sampled_diffusion[-1]

    idcs = np.random.choice(N_samples_per_diffusion, size=num_save, replace=False)
    for j, idx in enumerate(idcs):
        fn = fn_base + "_sample_%02d.pdb"%j
        inpaint_sample = sample_t0[j]
        inpaint_sample = inpaint_sample[:target_len]
        du.save_bb_as_pdb(inpaint_sample, fn)
