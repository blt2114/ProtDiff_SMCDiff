import copy
import torch
from data import utils as du
from inpainting import inpaint

def load_pdb_motif_problem(motif_start, motif_end, pdb_name, base_dir="./"):
    """

    """
    inpaint_example_dir = base_dir + "./pdbs/inpainting_targets/"
    fn = inpaint_example_dir +pdb_name+ "_1_A.pdb"
    feats = du.parse_pdb(pdb_name, fn, scale_factor=10)
    true_len = int(sum(feats['bb_mask']))
    target_len = true_len

    full_ca_xyz_true, motif_idcs, inpainting_task_name = inpaint.motif_scaffolding_params(
        feats, motif_start, motif_end, target_len, pdb_name)
    motif_ca_xyz = copy.deepcopy(feats['bb_positions'][motif_idcs])
    motif_ca_xyz = torch.tensor(motif_ca_xyz).float()

    return pdb_name, target_len, motif_ca_xyz, full_ca_xyz_true, motif_idcs, inpainting_task_name


def load_rsv_motif_problem(motif_start, motif_end, base_dir="./"):
    """
    Real Motif: residues 16-34 out of 62 total

    """
    pdb_name = "rsv"
    inpaint_example_dir = base_dir +"./pdbs/inpainting_targets/"
    fn = inpaint_example_dir + "rsv_Jue_design.pdb"
    feats = du.parse_pdb(pdb_name, fn, scale_factor=10)
    true_len = int(sum(feats['bb_mask']))
    target_len = true_len

    full_ca_xyz_true, motif_idcs, inpainting_task_name = inpaint.motif_scaffolding_params(
        feats, motif_start, motif_end, target_len, pdb_name)
    motif_ca_xyz = copy.deepcopy(feats['bb_positions'][motif_idcs])
    motif_ca_xyz = torch.tensor(motif_ca_xyz).float()

    return pdb_name, target_len, motif_ca_xyz, full_ca_xyz_true, motif_idcs, inpainting_task_name


def load_EFHand_motif_problem(motif_start, motif_end, motif2_start, motif2_end,
        base_dir="./"):
    """
    True motif: 1-4 and 31-43 of total 53 residues
    """
    pdb_name = "EFHand"
    inpaint_example_dir = base_dir + "./pdbs/inpainting_targets/"
    fn = inpaint_example_dir + "1PRW-EF-hand_1motif_98_rd2.pdb"
    feats = du.parse_pdb(pdb_name, fn, scale_factor=10)
    true_len = int(sum(feats['bb_mask']))
    target_len = true_len

    full_ca_xyz_true, motif_idcs, inpainting_task_name = inpaint.motif_scaffolding_params(
        feats, motif_start, motif_end, target_len, pdb_name)

    motif_idcs += list(range(motif2_start, motif2_end+1))
    inpainting_task_name += "_motif2Range%d_%d"%(motif2_start, motif2_end)

    motif_ca_xyz = copy.deepcopy(feats['bb_positions'][motif_idcs])
    motif_ca_xyz = torch.tensor(motif_ca_xyz).float()

    return pdb_name, target_len, motif_ca_xyz, full_ca_xyz_true, motif_idcs, inpainting_task_name
