import numpy as np
import os
import re
from typing import List
from data import protein
from data import residue_constants
from scipy.spatial.transform import Rotation

CA_IDX = residue_constants.atom_order['CA']


def create_bb_prot(bb_pos: np.ndarray):
    assert bb_pos.ndim == 2
    assert bb_pos.shape[1] == 3
    n = bb_pos.shape[0]
    imputed_atom_pos = np.zeros([n, 37, 3])
    imputed_atom_pos[:, CA_IDX] = bb_pos
    imputed_atom_mask = np.zeros([n, 37])
    imputed_atom_mask[:, CA_IDX] = 1.0
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    b_factors = np.zeros([n, 37])
    aatype = np.zeros(n, dtype=np.int)
    return protein.Protein(
        atom_positions=imputed_atom_pos,
        atom_mask=imputed_atom_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors)


def write_prot_to_pdb(
        prot_pos: np.ndarray, file_path: str, overwrite=False, no_indexing=False):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if no_indexing:
        save_path = file_path.strip('.pdb') + '.pdb'
    else:
        save_path = file_path.strip('.pdb') + f'_{max_existing_idx+1}.pdb'
    with open(save_path, 'w') as f:
        if prot_pos.ndim == 3:
            for t, bb_pos in enumerate(prot_pos):
                bb_prot = create_bb_prot(bb_pos)
                pdb_prot = protein.to_pdb(bb_prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 2:
            bb_prot = create_bb_prot(prot_pos)
            pdb_prot = protein.to_pdb(bb_prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        f.write('END')
    return save_path


def rigids_to_se3_vec(frame, scale_factor=1.0):
    trans = frame[:, 4:] * scale_factor
    rotvec = Rotation.from_quat(frame[:, :4]).as_rotvec()
    se3_vec = np.concatenate([rotvec, trans], axis=-1)
    return se3_vec

