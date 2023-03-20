"""Data utility functions"""
import dataclasses
import numpy as np
import tree
import collections
import os
import pickle
import string
import torch
from typing import List, Dict, Any
from data import parsers, residue_constants
from Bio import PDB
from torch.utils.data import Dataset, DataLoader
from data import protein
#from model.openfold.openfold.utils import rigid_utils
import scipy as sp

defaultdict = collections.defaultdict

PKG_DIR = os.path.dirname(os.path.dirname(__file__))  # hacky way to get the base dir of the protein_diffusion package

# Model features corresponding parsed PDB files.
CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index'
]
UNPADDED_FEATS = [
    't'
]

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

MMSEQS_CLUSTER_DIR = '/data/rsg/chemistry/jyim/large_data/pdb_clusters/30_08_2021/'

move_to_np = lambda x: x.cpu().detach().numpy()




def normalize(x):
    return x / (np.linalg.norm(x, axis=-1)[..., None] + 1e-8)


def parse_pdb(pdb_name: str, pdb_path: str, scale_factor=1., mean_center=True):
    """
    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        mean_center: whether to mean center atom positions.

    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {
        chain.id: chain
        for chain in structure.get_chains() if chain.id == 'A'}
    # TODO: Add logic for handling multiple chains.
    assert len(struct_chains) == 1

    chain_prot = parsers.process_chain(struct_chains['A'], 'A')
    chain_dict = dataclasses.asdict(chain_prot)

    # Process features
    feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
    return parse_chain_feats(
        feat_dict, scale_factor=scale_factor, mean_center=mean_center)


def parse_chain_feats(chain_feats, scale_factor=1., mean_center=True):
    ca_idx = residue_constants.atom_order['CA']
    bb_pos = chain_feats['atom_positions'][:, ca_idx]
    bb_center = np.mean(bb_pos, axis=0)
    centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
    chain_feats['bb_positions'] = chain_feats['atom_positions'][:, ca_idx]
    chain_feats['bb_mask'] = chain_feats['atom_mask'][:, ca_idx]
    return chain_feats


def pad_pdb_feats(raw_feats, max_len):
    padded_feats = {
        feat_name: pad(feat, max_len)
        for feat_name, feat in raw_feats.items() if feat_name not in UNPADDED_FEATS
    }
    for feat_name in UNPADDED_FEATS:
        padded_feats[feat_name] = raw_feats[feat_name]
    # Local frames need special handling.
    if 'bb_local_frames' in raw_feats:
        padded_feats['bb_local_frames'] = pad_local_frames(
            raw_feats['bb_local_frames'], max_len)
    return padded_feats


def pad(x: np.ndarray, max_len: int, pad_idx=0):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[0]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f'Invalid pad amount {pad_amt}')
    pad_widths[pad_idx] = (0, pad_amt)
    return np.pad(x, pad_widths)


def pad_local_frames(frames, max_len):
    seq_len = frames.shape[0]
    pad_amt = max_len - seq_len
    flat_padding = np.concatenate(
        [np.zeros(3), np.identity(3).reshape(-1)]).reshape(4, 3)
    padding = np.tile(flat_padding[None], [pad_amt, 1, 1])
    return np.concatenate([frames, padding], axis=0)


def concat_np_features(
        np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict


def write_pkl(save_path: str, pkl_data: Any):
    """Serialize data into a pickle file."""
    with open(save_path, 'wb') as handle:
        pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=False):
    """Read data from a pickle file."""
    with open(read_path, 'rb') as handle:
        try:
            return pickle.load(handle)
        except Exception as e:
            if verbose:
                print(f'Failed to read {read_path}')
            raise(e)


def create_protein_from_feats(prot_feats: Dict[str, np.ndarray], to_pdb=False):
    """Helper function to convert features to AF2 protein object.

    Args:
        prot_feats: Dict with protein features.
        to_pdb: whether to output protein as PDB string.

    Returns:
        Either protein object or PDB string of prot_feats.
    """
    prot_len = len(prot_feats['aatype'])
    bfactor = 1.0
    prot = protein.Protein(
        atom_positions=prot_feats['atom_positions'],
        aatype=prot_feats['aatype'],
        atom_mask=prot_feats['atom_mask'],
        residue_index=np.arange(prot_len),
        chain_index=prot_feats['chain_index'],
        b_factors=np.ones_like(prot_feats['atom_mask']) * bfactor
    )
    if to_pdb:
        return protein.to_pdb(prot)
    return prot


def write_pdb_from_feats(prot_feats, file_path):
    pdb_str = create_protein_from_feats(prot_feats, to_pdb=True)
    with open(file_path, 'w') as f:
        f.write(pdb_str)


def create_data_loader(
        torch_dataset: Dataset,
        batch_size,
        shuffle,
        num_workers=0,
        np_collate=False,
        prefetch_factor=2):
    """Creates a data loader with jax compatible data structures."""
    if np_collate:
        collate_fn = lambda x: concat_np_features(x, add_batch_dim=True)
    else:
        collate_fn = None
    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers)



def get_local_coordinates(bb_pos, bb_mask=None):
    if bb_pos.ndim != 2:
        raise ValueError(
            f'Expected [N, 3] shape for bb_pos but got {bb_pos.shape}')
    if bb_mask is not None and bb_mask.ndim != 1:
        raise ValueError(
            f'Expected [N] shape for bb_mask but got {bb_mask.shape}')

    if bb_mask is not None:
        bb_pos = bb_pos[np.where(bb_mask)]

    # Calculate direction
    bb_dir = normalize(bb_pos[None, :, :] - bb_pos[:, None, :])
    num_res = bb_pos.shape[0]
    x_i = np.arange(num_res)
    x_j = (np.tile([1], num_res) + x_i) % num_res
    x = bb_dir[(x_i, x_j)]

    y_i = np.arange(num_res)
    y_j = (np.tile([num_res - 1], num_res) + y_i) % num_res
    y = bb_dir[(y_i, y_j)]

    cx = normalize(np.cross(x, y))
    # Shape: [N, 4, 3] where 4 is for origin plus axes
    local_frames = np.stack([bb_pos, x, y, cx], axis=1)

    if bb_mask is not None:
        return pad_local_frames(local_frames, bb_mask.shape[0])
    return local_frames


def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int


def read_mmseq_clusters(cluster_level=30):
    cluster_path = os.path.join(
        MMSEQS_CLUSTER_DIR, f'bc-{cluster_level}.out')
    cluster_to_pdb_id = defaultdict(set)
    pdb_id_to_cluster = defaultdict(set)
    pdb_chain_id_to_cluster = defaultdict(set)
    with open(cluster_path, 'r') as cluster_file:
        for cluster_id, cluster_pdbs in enumerate(cluster_file):
            for pdb_chain in cluster_pdbs.strip('\n').split(' '):
                if '_' not in pdb_chain:
                    pdb_id = pdb_chain
                    chain_id = None
                else:
                    pdb_id, chain_id = pdb_chain.split('_')
                pdb_id = pdb_id.lower()
                cluster_to_pdb_id[cluster_id].add(pdb_id)
                pdb_id_to_cluster[pdb_id].add(cluster_id)
                if chain_id is not None:
                    chain_id = chain_str_to_int(chain_id)
                    pdb_chain_id_to_cluster[
                        f'{pdb_id}_{chain_id}'].add(cluster_id)
    return cluster_to_pdb_id, pdb_id_to_cluster, pdb_chain_id_to_cluster

def positional_embedding(N, embed_size):
    """positional_embedding creates sine / cosine positional embeddings as described
    in `Attention is all you need'

    Args:
        N: number of positions to embed
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    idx = torch.arange(N)
    K = torch.arange(embed_size//2)
    pos_embedding_sin = torch.sin(idx[:,None] * np.pi / (N**(2*K[None]/embed_size)))
    pos_embedding_cos = torch.cos(idx[:,None] * np.pi / (N**(2*K[None]/embed_size)))
    pos_embedding = torch.concat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding

def add_timestep(G, t):
    '''
    Add timestep information to a graph
    '''
    G.ndata['t'] = torch.full_like(G.nodes(), t)
    return G

def add_residue_index(G):
    '''
    Adds sequential residue indices to graph nodes
    '''
    device = G.device
    G.ndata['residue_index'] = torch.arange(G.num_nodes()).to(device)

    return G

def construct_rigid_frames(trans, device, rot=None):
    assert len(trans.shape) == 3
    batch_size, num_res, _ = trans.shape
    if rot is None:
        rand_rots = sp.spatial.transform.Rotation.random(
            batch_size * num_res).as_matrix().reshape(batch_size, num_res, 3, 3)
        rot = torch.Tensor(rand_rots).to(device)
    bb_frames = rigid_utils.Rigid(
        rots=rigid_utils.Rotation(rot_mats=rot),
        trans=trans)
    if device == 'cuda':
        bb_frames = bb_frames.cuda()
    return bb_frames


def construct_single_rigid_frames(trans, device, rot=None):
    num_res, _ = trans.shape
    if rot is None:
        rand_rots = sp.spatial.transform.Rotation.random(
            num_res).as_matrix()
        rot = torch.Tensor(rand_rots).to(device)
    bb_frames = rigid_utils.Rigid(
        rots=rigid_utils.Rotation(rot_mats=rot),
        trans=trans)
    if device == 'cuda':
        bb_frames = bb_frames.cuda()
    return bb_frames


def rigid_frames_from_all_atom(all_atom_pos):
    rigid_atom_pos = []
    for atom in ['N', 'CA', 'C']:
        atom_idx = residue_constants.atom_order[atom]
        atom_pos = all_atom_pos[..., atom_idx, :]
        rigid_atom_pos.append(atom_pos)
    return rigid_utils.Rigid.from_3_points(*rigid_atom_pos)


def rigid_transform_3D(A, B):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected


def frame_normalize(frame_tensor_7):
    """Zero-center translation and normalize quaternions."""
    frame_quats = frame_tensor_7[:, :4]
    frame_trans = frame_tensor_7[:, 4:]
    normalized_frame_trans = (
        frame_trans - np.mean(frame_trans, axis=0, keepdims=True))
    # normalized_frame_quats = frame_quats / np.linalg.norm(
    #     frame_quats, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        np.sum(normalized_frame_trans, axis=0), 0, atol=1e-3)
    # np.testing.assert_allclose(
    #     np.linalg.norm(normalized_frame_quats, axis=-1), 1., atol=1e-7)
    return np.concatenate([frame_quats, normalized_frame_trans], axis=-1)

    normalized_frame_tensor_7 = (
        frame_tensor_7 - np.mean(frame_tensor_7, axis=0, keepdims=True))
    np.testing.assert_allclose(
        np.sum(normalized_frame_tensor_7, axis=0), 0, atol=1e-3)
    return normalized_frame_tensor_7


def sample_inpaint_mask(percentage, num_masks, num_res, sample_mask_len=True):
    num_masked_res = int(num_res * percentage)
    num_masked_res = np.maximum(num_masked_res, num_masks)
    if sample_mask_len:
        mask_lens = np.random.choice(np.arange(num_masked_res), num_masks, replace=False)
    else:
        mask_lens = num_masked_res
    mask_starts = np.random.choice(np.arange(num_res), num_masks, replace=False)
    mask_ends = np.minimum(mask_starts + mask_lens.astype(int), num_res)
    final_mask = np.zeros(num_res)
    for i,j in zip(mask_starts, mask_ends):
        final_mask[i:j] = 1
    return final_mask

def create_bb_prot(model_pos):
    ca_idx = residue_constants.atom_order['CA']
    n = model_pos.shape[0]
    imputed_atom_pos = np.zeros([n, 37, 3])
    imputed_atom_pos[:, ca_idx] = model_pos
    imputed_atom_mask = np.zeros([n, 37])
    imputed_atom_mask[:, ca_idx] = 1.0
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    b_factors = np.zeros([n, 37])
    return protein.Protein(
      atom_positions=imputed_atom_pos,
      atom_mask=imputed_atom_mask,
      aatype=np.zeros(n, dtype=int),
      residue_index=residue_index,
      chain_index=chain_index,
      b_factors=b_factors)

def save_bb_as_pdb(bb_positions, fn):
    """save_bb_as_pdb saves generated c-alpha positions as a pdb file

    Args:
        bb_positions: c-alpha coordinates (before upscaling) of shape [seq_len, 3],
            not including masked residues

    """
    with open(fn, 'w') as f:
        # since trained on downscaled data, scale back up appropriately
        prot_pos = bb_positions*10.
        bb_prot = create_bb_prot(prot_pos)
        pdb_prot = protein.to_pdb(bb_prot, model=1, add_end=True)
        f.write(pdb_prot)

# Save coordinates as a PDB movie
def save_bb_as_pdb_movie(bb_positions, bb_mask, fn):
    """save_bb_as_pdb_movie saves generated c-alpha positions as a pdb file

    Args:
        bb_positions: c-alpha coordinates (before upscaling) of shape [B, seq_len, 3],
            not including masked residues
    """
    B = bb_positions.shape[0]

    # Make interpolation movie
    with open(fn, 'w') as f:
        for b in range(B):
            prot_pos = bb_positions[b]*10. # since trained on downscaled data, scale back up appropriately
            prot_pos  = tree.map_structure(
                lambda x: x[torch.where(bb_mask[b]!=0)],
            prot_pos)
            prot_pos -= np.mean(move_to_np(prot_pos), axis=0)[None]
            bb_prot = create_bb_prot(prot_pos)
            pdb_prot = protein.to_pdb(bb_prot, model=b+1, add_end=False)
            f.write(pdb_prot)
        f.write('END')
