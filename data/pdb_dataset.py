import os
import ml_collections
import numpy as np
import pandas as pd
import torch
import random

from torch.utils.data import Dataset
from data import utils as du


def get_config(config_override=None, debug=False) -> ml_collections.ConfigDict:
    m = lambda x, y: y if debug else x
    cfg = ml_collections.ConfigDict({
        'pdb_csv_path': '',
        'pdb_self_consistency_path': '',

        'max_len': 128,
        'min_len': 40,
        'crop_len': None,

        'plddt_filter': None,
        'tm_filter': None,
        'rmsd_filter': None,

        'cropping_mode': 'contiguous_sequence',
        'max_chain_filter': 1,
        'monomer_only': True,
        'scale_factor': 10.,
        'inpainting_training': False,
        'inpainting_percentage': [0.1, 0.5],
        'num_inpaint_masks': [1, 2],
    })
    if config_override is not None:
        cfg.update(config_override)
    return cfg


class PdbDataset(Dataset):
    def __init__(
                self,
                 *,
                pdb_csv_path,
                pdb_self_consistency_path,
                plddt_filter,
                tm_filter,
                rmsd_filter,
                max_len,
                min_len,
                monomer_only,
                diffusion_fn,
                T,
                crop_len,
                cropping_mode,
                max_chain_filter,
                scale_factor,
                inpainting_training,
                inpainting_percentage,
                num_inpaint_masks,
                pdb_name_allowlist=None,
                return_labels=True,
                return_name=False):
        # Read and process CSV
        pdb_csv = pd.read_csv(pdb_csv_path)
        pdb_self_consistency = pd.read_csv(pdb_self_consistency_path)

        missing_pdb = set(pdb_self_consistency.pdb_name) - set(pdb_csv.pdb_name)
        if len(missing_pdb):
            raise Exception(f'Missing PDBs {missing_pdb}')

        self.raw_csv = pdb_csv
        self.sc_csv = pdb_self_consistency
        if max_chain_filter is not None:
            pdb_csv = pdb_csv[pdb_csv.num_chains <= max_chain_filter]
        if max_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= max_len]
        if min_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= min_len]
            pdb_csv = pdb_csv[pdb_csv.seq_len >= min_len]
        if monomer_only:
            pdb_csv = pdb_csv[pdb_csv.oligomeric_detail == 'monomeric']
        if pdb_name_allowlist is not None:
            pdb_csv = pdb_csv[pdb_csv.pdb_name.isin(pdb_name_allowlist)]
            assert len(pdb_csv) == len(pdb_name_allowlist)
        if plddt_filter is not None or rmsd_filter is not None or tm_filter is not None:
            pdb_csv = pd.merge(
                pdb_self_consistency, pdb_csv,
                left_on='pdb_name', right_on='pdb_name')
        if plddt_filter is not None:
            pdb_csv = pdb_csv[pdb_csv.min_plddt > plddt_filter]
        if rmsd_filter is not None:
            pdb_csv = pdb_csv[pdb_csv.min_rmsd < rmsd_filter]
        if tm_filter is not None:
            pdb_csv = pdb_csv[pdb_csv.max_tm_score > tm_filter]
        self.csv = pdb_csv
        assert len(pdb_csv)

        # PDB processing parameters
        self.crop_len = crop_len
        self.cropping_mode = cropping_mode
        self.scale_factor = scale_factor
        self.max_len = max_len
        self.return_labels = return_labels
        self.return_name = return_name

        # Diffusion parameters
        self.diffusion_fn = diffusion_fn
        self.T = T

        self.inpainting_training = inpainting_training
        self.inpainting_percentage = inpainting_percentage
        self.num_inpaint_masks = num_inpaint_masks

    def _process_csv_row(self, csv_row):
        pdb_name = csv_row['pdb_name']
        processed_file_path = csv_row['processed_path']
        chain_feats = du.read_pkl(processed_file_path)
        chain_feats = du.parse_chain_feats(
            chain_feats, scale_factor=self.scale_factor)
        modeled_idx = chain_feats['modeled_idx']
        del chain_feats['modeled_idx']
        chain_feats = jax.tree_map(
            lambda x: x[(modeled_idx,)], chain_feats)
        if self.crop_len:
            if self.cropping_mode == 'contiguous_sequence':
                seq_len = chain_feats['aatype'].shape[0]
                if seq_len <= self.crop_len:
                    start_idx = 0
                else:
                    start_idx = random.randrange(seq_len - self.crop_len)
                chain_feats = jax.tree_map(
                    lambda x: x[start_idx:start_idx+self.crop_len], chain_feats)
            else:
                raise ValueError(
                    f'Unrecognized cropping mode {self.cropping_mode}')
        return chain_feats, pdb_name


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        csv_row = self.csv.iloc[idx]
        chain_feats, pdb_name = self._process_csv_row(csv_row)
        bb_mask = chain_feats['bb_mask'][..., None]
        num_res = bb_mask.shape[0]
        if self.inpainting_training:
            p0, p1 = self.inpainting_percentage
            inpaint_percent = p0 + np.random.random() * (p1 - p0)
            num_masks = np.random.choice(np.arange(
                self.num_inpaint_masks[0], self.num_inpaint_masks[1]+1),
                replace=False)
            inpaint_mask = du.sample_inpaint_mask(
                inpaint_percent, num_masks, num_res)
            chain_feats['bb_corrupted_mask'] = inpaint_mask
            noise_fn = lambda x: x * inpaint_mask[:, None]
        else:
            chain_feats['bb_corrupted_mask'] = bb_mask
            noise_fn = None

        t = np.random.choice(self.T, 1)[0]
        bb_corrupted, bb_noise = self.diffusion_fn(
            chain_feats['bb_positions'], t, noise_fn=noise_fn)
        bb_mask = chain_feats['bb_mask'][..., None]
        chain_feats['bb_corrupted'] = bb_corrupted * bb_mask
        chain_feats['bb_noise'] = bb_noise * bb_mask
        chain_feats['t'] = t
        # Pad residue dimension so all examples are same length
        if self.crop_len is None:
            pad_len = self.max_len
        else:
            pad_len = self.crop_len
        chain_feats = du.pad_pdb_feats(
            chain_feats, pad_len)

        ret = [chain_feats]
        if self.return_labels:
            ret.append(chain_feats['bb_noise'])
        if self.return_name:
            ret.append(pdb_name)
        if len(ret) == 1:
            return ret[0]
        return ret
