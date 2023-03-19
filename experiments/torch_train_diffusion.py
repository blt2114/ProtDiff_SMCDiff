"""Pytorch script for training protein diffusion.

Instructions:

To run:

> python torch_train_diffusion.py

"""

import os
import argparse
import copy
import torch
import getpass
from datetime import datetime
from typing import Dict
import time
import tree
import numpy as np
import GPUtil
import ml_collections

from data import pdb_dataset
from data import diffuser
from data import utils as du
from model import reverse_diffusion


ConfigDict = ml_collections.ConfigDict

move_to_np = lambda x: x.cpu().detach().numpy()


def gather_args():
    """Parses command line."""
    parser = argparse.ArgumentParser(prog="Protein diffusion experiment")
    parser.add_argument(
        "--disable_wandb",
        action='store_true',
        help="Debug mode. Turns off Wandb."
    )
    parser.add_argument(
        "--b_0",
        type=float, default=None,
        help="Override b_0 parameter."
    )
    parser.add_argument(
        "--b_T",
        type=float, default=None,
        help="Override b_T parameter."
    )
    parser.add_argument(
        "--plddt_filter",
        type=float, default=None,
        help="Override plddt_filter parameter."
    )
    parser.add_argument(
        "--tm_filter",
        type=float, default=None,
        help="Override tm_filter parameter."
    )
    parser.add_argument(
        "--rmsd_filter",
        type=float, default=None,
        help="Override rmsd_filter parameter."
    )
    parser.add_argument(
        "--batch_size",
        type=int, default=None,
        help="batch_size."
    )
    parser.add_argument(
        "--max_len",
        type=int, default=None,
        help="max_len."
    )
    parser.add_argument(
        "--num_layers",
        type=int, default=None,
        help="num_layers."
    )
    parser.add_argument(
        "--disable_positional_encoding",
        action='store_false',
        help="Disable positional encoding."
    )
    parser.add_argument(
        "--disable_relative_encoding",
        action='store_false',
        help="Disable relative positional encoding."
    )
    parser.add_argument(
        "--scale_eps",
        action='store_true',
        help="Enable epsilon scaling."
    )
    parser.add_argument(
        "--disable_monomer",
        action='store_false',
        help="Don't filter monomer."
    )
    parser.add_argument(
        "--inpainting_training",
        action='store_true',
        help="Enable inpainting training."
    )
    parser.add_argument(
        "--network",
        type=str, default=None,
        help="Override network parameter."
    )
    parser.add_argument(
        "--exp_name",
        type=str, default=None,
        help="Custom wandb experiment identifier."
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Debug mode. Uses smaller model."
    )
    return parser.parse_args()


def get_config(debug=False):
    m = lambda x, y: y if debug else x
    data_config = pdb_dataset.get_config()

    T = m(1024, 256)
    N = data_config['max_len']
    cfg = ConfigDict({
        'experiment': {
            'T': T,
            'batch_size': m(16, 2),
            'learning_rate': 1e-4,
            'train_steps': 1000000,
            'ckpt_freq': 100000,
            'ckpt_dir': '',
            'log_freq': 1000,
            'b_0': 0.0001,
            'b_T': 0.02,
        },
        'model': reverse_diffusion.get_config(debug=debug),
    })
    cfg['data'] = data_config
    return cfg


def write_checkpoint(
        ckpt_path: str,
        exp_state: Dict,
        log_data: Dict,
        exp_cfg: ConfigDict):
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        step: Experiment step at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
    """
    ckpt_state = {
        'exp_state': exp_state,
        'log_data': log_data,
        'cfg': exp_cfg
    }
    ckpt_dir = os.path.dirname(ckpt_path)
    for fname in os.listdir(ckpt_dir):
        os.remove(os.path.join(ckpt_dir, fname))
    print(f'Serializing experiment state to {ckpt_path}')
    du.write_pkl(ckpt_path, ckpt_state)


def flatten_cfg(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_cfg(v)
            ])
        else:
            flattened.append((k, v))
    return flattened


def t_stratified_loss(batch_t, batch_loss, batch_mask, T, bin_size=100):
    """Stratify loss by binning t."""
    flat_mask = batch_mask.flatten()
    flat_losses = batch_loss.flatten()[np.where(flat_mask)]
    flat_t = batch_t.flatten()[np.where(flat_mask)]
    num_bins = T // bin_size
    bin_edges = np.arange(num_bins + 2) * bin_size
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    for t_bin in np.unique(bin_idx).tolist():
        t_range = f'loss t=[{t_bin*bin_size},{(t_bin+1)*bin_size})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


class Experiment:
    # TODO: multi-gpu training

    def __init__(self, cfg):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """

        # Configs
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Model functions
        if self._data_cfg.crop_len is None:
            N = self._data_cfg.max_len
        else:
            N = self._data_cfg.crop_len
        self.max_len = N
        self._model = reverse_diffusion.ReverseDiffusionDense(
            T=self._exp_cfg.T,
            B=self._exp_cfg.batch_size,
            max_len=N,
            device=self.device,
            b_0=self._exp_cfg.b_0,
            b_T=self._exp_cfg.b_T,
            **self._model_cfg
        ).to(self.device)

        # Training objects
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._exp_cfg.learning_rate)

        self._diffuser = diffuser.Diffuser(
            T=self._exp_cfg.T,
            b_0=self._exp_cfg.b_0,
            b_T=self._exp_cfg.b_T,
        )

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def cfg(self):
        return self._cfg

    def update_fn(self, data, labels):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data, labels)
        loss.backward()
        self._optimizer.step()
        return loss, aux_data

    def loss_fn(self, batch, labels):
        """loss_fn computes the loss."""
        pred = self.model(batch)
        bb_mask = batch['bb_mask']
        t = batch['t'] # [B]

        # Kt_1 (K^{t-1}), of shape [B, N, N]
        errors = pred - labels # of shape [B, N, D]
        # NOTE: To account for 0 indexing vs 1 indexing, we subtract 1 from the
        # power here.  It should otherwise be K^(t-1) instead of K^t.
        Kt_1 = self._diffuser.K ** t
        Kt_1 = torch.Tensor(Kt_1).double().to(self.device)
        losses = torch.mean((
            torch.einsum('ijk,ikm->ijm', Kt_1, errors)
            )**2, dim=(-1,))
        aux_data = {
            'pred': pred,
            'losses': losses,
            'labels': labels
        }
        final_loss = torch.sum(losses) / (torch.sum(bb_mask) + 1e-10)
        return final_loss, aux_data

    def create_dataset(self, is_training):
        global_cfg = dict(
            diffusion_fn=self._diffuser.closed_form_forward_diffuse,
            return_labels=True,
            return_name=not is_training,
            T=self._exp_cfg.T,
        )
        dataset = pdb_dataset.PdbDataset(
            **global_cfg,
            **self._data_cfg)
        return dataset

    def sample_reverse_diffusion(
            self, bb_mask, log_freq=100, process=True, noise_scale=1.0):
        batch_size, seq_len = bb_mask.shape
        x_T = torch.Tensor(
            np.random.normal(size=[batch_size, seq_len, 3])).type(torch.float32)
        x_T *= bb_mask[..., None]
        sampled_diffusion = [x_T]
        sample_feats = {
            'bb_mask': torch.Tensor(bb_mask),
            'residue_index': torch.tile(torch.arange(seq_len), (batch_size, 1)),
        }
        for t in reversed(range(self._exp_cfg.T)):
            x_t = sampled_diffusion[-1]
            sample_feats['bb_corrupted'] = copy.copy(x_t)
            sample_feats['t'] = torch.tensor([t for _ in range(batch_size)])
            sample_feats = tree.map_structure(
                lambda x: x.to(self.device), sample_feats)

            e_t = move_to_np(self.model(sample_feats))

            x_t_1 = self.diffuser.ar_reverse_diffusion(
                x_t, e_t, t,  noise_scale=noise_scale).type(torch.float32)
            x_t_1 *= move_to_np(sample_feats['bb_mask'][..., None])

            sampled_diffusion.append(x_t_1)
            if t % log_freq == (log_freq-1):
                print(f'On {t}')
        # Shape=[B, T, N, 3]
        # B: batch size
        # T: diffusion length
        # N: sequence length
        sampled_diffusion = np.stack(sampled_diffusion, axis=0).swapaxes(0, 1)
        # Split batch dimension into list
        sampled_diffusion = [x[0] for x in np.split(sampled_diffusion, batch_size)]
        # return sampled_diffusion
        if process:
            sampled_bb_masks = [
                x[0] for x in np.split(bb_mask, batch_size)]
            sampled_diffusion = [
                self.process_bb_pos(
                    sampled_diffusion[i],
                    sampled_bb_masks[i])
                for i in range(batch_size)
            ]
        return sampled_diffusion

    def inpaint_reverse_diffusion(
            self, initial_pos, bb_mask, inpaint_mask,
            log_freq=100, process=True, noise_scale=1.0):
        assert self._model_cfg.inpainting_training, 'Inpaint training is False.'
        batch_size, seq_len = bb_mask.shape
        noise_T = np.random.normal(size=[batch_size, seq_len, 3])
        noise_T = noise_T * bb_mask[..., None] * inpaint_mask[..., None]
        x_T = initial_pos * (1 - inpaint_mask[..., None]) + noise_T * inpaint_mask[..., None]
        sampled_diffusion = [torch.Tensor(x_T)]
        sample_feats = {
            'bb_mask': torch.Tensor(bb_mask),
            'bb_corrupted_mask': torch.Tensor(inpaint_mask)
        }
        torch_inpaint_mask = torch.Tensor(inpaint_mask[..., None])
        for t in reversed(range(self._exp_cfg.T)):
            x_t = sampled_diffusion[-1]
            sample_feats['bb_corrupted'] = copy.copy(x_t)
            sample_feats['t'] = torch.tensor([t for _ in range(batch_size)])
            sample_feats = tree.map_structure(
                lambda x: x.to(self.device), sample_feats)

            e_t = move_to_np(self.model(sample_feats))
            x_t_1 = self.diffuser.ar_reverse_diffusion(
                x_t, e_t, t, mask=torch_inpaint_mask,
                noise_scale=noise_scale).type(torch.float32)
            x_t_1 *= move_to_np(sample_feats['bb_mask'][..., None])

            sampled_diffusion.append(x_t_1)
            if t % log_freq == (log_freq-1):
                print(f'On {t}')
        # Shape=[B, T, N, 3]
        # B: batch size
        # T: diffusion length
        # N: sequence length
        sampled_diffusion = np.stack(sampled_diffusion, axis=0).swapaxes(0, 1)
        # Split batch dimension into list
        sampled_diffusion = [x[0] for x in np.split(sampled_diffusion, batch_size)]
        # return sampled_diffusion
        if process:
            sampled_bb_masks = [
                x[0] for x in np.split(bb_mask, batch_size)]
            sampled_diffusion = [
                self.process_bb_pos(
                    sampled_diffusion[i],
                    sampled_bb_masks[i])
                for i in range(batch_size)
            ]
        return sampled_diffusion

    def process_bb_pos(self, bb_pos, bb_mask):
        def _process(x):
            x = x[np.where(bb_mask)]
            x_center = np.mean(x, axis=0)
            return (x - x_center) * self._data_cfg.scale_factor
        if bb_pos.ndim == 2:
            return _process(bb_pos)
        elif bb_pos.ndim == 3:
            bb_pos_processed = np.array([_process(bb_pos_b) for bb_pos_b in
                bb_pos])
            return bb_pos_processed
        else:
            raise ValueError(f'Unsupported bb_pos shape {bb_pos.shape}')


def run():
    exp_args = gather_args()
    user = getpass.getuser()

    # Set environment variables for which GPUs to use.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    chosen_gpu = ''.join(
        [str(x) for x in GPUtil.getAvailable(order='memory')])
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen_gpu
    print(f"Using GPUs: {chosen_gpu}")

    # Initialize experiment
    cfg = get_config(debug=exp_args.debug)
    exp_cfg = cfg.experiment
    if exp_args.b_0 is not None:
        exp_cfg.b_0 = exp_args.b_0
    if exp_args.b_T is not None:
        exp_cfg.b_T = exp_args.b_T
    if exp_args.batch_size is not None:
        exp_cfg.batch_size = exp_args.batch_size
    if exp_args.max_len is not None:
        cfg.data.max_len = exp_args.max_len
    if exp_args.network is not None:
        cfg.model.network = exp_args.network
    if exp_args.plddt_filter is not None:
        cfg.data.plddt_filter = exp_args.plddt_filter
    if exp_args.tm_filter is not None:
        cfg.data.tm_filter = exp_args.tm_filter
    if exp_args.rmsd_filter is not None:
        cfg.data.rmsd_filter = exp_args.rmsd_filter
    if exp_args.inpainting_training is not None:
        cfg.data.inpainting_training = exp_args.inpainting_training
    if exp_args.num_layers is not None:
        cfg.model.num_layers = exp_args.num_layers

    cfg.model.scale_eps = exp_args.scale_eps
    cfg.model.use_positional_encoding = exp_args.disable_positional_encoding
    cfg.model.use_rel_positional_encoding = exp_args.disable_relative_encoding
    cfg.data.monomer_only = exp_args.disable_monomer

    exp = Experiment(cfg)
    print('Initializing experiment')
    train_dataset = exp.create_dataset(True)
    train_data_pipeline = du.create_data_loader(
        train_dataset,
        np_collate=False,
        batch_size=exp_cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f'Number of train examples: {len(train_dataset)}')
    num_parameters = sum(p.numel() for p in exp.model.parameters())
    print(f'Number of parameters {num_parameters}')

    # Checkpoint
    dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
    ckpt_dir = os.path.join(
        exp_cfg.ckpt_dir, 'torch_train_diffusion', dt_string)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    # Set-up remote experiment manager.
    wandb_enabled = not exp_args.disable_wandb
    if wandb_enabled:
        scratch_dir = f'/tmp/{user}/wandb'
        os.makedirs(scratch_dir, exist_ok=True)
        flat_cfg = dict(flatten_cfg(cfg.to_dict()))
        flat_cfg['ckpt_dir'] = ckpt_dir
        wandb.init(
            config=flat_cfg,
            project='[Protein diffusion]',
            dir=scratch_dir,
            name=exp_args.exp_name
        )

    # Run training
    periodic_logs = []
    log_lossses = []
    if not exp_args.disable_wandb:
        log_freq = exp_cfg.log_freq
    else:
        log_freq = 10
    ckpt_freq = exp_cfg.ckpt_freq
    start_time = time.time()
    step_time = time.time()
    step = 0
    num_epochs = 0
    log_data = {}
    device = exp.device
    exp.model.train()
    while step < exp_cfg.train_steps:
        num_epochs += 1
        for (train_features, train_labels) in train_data_pipeline:
            train_features = tree.map_structure(
                lambda x: x.to(device), train_features)
            train_labels = train_labels.to(device) * train_features['bb_mask'][..., None]

            step += 1
            loss, aux_data = exp.update_fn(
                train_features, train_labels)
            log_lossses.append(move_to_np(loss))

            # Log results
            if step % log_freq == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                step_per_sec = log_freq / elapsed_time
                ms_per_step = 1000 / step_per_sec
                rolling_loss = np.mean(log_lossses)
                print(f'[{step+1}]: loss={rolling_loss:.5f}, steps/sec={step_per_sec:.5f}, ms/step={ms_per_step:.2f}')
                log_lossses = []
                log_data = {
                    'avg_loss': rolling_loss,
                    'aux_data': aux_data,
                    'steps_per_sec': step_per_sec,
                    'num_epochs': num_epochs
                }
                periodic_logs.append((step, log_data))

            # Remote log to Wandb
            if wandb_enabled:
                step_time = time.time() - step_time
                example_per_sec = exp_cfg.batch_size / step_time
                batched_t = torch.ones_like(
                    train_features['bb_mask']) * train_features['t'][:, None]
                stratified_loss = t_stratified_loss(
                    move_to_np(batched_t),
                    move_to_np(aux_data['losses']),
                    move_to_np(train_features['bb_mask']),
                    exp_cfg.T)
                wandb.log(
                    {
                        'loss': loss,
                        'examples_per_sec': example_per_sec,
                        'num_epochs': num_epochs,
                        **stratified_loss
                    },
                    step=step)
                step_time = time.time()

            # Take checkpoint
            if step == 1 or step % ckpt_freq == 0:
                # Save first checkpoint to catch checkpoint bugs sooner.
                ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{step}.pkl')
                write_checkpoint(
                    ckpt_path, exp.model.state_dict(), periodic_logs, cfg)
    print('Finished')


if __name__ == '__main__':
    run()
