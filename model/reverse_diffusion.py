import ml_collections
import torch
from torch import nn
from model import egnn_dense
from scipy import stats
from data import diffuser
import functools as fn

import numpy as np

pi = 3.141592653589793
Tensor = torch.Tensor
ConfigDict = ml_collections.ConfigDict


def index_embedding(residue_offsets, N, embed_size, device=None):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        residue_offsets: offsets of size [..., N_edges] of type integer
        N: number of residues
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    if device is None:
        device = 'cpu'
    K = torch.arange(embed_size//2).to(device)
    pos_embedding_sin = torch.sin(residue_offsets[..., None] *pi / (N**(2*K[None]/embed_size)))
    pos_embedding_cos = torch.cos(residue_offsets[..., None] *pi / (N**(2*K[None]/embed_size)))
    pos_embedding = torch.concat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


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
    pos_embedding_sin = torch.sin(idx[:,None] * pi / (N**(2*K[None]/embed_size)))
    pos_embedding_cos = torch.cos(idx[:,None] * pi / (N**(2*K[None]/embed_size)))
    pos_embedding = torch.concat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class Embedding(nn.Module):

    def __init__(self, N, T, pos_embed_size, output_embed_size, use_positional_encoding):
        super(Embedding, self).__init__()
        self.pos_embed_size = pos_embed_size
        self.use_positional_encoding = use_positional_encoding

        self.pos_embedding_N = fn.partial(
            index_embedding, N=N, embed_size=pos_embed_size)

        # Positional embedding for time step.
        # Make this second positional encoding independent of the
        # chain positional embedding
        pos_embedding_T = positional_embedding(T, pos_embed_size)
        R = stats.ortho_group.rvs(pos_embed_size)
        R = Tensor(R)
        pos_embedding_T = pos_embedding_T @ R
        self.register_buffer('pos_embedding_T', pos_embedding_T)


        #self.output_embedding = nn.Linear(
        #    pos_embed_size + 1, output_embed_size, bias=False)

    def forward(self, residue_indices, B, t, N, mask, device):
        """forward runs the embedding module on a batch of inputs.

        Args:
            B, t, N : batchsize, time step, chain size

        Returns:
            embeddings of dimension [Batch, T, N, self.embed_size*2]
        """
        embed_T = torch.tile(self.pos_embedding_T[t][:, None, :], [1, N, 1])
        if self.use_positional_encoding:
            residue_indices = residue_indices.reshape([B, -1])
            embed_N = self.pos_embedding_N(residue_indices, device=device)
            embed_N = embed_N.reshape([B, N, self.pos_embed_size])
            return embed_N + embed_T
        return embed_T


def get_config(debug=False):
    m = lambda x, y: y if debug else x
    cfg = ConfigDict({
        'pos_embed_size': 256,
        'h_embed_size': 256,
        'num_heads': 4,
        'num_layers': 4,
        'n_layers_per_egnn': 1,
        'e_embed_size': None,
        'network': 'egnn_dense',
        'use_positional_encoding': True,
        'use_rel_positional_encoding': True,
        'scale_eps': False,
    })
    return cfg


class ReverseDiffusionDense(nn.Module):

    def __init__(
            self,
            *,
            T,
            B,
            max_len,
            device,
            pos_embed_size,
            h_embed_size,
            num_layers,
            e_embed_size,
            n_layers_per_egnn,
            network,
            use_sequence_transformer,
            use_positional_encoding,
            use_rel_positional_encoding,
            b_0,
            b_T,
            scale_eps,
            ):
        super(ReverseDiffusionDense, self).__init__()
        self.T = T
        self.B = B
        self.max_len = max_len
        self.pos_embed_size = pos_embed_size
        self.h_embed_size = h_embed_size
        self.e_embed_size = e_embed_size if e_embed_size is not None else h_embed_size
        self.device = device
        self.network = network
        self.use_sequence_transformer = use_sequence_transformer
        self.use_positional_encoding = use_positional_encoding
        self.use_rel_positional_encoding = use_rel_positional_encoding
        self.scale_eps = scale_eps

        # self.node_embedding_layer
        self.embedding_layer = Embedding(
            max_len, T, self.pos_embed_size, h_embed_size, self.use_positional_encoding)
        self.edge_embedding_layer = fn.partial(
            index_embedding, N=max_len, embed_size=self.e_embed_size)
        self.diffuser = diffuser.Diffuser(T=T, b_0=b_0, b_T=b_T)
        self.cum_a_schedule = torch.Tensor(
            self.diffuser.cum_a_schedule).to(device)

        # num - layers
        self.layers = []
        for i in range(num_layers):
            layer = []
            egnn = egnn_dense.EGNN(
                in_node_nf=self.h_embed_size,
                hidden_nf=self.h_embed_size, # dimension of messages?
                out_node_nf=self.h_embed_size,
                in_edge_nf=self.e_embed_size if use_rel_positional_encoding else 0,
                n_layers=n_layers_per_egnn,
                normalize=True
            ).to(device)
            layer.append(egnn)
            layer.append(nn.LayerNorm(self.h_embed_size))
            self.layers.append(layer)

        self.layers_pytorch = nn.ModuleList([l for sublist in self.layers for l in sublist])

    def forward(self, input_feats):
        """forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            eps_theta_val: estimate of error for each step shape [B, N, 3]
        """
        # Scale protein positions to be on similar scale as noise distribution.
        bb_pos = input_feats['bb_corrupted'].type(torch.float32)# / 10.
        curr_pos = bb_pos.clone() # [B, N, D]
        bb_mask = input_feats['bb_mask'].type(torch.float32) # [B, N]
        bb_2d_mask = bb_mask[:, None, :] * bb_mask[:, :, None]
        bb_pos *= bb_mask[..., None]
        t = input_feats['t']
        B, N, _ = bb_pos.shape

        # Generate edge feature as embedding of residue offsets
        if self.use_rel_positional_encoding:
            res_index = input_feats['residue_index']
            edge_attr = res_index[:, :, None] - res_index[:, None, :]
            edge_attr = edge_attr.reshape([B, N**2])
            edge_attr = self.edge_embedding_layer(edge_attr, device=self.device)
            edge_attr = edge_attr.reshape([B, N, N, self.e_embed_size])
            assert edge_attr.shape[0] == B
            assert edge_attr.shape[1] == N
            assert edge_attr.shape[1] == edge_attr.shape[2]
        else:
            edge_attr = None

        # Node representations for first layer.
        H = self.embedding_layer(
            input_feats['residue_index'], B, t, N, bb_mask, device=self.device)
        for layer in self.layers:
            H *= bb_mask[..., None]
            curr_pos *= bb_mask[..., None]
            edge_attr *= bb_2d_mask[..., None]
            if len(layer) == 3:
                tfmr, egnn, norm = layer
                H = tfmr(H, src_key_padding_mask=1 - bb_mask)
            else:
                egnn, norm = layer
            H, curr_pos = egnn(H, curr_pos, edge_attr, mask=bb_mask)
            H *= bb_mask[..., None]
            H = norm(H)

        if self.scale_eps:
            cum_a_t = self.cum_a_schedule[t[:, None, None]]
            eps_theta_val = bb_pos - curr_pos * cum_a_t
            eps_theta_val = eps_theta_val / torch.sqrt(1 - cum_a_t)
            eps_theta_val = eps_theta_val * bb_mask[..., None]
        else:
            eps_theta_val = curr_pos - bb_pos
            eps_theta_val = eps_theta_val.reshape(bb_pos.shape)
            eps_theta_val = eps_theta_val * bb_mask[..., None]
        return eps_theta_val
