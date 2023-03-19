from torch import nn
import sys
import torch


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self,
                input_nf,
                output_nf,
                hidden_nf,
                edges_in_d=0,
                act_fn=nn.SiLU(),
                residual=True,
                normalize=False,
                coords_agg='mean',
                tanh=False,
                ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)


    def edge_model(self, h, radial, edge_attr):
        """edge_model creates edges features using node representations and
        edge features.

        Args:
            h: node features of shape [B, N, input_nf]
            radial: node distances [B, N, N, 1]
            edge_attr: [B, N, N, edges_in_d], with each input features

        Returns:
            messages for each edge of shape [B, N, N, hidden_nf]

        """
        B, N, _ = h.shape
        source = h[:, None].tile([1, N, 1, 1])
        target = h[:, :, None].tile([1, 1, N, 1])
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=3)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=3)
        out = out.reshape([B*N*N, -1])
        out = self.edge_mlp(out)
        out = out.reshape([B, N, N, -1])
        return out

    def node_model(self, h, edge_feat, mask):
        """node_model updates node features using messages from other nodes

        Args:
            h: node features of shape [B, N, input_nf]
            edge_feat: [B, N, N, hidden_nf], with each edge_feat[n,m] = message_{n,m}

        Returns:
            updated node features [B, N, output_nf]

        """
        B, N, _ = h.shape
        assert len(h.shape) == 3
        assert len(edge_feat.shape) == 4
        agg = torch.sum(edge_feat, axis=2)
        agg = torch.cat([h, agg], dim=-1)
        agg = agg * mask[:, :, None]
        agg = agg.reshape([B*N, -1])
        out = self.node_mlp(agg)
        out = out.reshape([B, N, -1])

        if self.residual:
            out = h + out
        return out

    def coord_model(self, coord, coord_diff, mask, edge_feat):
        """coord_model updates coordinates for all N nodes

        Args:
            coord: [B, N, D] coordinates of nodes
            coord_diff: [B, N, N, D] vector coordiate differences
            mask: [B, N] 1 if revealed, 0 if hidden
            edge_feat: [B, N, N, hidden_nf], with each edge_feat[n,m] = message_{n,m}

        Returns:
            updated coordinates of shape [B, N, D]
        """
        assert len(coord.shape) == 3
        assert len(coord_diff.shape) == 4
        assert len(mask.shape) == 2
        assert len(edge_feat.shape) == 4
        B, N, D = coord.shape

        mask_2d = mask[:, :, None] * mask[:, None, :]
        coord_diff = coord_diff.reshape([B, N**2, D])
        edge_feat = edge_feat.reshape([B, N**2, -1])
        embed_edge = self.coord_mlp(edge_feat)
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = mask_2d[..., None]*trans.reshape([B, N, N, D])

        if self.coords_agg == 'sum':
            agg = torch.sum(trans, axis=2)
        elif self.coords_agg == 'mean':
            # trans_agg = torch.sum(trans, axis=2)
            # mask_agg = torch.sum(mask_2d, axis=2, keepdim=True)
            agg = torch.sum(trans, axis=2) / (torch.sum(
                mask_2d, axis=2, keepdim=True) + 1e-10)
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, coord):
        """coord2radial returns the distances and vector displacements for each
        pair of coordinates.

        Args:
            coord: [B, N, D] coordinates of nodes

        Returns:
            radial: distances of shape [B, N, N, 1]
            coord_diff: (normalized?) displacements of shape [B, N, N, D]
        """
        coord_diff = coord[:, :, None] - coord[:, None]
        radial = torch.sum(coord_diff**2, axis=-1).unsqueeze(-1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, coord, mask=None, edge_attr=None, device='cpu'):
        """forward runs the equivariant convolutional layer forward

        Args:
            h: node features of shape [B, N, input_nf]
            coord: [B, N, D] coordinates of nodes
            edge_attr: [B, N, N, edges_in_d], with each input features
            mask: [B, N] 1 if revealed, 0 if hidden
        """
        if mask is None:
            mask = torch.ones(h.shape[:2]).to(device)
        coord *= mask[..., None]
        h *= mask[..., None]

        mask_2d = mask[:, :, None] * mask[:, None, :]
        radial, coord_diff = self.coord2radial(coord)
        radial *= mask_2d[..., None]
        coord_diff *= mask_2d[..., None]

        edge_feat = self.edge_model(h, radial, edge_attr)
        edge_feat *= mask_2d[..., None]
        coord = self.coord_model(coord, coord_diff, mask, edge_feat)
        coord *= mask[:, :, None]
        h = self.node_model(h, edge_feat, mask)
        h *= mask[:, :, None]
        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self,
                in_node_nf,
                hidden_nf,
                out_node_nf,
                in_edge_nf=0,
                device='cpu',
                act_fn=nn.SiLU(),
                n_layers=4,
                residual=True,
                normalize=False,
                tanh=False,
                ):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual,
                                                normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edge_attr, mask=None):
        """forward runs EGNN

        Args:
            h: node features of shape [B, N, input_nf]
            coord: [B, N, D] coordinates of nodes
            edge_attr: [B, N, N, edges_in_d], with each input features
            mask: [B, N] 1 if revealed, 0 if hidden
        """
        B, N, _ = h.shape
        h = h.reshape([B*N, -1])
        h = self.embedding_in(h)
        h = h.reshape([B, N, -1])

        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, x, mask=mask,
                    edge_attr=edge_attr, device=self.device)

        h = h.reshape([B*N, -1])
        h = self.embedding_out(h)
        h = h.reshape([B, N, -1])
        return h, x
