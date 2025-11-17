import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features=65, out_features=65, dropout=0.5, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """Batched forward pass.

        Parameters
        ----------
        h : Tensor
            Node features. Shape can be (N, F) for a single graph or (B, N, F) for a
            batch of graphs where B is batch size, N is number of nodes (padded to
            same size), F is in_features.
        adj : Tensor
            Adjacency matrices in dense form. Shape must align with *h*:
            (N, N) for a single graph or (B, N, N) for batch.
        """

        single_input = False
        if h.dim() == 2:
            # (N, F) -> (1, N, F)
            h = h.unsqueeze(0)
            adj = adj.unsqueeze(0)
            single_input = True

        # h: (B, N, F_in)
        B, N, _ = h.shape

        # Linear transform: (B, N, F_out)
        Wh = torch.matmul(h, self.W)  # self.W: (F_in, F_out)

        # Prepare attention scores
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]).squeeze(-1)  # (B, N)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :]).squeeze(-1)  # (B, N)
        e = Wh1.unsqueeze(-1) + Wh2.unsqueeze(-2)  # (B, N, N)
        e = self.leakyrelu(e)

        # Masked attention: only consider edges where adj > 0
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Weighted sum of neighbor features
        h_prime = torch.matmul(attention, Wh)  # (B, N, F_out)

        if self.concat:
            h_prime = F.elu(h_prime)

        # If original input was single graph, squeeze batch dimension back
        if single_input:
            h_prime = h_prime.squeeze(0)

        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # NOTE: This helper is no longer used in the batched implementation but is
        # kept for backward compatibility in case other code calls it directly.
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
