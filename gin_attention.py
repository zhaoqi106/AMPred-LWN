import torch
import torch.nn as nn
from torch_scatter import scatter


class GIN_AttProj(nn.Module):
    """PyTorch implementation of Global Attention Pooling used in AttentiveFP.

    This版本不依赖 DGL，直接基于 `torch_scatter` 对 PyG 的 batch 向量聚合，
    方便与 `GINet` 在同一 PyG 数据流里使用。
    """

    def __init__(self, node_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.feat_proj = nn.Linear(node_dim, out_dim)

    def forward(self, h_nodes: torch.Tensor, batch: torch.Tensor):
        """Compute graph-level embeddings.

        Parameters
        ----------
        h_nodes : Tensor(shape=[num_nodes, node_dim])
            Node embeddings from GIN.
        batch : Tensor(shape=[num_nodes,])
            Batch vector that maps each node to its graph index (PyG convention).
        Returns
        -------
        Tensor(shape=[batch_size, out_dim])
        """
        gate_scores = torch.sigmoid(self.gate_nn(h_nodes))  # (N,1)
        feat_proj = self.feat_proj(h_nodes)                 # (N,out_dim)
        gated_feat = gate_scores * feat_proj                # (N,out_dim)

        batch_size = int(batch.max().item() + 1)
        graph_embed = scatter(gated_feat, batch, dim=0, dim_size=batch_size, reduce='sum')
        return graph_embed 