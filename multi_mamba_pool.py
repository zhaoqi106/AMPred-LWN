import torch
import torch.nn as nn


class MultiHeadMambaPool(nn.Module):
    """Multi-head Mamba sequence pooling layer.

    Each head processes a separate slice of the channel dimension with its own
    stack of Mamba blocks; final graph embeddings from all heads are
    concatenated and layer-normalised.
    """

    def __init__(self, node_dim: int, out_dim: int, heads: int = 4, dropout_p: float = 0.1):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = heads
        self.d_head = out_dim // heads

        # Compatible import for both mamba-ssm >=2.3 and legacy versions
        try:
            from mamba_ssm.modules.mamba2 import MambaBlock  # type: ignore
        except ImportError:
            from mamba_ssm.modules.mamba2 import Mamba2 as MambaBlock  # type: ignore

        self.proj_in = nn.Linear(node_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)

        # Independent Mamba stack per head (2 blocks each)
        self.mamba_heads = nn.ModuleList([
            nn.Sequential(
                MambaBlock(d_model=self.d_head,
                           d_state=max(4, self.d_head // 8),
                           d_conv=4,
                           expand=4),
                MambaBlock(d_model=self.d_head,
                           d_state=max(4, self.d_head // 8),
                           d_conv=4,
                           expand=4),
            )
            for _ in range(heads)
        ])

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:  # (N_nodes, F)
        x = self.dropout(self.proj_in(x))  # (N, D)

        B = int(batch.max().item() + 1)
        lens = torch.bincount(batch, minlength=B)
        Lmax = int(lens.max().item())
        D = x.size(-1)

        padded = x.new_zeros(B, Lmax, D)
        idx = 0
        for b, L in enumerate(lens):
            padded[b, :L] = x[idx: idx + L]
            idx += L

        # (B, Lmax, heads, d_head)
        padded = padded.view(B, Lmax, self.heads, self.d_head)
        head_outs = []
        for h in range(self.heads):
            out_h = self.mamba_heads[h](padded[:, :, h, :])  # (B, Lmax, d_head)
            head_outs.append(out_h)
        seq_out = torch.cat(head_outs, dim=-1)  # (B, Lmax, D)

        mask = torch.arange(Lmax, device=x.device).unsqueeze(0) < lens.unsqueeze(1)
        summed = (seq_out * mask.unsqueeze(-1)).sum(dim=1)
        out = summed / lens.unsqueeze(-1)
        return self.norm(out)
