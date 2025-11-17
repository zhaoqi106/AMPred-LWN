import importlib.util
import os
import torch
import torch.nn as nn

# 动态导入原实现文件 `5_5_ConBiMamba.py`（文件名以数字开头无法直接 import）
_spec = importlib.util.spec_from_file_location(
    "_conbi_module",
    os.path.join(os.path.dirname(__file__), "5_5_ConBiMamba.py"),
)
_conbi_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conbi_module)

# 暴露包装类供外部引用
_BaseBlock = _conbi_module.ConbimambaBlock

class ConBiMambaBlock(nn.Module):
    """Thin wrapper around ConbimambaBlock.

    Exposes a unified ``dropout_p`` 参数；该值会同时用于 feed-forward、attention
    与 conv 模块的 dropout。其它参数与旧实现保持一致。
    """

    def __init__(self, dim: int, ff_mult: int = 4, reduction: int = 2, heads: int = 8,
                 dropout_p: float = 0.1):
        super().__init__()
        self.block = _BaseBlock(
            encoder_dim=dim,
            feed_forward_expansion_factor=ff_mult,
            conv_expansion_factor=reduction,
            num_attention_heads=heads,
            feed_forward_dropout_p=dropout_p,
            attention_dropout_p=dropout_p,
            conv_dropout_p=dropout_p,
        )

    def forward(self, fine_messages, coarse_messages, motif_features):
        """将三条信息流拼接为一个长度为 3 的序列后送入 ConbimambaBlock。

        输入形状： (B, C)
        拼接后： (B, 3, C)
        输出对时序维做平均池化得到 (B, C)
        """
        # (B,C) -> (B,3,C)
        seq = torch.stack([fine_messages, coarse_messages, motif_features], dim=1)

        # (B,3,C) -> (B,3,C)
        seq_out = self.block(seq)

        # 平均池化得到单向量表示 (B,C)
        return seq_out.mean(dim=1) 