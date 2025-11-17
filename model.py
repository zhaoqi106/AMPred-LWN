import torch
import torch.nn as nn
import dgl
import math
import os

from torch.nn import Linear

from layer import GraphAttentionLayer
import torch.nn.functional as F
from .MLP import MLP
from .embedding import LinearBn
from .layers import DGL_MPNNLayer
from .readout import Readout
# New: ConBiMambaBlock (efficient attention replacement)
from .conbi_mamba_block import ConBiMambaBlock
# ------------- Mamba-2 import for motif sequence modeling -------------
try:
    from mamba_ssm.modules.mamba2 import MambaBlock  # >=2.3
except ImportError:
    from mamba_ssm.modules.mamba2 import Mamba2 as MambaBlock
import torch.nn.functional as F
from models.ginet_finetune import GINet
from models.gin_attention import GIN_AttProj

class MambaPool(nn.Module):
    """Sequence-to-vector summarisation using a single ConBiMambaBlock followed by mean-pool.

    Parameters
    ----------
    node_dim : int
        Input feature dimension of each node.
    out_dim : int
        Dimension of the graph-level embedding after pooling.
    heads : int
        Multi-head parameter passed to ConBiMambaBlock for internal projection.
    """

    def __init__(self, node_dim: int, out_dim: int, heads: int = 4, dropout_p: float = 0.1):
        super().__init__()
        # 兼容新版 mamba-ssm：>=2.3 只暴露 `Mamba2` 而没有 `MambaBlock`
        try:
            from mamba_ssm.modules.mamba2 import MambaBlock  # 新旧版本兼容
        except ImportError:
            from mamba_ssm.modules.mamba2 import Mamba2 as MambaBlock

        self.proj_in = nn.Linear(node_dim, out_dim, bias=False)
        # Add dropout to improve regularization on small datasets
        self.dropout = nn.Dropout(p=dropout_p)  # fixed 10%; adjust if needed

        # 使用单序列 MambaBlock；d_state & d_conv 取安全默认
        # 注意：新版 Mamba2 的 `expand` 表示乘数而非绝对维度，这里取 2×d_model，
        # 同时避免产生 >64K 的 gating 维度。
        self.mamba = nn.Sequential(
            MambaBlock(d_model=out_dim, d_state=max(4, out_dim // 8), d_conv=4, expand=4),
            MambaBlock(d_model=out_dim, d_state=max(4, out_dim // 8), d_conv=4, expand=4),
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """x: (N_nodes, node_dim); batch: (N_nodes,) graph id per node→ (B, out_dim)"""
        x = self.dropout(self.proj_in(x))  # (N_nodes, D)

        # ----------- Batch padding -------------
        # 假设 batch 是从 0 开始的连续整数索引 (PyG / DGL 均满足)
        B = batch.max().item() + 1
        lens = torch.bincount(batch, minlength=B)             # (B,)
        max_L = int(lens.max().item())

        D = x.size(-1)
        padded = x.new_zeros(B, max_L, D)

        idx = 0
        for g in range(B):
            L = int(lens[g].item())
            padded[g, :L] = x[idx : idx + L]
            idx += L

        # ----------- Mamba & masked mean -------
        seq_out = self.mamba(padded)                              # (B, max_L, D)

        mask = (torch.arange(max_L, device=x.device)
                 .unsqueeze(0) < lens.unsqueeze(1))               # (B, max_L)

        summed = (seq_out * mask.unsqueeze(-1)).sum(dim=1)        # (B, D)
        outputs = summed / lens.unsqueeze(-1)                     # (B, D)

        return self.norm(outputs)  # (B, D)

    


class Transformer(nn.Module):
    def __init__(self,args):
        super(Transformer,self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(args.hid_dim, args.hid_dim,bias=False) for _ in range(3)])
        self.W_o=nn.Linear(args.hid_dim, args.hid_dim)
        self.heads=args.heads
        self.hid_dim = args.hid_dim
        self.d_k=args.hid_dim//args.heads
        self.q_linear = nn.Linear(args.hid_dim, args.hid_dim)
        self.k_linear = nn.Linear(args.hid_dim, args.hid_dim)
        self.v_linear = nn.Linear(args.hid_dim, args.hid_dim)
    def forward(self,fine_messages,coarse_messages,motif_features):
        batch_size=fine_messages.shape[0]
        hid_dim=fine_messages.shape[-1]
        # Q=motif_features
        # K=[]
        # K.append(fine_messages.unsqueeze(1))
        # K.append(coarse_messages.unsqueeze(1))
        # K=torch.cat(K,dim=1)
        # Q=Q.view(batch_size, -1, 1,hid_dim).transpose(1, 2)
        # K=K.view(batch_size, -1, 1,hid_dim).transpose(1, 2)
        # V=K
        Q = self.q_linear(motif_features)
        K = self.k_linear(motif_features)
        V = self.v_linear(motif_features)
        Q = Q.view(batch_size, -1, 1, hid_dim).transpose(1, 2)
        K = K.view(batch_size, -1, 1, hid_dim).transpose(1, 2)
        V = V.view(batch_size, -1, 1, hid_dim).transpose(1, 2)

        Q, K, V = [l(x).view(batch_size, -1,self.heads,self.d_k).transpose(1, 2)
                                      for l, x in zip(self.linear_layers, (Q,K,V))]   
        #print(Q[0],K.transpose(-2, -1)[0])
        message_interaction=torch.matmul( Q,K.transpose(-2, -1))/self.d_k
        #print(message_interaction[0])
        att_score=torch.nn.functional.softmax(message_interaction,dim=-1)
        motif_messages=torch.matmul(att_score, V).transpose(1, 2).contiguous().view(batch_size, -1, hid_dim)
        motif_messages=self.W_o(motif_messages)
        return motif_messages.squeeze(1)

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.weight = nn.Parameter(torch.randn(65, 65), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(65), requires_grad=True)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        X, A = inputs
        xw = torch.matmul(X, self.weight)
        out = torch.matmul(A, xw)

        out += self.bias
        out = self.relu(out)

        return out, A
class GraphormerBlock(nn.Module):
    """Graphormer-style Transformer layer with distance bias (simplified).

    Parameters
    ----------
    dim : int
        Hidden dimension.
    heads : int
        Number of attention heads.
    max_dist : int, optional
        Maximum shortest-path distance to embed separately. Distances > max_dist
        share the same embedding index (i.e. "far").
    """
    def __init__(self, dim: int, heads: int = 8, max_dist: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.d_k = dim // heads
        self.max_dist = max_dist

        # Projection matrices
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Distance (shortest-path) embeddings: shape (heads, max_dist+2)
        # 0 = self, 1..max_dist = exact distance, last idx = >max_dist
        self.dist_emb = nn.Parameter(torch.zeros(heads, max_dist + 2))
        nn.init.xavier_uniform_(self.dist_emb)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """x: (B, N, D); adj: (B, N, N) binary adjacency (0/1)."""
        B, N, _ = x.shape
        Q = self.q_proj(x).view(B, N, self.heads, self.d_k).transpose(1, 2)  # (B, H, N, d_k)
        K = self.k_proj(x).view(B, N, self.heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.heads, self.d_k).transpose(1, 2)

        # ----- distance bias -----
        device = x.device
        spd = torch.where(adj > 0, torch.ones_like(adj, device=device), torch.full_like(adj, 2, device=device))
        spd = spd.long()
        spd = spd.clamp(max=self.max_dist + 1)  # >max_dist → last bin
        # (B, H, N, N)
        bias = self.dist_emb[:, spd]  # broadcast gather: (H, B, N, N)
        bias = bias.permute(1, 0, 2, 3)  # (B,H,N,N)

        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,H,N,N)
        attn_scores = attn_scores + bias
        attn = attn_scores.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ V  # (B,H,N,d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)
        out = self.out_proj(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class MGraphModel(nn.Module):
    def __init__(self, atom_drop: float = 0.12, gat_drop: float = 0.5):
        super(MGraphModel, self).__init__()
        self.num_head = 4

        self.layers = nn.Sequential(
            GCN(),
            GCN(),
        )

        # Atom-branch Mamba: 2-layer linear SSM stack (replaces Graphormer)
        self.atom_mamba = nn.Sequential(
            MambaBlock(d_model=256, d_state=64, d_conv=4, expand=4, headdim=64),
            MambaBlock(d_model=256, d_state=64, d_conv=4, expand=4, headdim=64),
        )
        # 外层正则化：LayerNorm + Dropout
        self.atom_norm = nn.LayerNorm(256)
        self.atom_drop = nn.Dropout(p=atom_drop)

        self.proj = nn.Sequential(
            nn.Linear(25600, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # 4 头 GAT：每头 64 维 → total 256
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features=65, out_features=64, dropout=gat_drop)
            for _ in range(self.num_head)
        ])

        # ---------------- Structural bias embeddings ----------------
        self.spd_max = 4                  # shortest-path distance bins 0..4, >4 → 5
        bias_dim = 32
        self.spd_emb  = nn.Embedding(self.spd_max + 2, bias_dim)   # SPD bias
        self.bond_emb = nn.Embedding(8, bias_dim)                  # Bond type bias (0..7)
        self.deg_emb  = nn.Embedding(16, bias_dim)                 # Degree bias (0..15)
        self.bias_proj = nn.Linear(bias_dim, 256, bias=False)      # Project bias to node dim
        self.gate_fc   = nn.Linear(bias_dim, 256)                  # Gating function

    def forward(self, X, A):
        # Batched GAT: 单次前向处理整个 batch，结果与原逐样本实现一致
        # X: (B, N, F), A: (B, N, N)
                # 4 头 GAT：分别计算后拼接
        ax = torch.cat([att(X, A) for att in self.attentions], dim=2)  # (B, N, 256)

        # ----- Inject structural bias (SPD) -----
        B, N, _ = ax.shape
        device = ax.device
        dist = torch.where(A > 0, torch.ones_like(A, dtype=torch.long), torch.full_like(A, 2, dtype=torch.long))
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        dist = dist.masked_fill(eye, 0)         # 0 for self
        dist = dist.clamp(max=self.spd_max + 1) # >max → last bin
        spd_bias  = self.spd_emb(dist).mean(dim=2)          # (B,N,bias_dim)
        # --- bond type bias (simplified: edge exist → type 1, no edge → 0) ---
        bond_type = (A > 0).long()
        bond_bias = self.bond_emb(bond_type).mean(dim=2)
        # --- degree bias ---
        deg = A.sum(dim=-1).clamp(max=15).long()            # (B,N)
        deg_bias = self.deg_emb(deg)

        bias_all = spd_bias + bond_bias + deg_bias          # combine
        gate = torch.sigmoid(self.gate_fc(bias_all))        # (B,N,256)
        ax = ax + gate * self.bias_proj(bias_all)

        # --------- 2-layer Mamba stack for global context ---------
        if hasattr(self, 'atom_mamba'):
            ax = self.atom_mamba(ax)
            ax = self.atom_drop(self.atom_norm(ax))  # 正则化

        # 展平成向量后接投影层
        out = ax.view(ax.size(0), -1)
        out = self.proj(out)

        return out

class AMPred_LWN(nn.Module):
    def __init__(self,
                 out_dim: int,
                 args,
                 criterion_atom,
                 criterion_motif,
                 criterion_figerprint,
                 ):
        super(AMPred_LWN, self).__init__()
        self.args=args
        self.atom_encoder = nn.Sequential(
            LinearBn(args.atom_in_dim,args.hid_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p =args.drop_rate),
            LinearBn(args.hid_dim,args.hid_dim),
            nn.ReLU(inplace = True)
        )
        self.motif_encoder = nn.Sequential(
            LinearBn(args.ss_node_in_dim,args.hid_dim),
            nn.ReLU(inplace = True),
            nn.Dropout(p =args.drop_rate),
            LinearBn(args.hid_dim,args.hid_dim),
            nn.ReLU(inplace = True)
        )
        self.step=args.step 
        self.agg_op=args.agg_op
        self.mol_FP=args.mol_FP
        self.motif_mp_layer=DGL_MPNNLayer(args.hid_dim,nn.Linear(args.ss_edge_in_dim,args.hid_dim*args.hid_dim),args.resdual)
        self.atom_mp_layer=DGL_MPNNLayer(args.hid_dim,nn.Linear(args.bond_in_dim,args.hid_dim*args.hid_dim),args.resdual)
        self.motif_update=nn.GRUCell(args.hid_dim,args.hid_dim)
        self.atom_update=nn.GRUCell(args.hid_dim,args.hid_dim)
        self.fp_readout = Readout(args,ntype='atom',use_attention=args.attention)
        self.motif_readout=Readout(args,ntype='func_group',use_attention=args.attention)
        # Replace vanilla Transformer with ConBiMambaBlock for stronger global dependency modelling
        self.tr = ConBiMambaBlock(dim=args.hid_dim,
                            ff_mult=args.ff_mult,
                            reduction=2,
                            heads=args.heads,
                            dropout_p=getattr(args, 'conbi_drop', 0.1))

        # ----------------- GIN branch -----------------
        # instantiate frozen GIN model (pretrained)
        self.gin = GINet(task='classification').eval()
        for p in self.gin.parameters():
            p.requires_grad = False

        # load pretrained weights if provided
        try:
            ckpt_path = os.path.join(os.path.dirname(__file__), 'pretrained_gin', 'checkpoints', 'model.pth')
            state_dict = torch.load(ckpt_path, map_location='cpu')
            self.gin.load_my_state_dict(state_dict)
        except Exception as _:
            print('Warning: pretrained GIN weights not found, using random init (frozen).')

        # Default to MambaPool; allow disabling via --no_mamba_pool
        if hasattr(args, "no_mamba_pool") and args.no_mamba_pool:
            self.gin_pool = GIN_AttProj(node_dim=300, out_dim=128)
        else:
            from .multi_mamba_pool import MultiHeadMambaPool
            self.gin_out_dim = getattr(args, 'gin_pool_dim', 256)
            self.gin_pool = MultiHeadMambaPool(node_dim=300, out_dim=self.gin_out_dim, heads=4, dropout_p=getattr(args, 'mamba_pool_drop', 0.1))
            # if out_dim !=128, reduce to 128 to keep downstream dims
            # no reduction – downstream layers adapt to gin_out_dim
            
        # Normalize GIN branch embedding so that its scale is comparable with GAT and motif branches
        self.gin_norm = nn.Sequential(
            nn.BatchNorm1d(self.gin_out_dim),
            nn.ReLU(inplace=True)
        )

        #define the predictor

        atom_MLP_inDim=args.hid_dim*2
        Motif_MLP_inDim=args.hid_dim*2
        if self.mol_FP=='atom':
            atom_MLP_inDim=atom_MLP_inDim+args.mol_in_dim
        elif self.mol_FP=='ss':
            Motif_MLP_inDim=Motif_MLP_inDim+args.mol_in_dim
            #2215
            atom_MLP_inDim=167
        elif self.mol_FP=='both':
            atom_MLP_inDim=atom_MLP_inDim+args.mol_in_dim
            Motif_MLP_inDim=Motif_MLP_inDim+args.mol_in_dim

        
        self.output_af = MLP(atom_MLP_inDim,
                                 out_dim,
                                 dropout_prob=args.drop_rate, 
                                 num_neurons=args.num_neurons,input_norm=args.input_norm)
        self.output_ff = MLP(Motif_MLP_inDim,
                             out_dim,
                             dropout_prob=args.drop_rate,
                             num_neurons=args.num_neurons,input_norm=args.input_norm)
        self.criterion_atom =criterion_atom
        self.criterion_motif =criterion_motif
        self.criterion_figerprint =criterion_figerprint
        self.dist_loss=torch.nn.MSELoss(reduction='none')
        self.fu2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        lin_skip_in = 256 + 256 + self.gin_out_dim
        self.lin_skip = nn.Sequential(
            nn.Linear(lin_skip_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.lin_beta = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.fpecfp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.fprdit = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.fpmacc = nn.Sequential(
            nn.Linear(167, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        self.fp_all = nn.Sequential(
            nn.Linear(3239, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )
        # 在初始化時保存 Motif 表徵維度，供後續 Linear 使用
        self.motif_in_dim = Motif_MLP_inDim
        # ffn 首層 Linear 輸入維度應根據 hid_dim 變化 (motif_readout + raw FP)
        self.ffn = nn.Sequential(
            nn.Linear(self.motif_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=args.drop_rate),
        )

        # ----------- Motif sequence Mamba block (node-level) -----------
        self.motif_mamba = nn.Sequential(
            MambaBlock(d_model=args.hid_dim, d_state=64, d_conv=4, expand=4),
            MambaBlock(d_model=args.hid_dim, d_state=64, d_conv=4, expand=4),
        )
        self.motif_mamba_norm = nn.LayerNorm(args.hid_dim)
        self.motif_dropout = nn.Dropout(p=getattr(args, 'motif_drop', 0.1))

        # 通道级 β 参数 (逐通道控制残差比例)
        self.beta_channel = nn.Parameter(torch.zeros(args.hid_dim))

        # 门控残差 β MLP，用于融合原 motif 表征与 Mamba 输出 (逐通道)
        self.motif_beta = nn.Sequential(
            nn.Linear(args.hid_dim * 3, args.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hid_dim, args.hid_dim)
        )

        self.ffn1 = nn.Linear(128, 1)

        self.graph = MGraphModel(atom_drop=getattr(args, 'atom_drop', 0.12),
                                 gat_drop=getattr(args, 'gat_drop', 0.5))
    def forward(self, g, af, bf, fnf, fef,mf,labels,macc,fp,rdit,X,A, gin_data=None):
        #gat
        X = self.graph(X, A)
        with g.local_scope():
            ufnf=self.motif_encoder(fnf)
            uaf=self.atom_encoder(af)

            for i in range(self.step):
                ufnm=self.motif_mp_layer(g[('func_group', 'interacts', 'func_group')],ufnf,fef)
                uam=self.atom_mp_layer(g[('atom', 'interacts', 'atom')],uaf,bf)
                g.nodes['atom'].data['_uam']=uam
                if self.agg_op=='sum':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.sum('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))
                elif self.agg_op=='max':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.max('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))
                elif self.agg_op=='mean':
                    g.update_all(dgl.function.copy_u('_uam','uam'),dgl.function.mean('uam','agg_uam'),\
                             etype=('atom', 'a2f', 'func_group'))         
                augment_ufnm=g.nodes['func_group'].data['agg_uam']

                ufnm = self.tr(augment_ufnm, ufnm, ufnf)

                
                ufnf=self.motif_update(ufnm,ufnf)
                uaf=self.atom_update(uam,uaf)
                orig_ufnf = ufnf  # 保存原 motif 表征供后续残差融合
                # ------------------------------------------------------------------
                # Global Mamba block over motif node sequence (per-molecule padding)
                # ------------------------------------------------------------------
                lens = g.batch_num_nodes('func_group')                # list[int], len = B
                if isinstance(lens, torch.Tensor):
                    lens = lens.tolist()
                B = len(lens)
                max_L = max(lens)
                D = ufnf.size(-1)
                
                padded = ufnf.new_zeros(B, max_L, D)                  # (B, Lmax, D)
                start = 0
                for b, L in enumerate(lens):
                    padded[b, :L] = ufnf[start:start+L]
                    start += L

                seq_out = self.motif_mamba(padded)                    # (B, Lmax, D)
                seq_out = self.motif_dropout(self.motif_mamba_norm(seq_out))

                outputs = []
                for b, L in enumerate(lens):
                    outputs.append(seq_out[b, :L])
                new_ufnf = torch.cat(outputs, dim=0)
                beta = torch.sigmoid(self.beta_channel).unsqueeze(0)  # (1, D) 方便广播
                ufnf = orig_ufnf * beta + new_ufnf * (1 - beta)

                # ------------------ readout after Mamba ------------------
                #readout
                rdit = self.fpecfp(self.fprdit(rdit))
                ecfp = self.fpecfp(fp)
                macc_raw = macc  # preserve raw 167-dim fingerprint
                macc = self.fpmacc(macc_raw)



                motif_readout=self.motif_readout(g,ufnf)
                # Concatenate raw fingerprint to motif representation if configured
                if self.mol_FP in ['ss', 'both']:
                    motif_representation = torch.cat((motif_readout, macc_raw), dim=-1)
                else:
                    motif_representation = motif_readout
                motif_pred=self.ffn(motif_representation)
                fp_all = torch.cat((macc,ecfp,rdit),dim=-1)

                # ---------------- GIN branch embedding ----------------
                if gin_data is not None:
                    with torch.no_grad():
                        _, _, gin_nodes, gin_batch = self.gin(gin_data)
                    gin_emb = self.gin_pool(gin_nodes, gin_batch)  # (B, gin_out_dim)
                else:
                    gin_emb = torch.zeros(X.shape[0], self.gin_out_dim, device=X.device)
                # apply normalization so that GIN branch magnitude is similar to others
                gin_emb = self.gin_norm(gin_emb)

                fu1 = torch.cat((fp_all,X, gin_emb),-1)
                fu2 =motif_pred
                # fu2 = self.fu2(fu2)
                # fu2 =motif_pred
                x_r = self.lin_skip(fu1)
                # x_r= macc
                beta = self.lin_beta(torch.cat([x_r, fu2,   x_r-fu2], dim=1)).sigmoid()

                out = beta * x_r + (1 - beta) * fu2

                task_type = 'classification'
                dist_loss = torch.nn.MSELoss(reduction='none')

                if task_type == 'classification':
                    logits = torch.sigmoid(self.ffn1(out))
                    dist_fp_fg_loss = dist_loss(torch.sigmoid(x_r), torch.sigmoid(motif_pred)).mean()

                x_r = self.ffn1(out)
                motif_pred =self.ffn1(motif_pred)
                return x_r,motif_pred,logits,dist_fp_fg_loss,out
