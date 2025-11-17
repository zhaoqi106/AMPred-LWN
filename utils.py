import numpy as np
import torch
from rdkit import Chem
import random
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_score, recall_score, f1_score
import dgl
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def AUC(tesYAll, tesPredictAll):
    tesAUC = roc_auc_score(tesYAll, tesPredictAll)
    tesAUPR = average_precision_score(tesYAll, tesPredictAll)
    return tesAUC,tesAUPR
def RMSE(tesYAll,tesPredictAll):
    return mean_squared_error(tesYAll, tesPredictAll,squared=False),0
def confusion_matrix1(true_y,PED):
    TN, FP, FN, TP = confusion_matrix(true_y,PED).ravel()
    return TN, FP, FN, TP
# ---------------- Bond flip augmentation -----------------
AROMATIC_ID = 3   # depends on your encoding
SINGLE_ID   = 0
DOUBLE_ID   = 1

def random_aromatic_flip(g: dgl.DGLHeteroGraph, p_flip: float = 0.15, k_max: int = 2, min_atoms: int = 20):
    """Randomly flip a subset of aromatic bonds to single/double.
    Works in-place.
    """
    etype = ('atom', 'interacts', 'atom')
    if etype not in g.canonical_etypes:
        return g
    bf = g.edges[etype].data['feat']  # assume bond type at index 0
    aromatic_eids = (bf[:, 0] == AROMATIC_ID).nonzero(as_tuple=False).view(-1)
    # Skip small molecules
    if g.num_nodes('atom') < min_atoms or aromatic_eids.numel() == 0:
        return g
    k = min(max(1, int(len(aromatic_eids)*p_flip)), k_max)
    choice = aromatic_eids[torch.randperm(len(aromatic_eids))[:k]]
    new_type = torch.full((k,), SINGLE_ID, dtype=bf.dtype, device=bf.device)
    double_mask = torch.rand(k, device=bf.device) > 0.5
    new_type[double_mask] = DOUBLE_ID
    bf[choice, 0] = new_type
    g.edges[etype].data['feat'] = bf
    return g

class GraphDataset_Classification(Dataset):
    def __init__(self, g_list, y_tensor, fp_list, macc_list, ecfp_list, x_list, a_list, gin_data_list, augment=True):
        self.g_list = g_list
        self.y_tensor = y_tensor
        self.fp_list = fp_list
        self.macc_list = macc_list
        self.ecfp_list = ecfp_list
        self.x_list = x_list
        self.a_list = a_list
        self.gin_data_list = gin_data_list
        self.len = len(g_list)
        self.augment = augment

    def __getitem__(self, idx):
        # 只返回原始图；增广在 DataLoader.collate_fn 中统一进行
        g = self.g_list[idx]
        return (g, self.y_tensor[idx], self.fp_list[idx], self.macc_list[idx],
                self.ecfp_list[idx], self.x_list[idx], self.a_list[idx], self.gin_data_list[idx])

    def __len__(self):
        return self.len


class GraphDataLoader_Classification(DataLoader):

    def __init__(self, *args, augment=True, **kwargs):
        self.augment = augment
        kwargs['collate_fn'] = self.collate_fn
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        gs   = [item[0] for item in batch]
        ys   = torch.stack([item[1] for item in batch])
        fp   = torch.stack([item[2] for item in batch])
        macc = torch.stack([item[3] for item in batch])
        ecfp = torch.stack([item[4] for item in batch])
        X    = torch.stack([item[5] for item in batch])
        A    = torch.stack([item[6] for item in batch])
        from torch_geometric.data import Batch
        pyg_batch = Batch.from_data_list([item[7] for item in batch])

        batched_g = dgl.batch(gs)
        if self.augment:
            random_aromatic_flip(batched_g, p_flip=0.15, k_max=1)

        return (batched_g, ys, macc, fp, ecfp, X, A, pyg_batch)

class GraphDataset_Regression(Dataset):
    def __init__(self, g_list, y_tensor):
        self.g_list = g_list
        self.y_tensor = y_tensor
        self.len = len(g_list)

    def __getitem__(self, idx):
        return self.g_list[idx], self.y_tensor[idx]
    def __len__(self):
        return self.len

class GraphDataLoader_Regression(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader_Regression, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[0] for item in batch])
        batched_ys = torch.stack([item[1] for item in batch])
        return (batched_gs, batched_ys)

def tsen(epoch,out_all,traYAll):
    if  epoch == 199:
        out = torch.tensor(out_all)

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        # 假设 X 和 y 已经定义
        X = out  # 特征数据
        y = traYAll  # 标签meidai

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA 预处理
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        # 自定义颜色
        colors = np.array(['', ''])  # 在这里定义自定义颜色
        colors = ['#d86967', '#58539f']
        cmap_custom = ListedColormap(colors)
        # 使用 t-SNE 进行降维
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
        X_tsne = tsne.fit_transform(X_pca)

        # 可视化 t-SNE 结果
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap_custom, s=25)
        # ,edgecolors='#778899'
        ax = plt.gca()  # 获取当前坐标轴
        ax.spines['top'].set_visible(True)  # 隐藏上边框
        ax.spines['right'].set_visible(True)  # 隐藏右边框
        ax.spines['right'].set_color('#DCDCDC')  # 设置左边框颜色
        ax.spines['right'].set_linewidth(1)  # 设置左边框宽度
        ax.spines['top'].set_color('#DCDCDC')  # 设置下边框颜色
        ax.spines['top'].set_linewidth(1)  # 设置下边框宽度
        ax.spines['left'].set_color('#DCDCDC')  # 设置左边框颜色
        ax.spines['left'].set_linewidth(1)  # 设置左边框宽度
        ax.spines['bottom'].set_color('#DCDCDC')  # 设置下边框颜色
        ax.spines['bottom'].set_linewidth(1)  # 设置下边框宽度
        plt.title("")
        plt.xlabel("t-SNE-0",fontsize=28)
        plt.ylabel("t-SNE-1",fontsize=28)

        plt.xticks([])
        plt.yticks([])
        handles, _ = scatter.legend_elements()
        legend_labels = ['Mutagens','Non-Mutagens']
        plt.legend(handles=handles, labels=legend_labels,fontsize=18,loc='upper left',handletextpad=0,borderpad=0.2)

        plt.show()