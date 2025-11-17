import datetime
import time
import argparse
import numpy as np
import dgl
import torch
# Enable cuDNN globally for performance
torch.backends.cudnn.enabled = True   # 启用 cuDNN
torch.backends.cudnn.benchmark = True   # 开启 benchmark，让 cuDNN 选择最快算法
torch.backends.cudnn.deterministic = False  # 允许使用非确定性但更快的算法
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import scatter
from torch.utils.tensorboard import SummaryWriter
import math
from models.losses import FocalLoss, AUCROC  # 使用本地可微 AUROC 损失


from models.model import AMPred_LWN
from models.utils import GraphDataset_Classification,GraphDataLoader_Classification,\
                  AUC,RMSE,\
                  GraphDataset_Regression,GraphDataLoader_Regression,confusion_matrix1
from torch.optim import Adam
from data.split_data import get_classification_dataset
# (已在顶部一次性导入 FocalLoss, AUCROC)

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

writer = SummaryWriter('logs')

# -----------------------------------------------------------------------------
# Utility: ensure fused Mamba2 CUDA kernels are available. If not, abort run so
# that the user can rebuild mamba-ssm / causal-conv1d with CUDA support.
# -----------------------------------------------------------------------------


def _ensure_fused_kernels():
    """Abort execution if the fast Triton + causal-conv1d kernels are missing."""

    import importlib.util

    has_triton_scan = importlib.util.find_spec(
        "mamba_ssm.ops.triton.ssd_combined"
    ) is not None

    try:
        from causal_conv1d import causal_conv1d_fn  # noqa: F401 – only check presence

        has_causal_conv = causal_conv1d_fn is not None
    except ImportError:
        has_causal_conv = False

    if not (has_triton_scan and has_causal_conv):
        missing = []
        if not has_triton_scan:
            missing.append("mamba_ssm Triton kernels")
        if not has_causal_conv:
            missing.append("causal_conv1d CUDA extension")

        raise RuntimeError(
            "\n\n[Error] 未检测到加速核：{}。\n"
            "请先执行：\n"
            "    MAMBA_FORCE_BUILD=1 CAUSAL_CONV1D_FORCE_BUILD=1 \\\npip install --no-binary mamba-ssm --no-binary causal-conv1d --force-reinstall --upgrade mamba-ssm causal-conv1d\n"
            .format(", ".join(missing))
        )


# Call the check as early as possible so training won't proceed on slow path
_ensure_fused_kernels()

# ------------------- BatchNorm recalibration helper --------------------

def bn_recalibrate(model, loader, device, max_iters: int = 200):
    """Recompute running_mean/var of BatchNorm layers on a subset of
    downstream data to mitigate distribution shift after loading
    pretrained weights. GIN branch stats remain frozen (eval mode)."""
    import torch

    model.train()
    # Match original training behaviour: GIN as a whole stays in train mode
    # but its BatchNorm layers remain eval (no running stats update)
    if hasattr(model, "gin"):
        model.gin.train()
        for _m in model.gin.modules():
            if isinstance(_m, torch.nn.BatchNorm1d):
                _m.eval()

    with torch.no_grad():
        for i, (gs, labels, macc, fp, rdit, x, a, pyg_batch) in enumerate(loader):
            if i >= max_iters:
                break
            pyg_batch = pyg_batch.to(device)
            gs = gs.to(device)
            fp = fp.to(device).float()
            macc = macc.to(device).float()
            rdit = rdit.to(device).float()
            x = x.to(device)
            a = a.to(device)

            # Extract heterograph features required by AMPred_LWN.forward
            af  = gs.nodes['atom'].data['feat']
            bf  = gs.edges[('atom','interacts','atom')].data['feat']
            fnf = gs.nodes['func_group'].data['feat']
            fef = gs.edges[('func_group','interacts','func_group')].data['feat']
            molf = gs.nodes['molecule'].data['feat']

            dummy_label = torch.zeros_like(labels, dtype=torch.float, device=device)

            # Forward pass (no gradient) just to update BN running stats
            model(gs, af, bf, fnf, fef, molf,
                  dummy_label, macc, fp, rdit, x, a,
                  gin_data=pyg_batch)

    model.eval()


def main(args):
    max_score_list=[]
    max_aupr_list=[]
    task_type=None
    if args.dataset in ['Tox21', 'ClinTox', 'AMES',
                      'SIDER', 'BBBP', 'BACE']:
        task_type='classification'
    else:
        task_type='regression'
    strpath = f'results/AMES/test{args.seed}.txt'
    for seed in range(args.seed,args.seed+args.folds):
        print('folds:',seed)
        f = open(strpath,'a',encoding='utf-8')
        f.write(f"##########folds:{seed}\n")
        f.close()
        if task_type=='classification':
            metric=AUC

            train_gs,train_ls,train_tw,val_gs,val_ls,test_gs,test_ls,\
                morgan_fp_list_train,morgan_fp_list_val,morgan_fp_list_test,\
                maccs_fp_train,maccs_fp_val,maccs_fp_test,\
                rdit_fp_train,rdit_fp_val,rdit_fp_test,\
                X_train,X_val,X_test,\
                A_train,A_val,A_test, gin_train, gin_val, gin_test = \
                    get_classification_dataset(args.dataset, args.n_jobs, seed, args.split_ratio)

            print(len(train_ls),len(val_ls),len(test_ls),train_tw)

            # morgan_fp_list_train = torch.FloatTensor(morgan_fp_list_train)
            train_ds = GraphDataset_Classification(train_gs, train_ls, morgan_fp_list_train, maccs_fp_train,
                                                  rdit_fp_train, X_train, A_train, gin_train)
            train_dl = GraphDataLoader_Classification(
                train_ds,
                num_workers=0,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                augment=not args.no_aromatic_flip,
            )
            # 根据数据集正负样本比设置 pos_weight
            with torch.no_grad():
                all_labels = torch.cat([train_ls, val_ls, test_ls])
                pos_cnt = float(all_labels.sum())
                neg_cnt = float(all_labels.numel() - pos_cnt)
                pos_weight = torch.tensor([neg_cnt / max(pos_cnt,1.0)], device=args.device)
            # --------- Optional reweighting: allow Optuna to tune via POS_BETA ---------
            import os
            beta_env = float(os.getenv("POS_BETA", "0.8873106673688184"))  # default from 4th-ranked Optuna Ames trial
            pos_weight = pos_weight * beta_env

            if args.loss_type == 'aucroc':
                # 采用可微 AUC-ROC 损失；imratio 为正样本比例
                imratio = float(pos_cnt) / float(pos_cnt + neg_cnt + 1e-8)
                criterion_atom = AUCROC(imratio=imratio, gamma=args.gamma)
                criterion_fg   = AUCROC(imratio=imratio, gamma=args.gamma)
                criterion_figerprint = AUCROC(imratio=imratio, gamma=args.gamma)
            else:
                criterion_atom = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                criterion_fg   = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                criterion_figerprint = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            print("")

        # prepare validation gin batch once (static)
        from torch_geometric.data import Batch as PygBatch

        val_pyg_batch = PygBatch.from_data_list(gin_val).to(args.device)

        val_gs = dgl.batch(val_gs).to(args.device)
        val_labels=val_ls.to(args.device)
        morgan_fp_list_val=torch.tensor(morgan_fp_list_val).to(args.device)
        maccs_fp_val = torch.tensor(maccs_fp_val).to(args.device)
        rdit_fp_val = torch.tensor(rdit_fp_val).to(args.device)

        test_pyg_batch = PygBatch.from_data_list(gin_test).to(args.device)

        test_gs=dgl.batch(test_gs).to(args.device)
        test_labels=test_ls.to(args.device)
        morgan_fp_list_test=torch.tensor(morgan_fp_list_test).to(args.device)
        maccs_fp_test = torch.tensor(maccs_fp_test).to(args.device)
        rdit_fp_test = torch.tensor(rdit_fp_test).to(args.device)

        X_val = torch.tensor(X_val).to(args.device)
        A_val = torch.tensor(A_val).to(args.device)
        X_test = torch.tensor(X_test).to(args.device)
        A_test= torch.tensor(A_test).to(args.device)
        model = AMPred_LWN(
            val_labels.shape[1],
            args,
            criterion_atom,
            criterion_fg,
            criterion_figerprint,
        ).to(args.device)

        # torch.compile disabled – running model entirely in eager mode for
        print(model)
        # ---- load pretrained encoder ----
        if args.pretrained:
            ckpt = torch.load(args.pretrained, map_location=args.device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                sd = ckpt['model']
            else:
                sd = ckpt
            # 保留 MolCLR 预训练的 GIN 权重：跳过所有以 "gin." 开头的键
            filtered_sd = {k: v for k, v in sd.items() if not k.startswith('gin.')}
            load_res = model.load_state_dict(filtered_sd, strict=False)
            skipped = len(sd) - len(filtered_sd)
            print(f"[Info] loaded pretrained encoder from {args.pretrained} (skipped {skipped} gin.* params)")
            # ---------------- Optional BN recalibration ----------------
            if getattr(args, 'bn_recalibrate', False):
                print('[Info] Running BatchNorm recalibration pass ...')
                bn_recalibrate(model, train_dl, args.device)
                print('[Info] BN recalibration done.')
        # ---------------- Optimizer with separate LR for gin_pool ----------------
        gin_pool_params = list(model.gin_pool.parameters())
        # avoid duplicate parameters by comparing id
        other_params = [p for p in model.parameters() if id(p) not in {id(param) for param in gin_pool_params}]
        opt = Adam([
            {'params': other_params, 'lr': args.learning_rate},        # e.g. 1e-5
            {'params': gin_pool_params, 'lr': args.gin_lr}                  # attention pool (GIN attention LR)
        ], weight_decay=args.weight_decay)
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
        #scheduler = torch.optim.lr_scheduler.CosineAnnea lingWarmRestarts(opt,T_0=50,eta_min=1e-4,verbose=True)
    
        
        best_val_score=0 if task_type=='classification' else 999
        best_val_aupr=0 if task_type=='classification' else 999
        best_epoch=0
        best_test_score=0
        best_test_aupr=0
        best_val_acc_at_best_auroc = 0
        best_test_acc_at_best_auroc = 0
        # initialize holder for best weights and metrics this fold
        best_state_dict = None
        best_metrics_dict = None
        
        for epoch in range(args.epoch):
            epoch_start = time.time()
            f = open(strpath,'a',encoding='utf-8')
            model.train()
            # Keep GIN BatchNorm layers in eval mode so running stats stay frozen
            for _m in model.gin.modules():
                if isinstance(_m, torch.nn.BatchNorm1d):
                    _m.eval()

            traYAll = []
            traPredictAll = []
            # for accurate average loss (handles last batch smaller)
            loss_sum = 0.0   # sum of loss weighted by batch size
            sample_cnt = 0   # total number of samples processed
            out_all =[]

            for i, (gs, labels, macc, fp, rdit, x, a, pyg_batch) in enumerate(train_dl):
                pyg_batch = pyg_batch.to(args.device)
                traYAll += labels.detach().cpu().numpy().tolist()

                gs = gs.to(args.device)
                labels = labels.to(args.device).float()
                fp = fp.to(args.device).float()
                macc = macc.to(args.device).float()
                rdit = rdit.to(args.device).float()
                x = x.to(args.device)
                a = a.to(args.device)
                # print(i)
                af=gs.nodes['atom'].data['feat']
                bf = gs.edges[('atom', 'interacts', 'atom')].data['feat']
                fnf = gs.nodes['func_group'].data['feat']
                fef=gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                molf=gs.nodes['molecule'].data['feat']
                fp_pred, fg_pred, logits, dist_fp_fg_loss, out = model(gs, af, bf, fnf, fef, molf, labels,
                                                                      macc, fp, rdit, x, a, gin_data=pyg_batch)
                out = out.cpu().detach().numpy().tolist()
                # motif__1.append(motif__)
                # macc__1.append(macc__)
                # logitss = torch.cat((motif__,macc__),-1)
                # ########logitss_list+=logitss.detach().cpu().numpy().tolist()
                ######################################
                # if task_type == 'classification':
                #     logits = (torch.sigmoid(fp_pred)+torch.sigmoid(fg_pred)) / 2
                #     dist_fp_fg_loss = dist_loss(torch.sigmoid(fp_pred), torch.sigmoid(fg_pred)).mean()
                # else:
                #     logits = (fp_pred + fg_pred) / 2
                #     dist_fp_fg_loss = dist_loss(fp_pred, fg_pred).mean()
                # loss_atom = criterion_atom(fp_pred, labels).mean()
                if args.loss_type == 'aucroc':
                    # AUCROC expects 1-D tensors. Flatten predictions and labels to avoid
                    # shape mismatch errors (B,1) -> (B,).
                    loss_fp = criterion_atom(fp_pred.view(-1), labels.view(-1))
                    loss_motif = criterion_fg(fg_pred.view(-1), labels.view(-1))
                else:
                    loss_fp = criterion_atom(fp_pred, labels).mean()
                    loss_motif = criterion_fg(fg_pred, labels).mean()
                # 余弦衰减 λ：前期约束更强，后期放松
                lambda_dist = args.dist * 0.5 * (1 + math.cos(math.pi * epoch / args.epoch))
                # 按 beta_fp 调整指纹分支在总损失中的权重
                loss = args.beta_fp * loss_fp + loss_motif + lambda_dist * dist_fp_fg_loss

                # (optional) you can uncomment to monitor effective λ
                # if i == 0:
                #     print(f"[Epoch {epoch}] λ={lambda_dist:.4f}")
                ##################################################
                opt.zero_grad()
                loss.backward()
                # Gradient clipping to stabilize training when gin_pool uses higher LR
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                # accumulate weighted loss
                bs = labels.size(0)
                loss_sum += loss.item() * bs
                sample_cnt += bs
                opt.step()
                traPredictAll += logits.squeeze(-1).detach().cpu().numpy().tolist()
                out_all += out
            if i == 0:  # 或 i % 50 == 0
                print(f"[Epoch {epoch}] "
                      f"loss_fp={loss_fp.item():.4f}, "
                      f"loss_motif={loss_motif.item():.4f}, "
                      f"dist={dist_fp_fg_loss.item():.4f}, "
                      f"λ*dist={(lambda_dist*dist_fp_fg_loss).item():.4f}, "
                      f"β_fp={args.beta_fp}")

            # from models.utils import tsen
            # tsen(epoch,out_all,traYAll)
            train_score,train_AUPRC=metric(traYAll,traPredictAll)



            model.eval()
            PED = []
            with torch.no_grad():
                    val_fp = morgan_fp_list_val
                    val_macc = maccs_fp_val
                    val_rdit = rdit_fp_val
                    val_X = X_val
                    val_A = A_val
                    val_af = val_gs.nodes['atom'].data['feat']
                    val_bf = val_gs.edges[('atom', 'interacts', 'atom')].data['feat']
                    val_fnf = val_gs.nodes['func_group'].data['feat']
                    val_fef=val_gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                    val_molf=val_gs.nodes['molecule'].data['feat']
                    val_logits_fp, val_logits_motif, val_logits, val_dist_fp_fg_loss, val_out = model(val_gs, val_af, val_bf, val_fnf, val_fef, val_molf,
                                                                               val_labels, val_macc, val_fp, val_rdit, val_X, val_A,
                                                                               gin_data=val_pyg_batch)

                    test_rdit = rdit_fp_test
                    test_fp = morgan_fp_list_test
                    test_macc = maccs_fp_test
                    test_X = X_test
                    test_A = A_test

                    test_af = test_gs.nodes['atom'].data['feat']
                    test_bf = test_gs.edges[('atom', 'interacts', 'atom')].data['feat']
                    test_fnf = test_gs.nodes['func_group'].data['feat']
                    test_fef=test_gs.edges[('func_group', 'interacts', 'func_group')].data['feat']
                    test_molf=test_gs.nodes['molecule'].data['feat']
                    # Use test_labels (not val_labels) when running the model on test set
                    test_logits_fp, test_logits_motif, test_logits, test_dist_fp_fg_loss, test_out = model(test_gs, test_af, test_bf, test_fnf, test_fef, test_molf,
                                                                                    test_labels, test_macc, test_fp, test_rdit, test_X, test_A,
                                                                                    gin_data=test_pyg_batch)
                    ###################################################
                    # if task_type=='classification':
                    #     val_logits=(torch.sigmoid(val_logits_fp)+torch.sigmoid(val_logits_motif))/2
                    #     test_logits=(torch.sigmoid(test_logits_fp)+torch.sigmoid(test_logits_motif))/2
                    # else:
                    #     val_logits=(val_logits_fp+val_logits_motif)/2
                    #     test_logits=(test_logits_fp+test_logits_motif)/2

                    # pred_y.extend(test_logits.detach().cpu().numpy())




                    val_score,val_AUPRC= metric(val_labels.detach().cpu().numpy().tolist(), val_logits.squeeze(-1).detach().cpu().numpy().tolist())

                    test_score,test_AUPRC=metric(test_labels.detach().cpu().numpy().tolist(), test_logits.squeeze(-1).detach().cpu().numpy().tolist())
                    # ----- Save first 50 test labels & logits for inspection -----


                    PED.extend(test_logits.squeeze(-1).detach().cpu().numpy().round().astype(int))
                    # test_lables=test_labels.detach().cpu()
                    # ture_y.extend(test_lables)


                    # acc = accuracy_score(ture_y, PED)
                    # auc = roc_auc_score(ture_y, pred_y)
                    ###################################################
                    if task_type=='classification':
                        val_pred_for_acc = val_logits.squeeze(-1).detach().cpu().numpy().round().astype(int)
                        val_acc = accuracy_score(val_labels.detach().cpu().numpy().tolist(), val_pred_for_acc)
                        test_acc_current = accuracy_score(test_labels.detach().cpu().numpy().tolist(), PED)

                        # ---- compute detailed metrics once per epoch ----
                        acc = test_acc_current
                        precision = precision_score(test_labels.detach().cpu().numpy().tolist(), PED)
                        recall = recall_score(test_labels.detach().cpu().numpy().tolist(), PED)
                        f1 = f1_score(test_labels.detach().cpu().numpy().tolist(), PED)
                        TN, FP, FN, TP = confusion_matrix(test_labels.detach().cpu().numpy().tolist(), PED).ravel()
                        SPE = TN / (TN + FP)
                        SEN = TP / (TP + FN)
                        NPV = TN / (TN + FN)
                        PPV = TP / (TP + FP)
                        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

                        if best_val_score<val_score:
                            best_val_score=val_score
                            best_test_score=test_score
                            best_epoch=epoch
                            best_val_acc_at_best_auroc = val_acc
                            best_test_acc_at_best_auroc = test_acc_current
                            # cache best weights & metrics in memory
                            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                            best_metrics_dict = {
                                'epoch': epoch,
                                'AUROC': test_score,
                                'AUPRC': test_AUPRC,
                                'ACC': test_acc_current,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
                                'SPE': SPE, 'SEN': SEN, 'NPV': NPV, 'PPV': PPV, 'MCC': MCC
                            }
                        if best_val_aupr<val_AUPRC:
                            best_val_aupr=val_AUPRC
                            best_test_aupr=test_AUPRC
                        print('#####################')

                        print("-------------------Epoch {}-------------------".format(epoch))
                        print(f"Epoch time: {time.time() - epoch_start:.2f}s")

                        f.write(f"Epoch:{epoch}\t")
                        avg_loss = loss_sum / sample_cnt if sample_cnt else 0.0
                        print(f":loss=={avg_loss}")
                        # f.write(f"loss=={loss_all / i}\n")
                        print("Train AUROC: {}".format(train_score)," Train AUPRC: {}".format(train_AUPRC))
                        print("Val AUROC: {}".format(val_score)," Val AUPRC: {}".format(val_AUPRC))
                        print("Test AUROC: {}".format(test_score)," Test AUPRC: {}".format(test_AUPRC))
                        # writer.add_scalar('Train/auc1', train_score, global_step=epoch)
                        # writer.add_scalar('Val/auc1', val_score, global_step=epoch)
                        # writer.add_scalar('Test/auc1', test_score, global_step=epoch)
                        print(f"Val ACC: {val_acc:.4f}, Test ACC: {test_acc_current:.4f}")
                        print(f"--- Best Performance (at epoch {best_epoch} based on Val AUROC) ---")
                        print(f"    Best Val AUROC: {best_val_score:.4f}, Corresponding Val ACC: {best_val_acc_at_best_auroc:.4f}")
                        print(f"    Corresponding Test AUROC: {best_test_score:.4f}, Corresponding Test ACC: {best_test_acc_at_best_auroc:.4f}")
                        f.write(
                            f"Train AUROC: {train_score}\t" + f"Val AUROC: {val_score}\t" + f'Test AUROC: {test_score}\t\n')


                        
                        if epoch == 199:
                            pass  # metrics writing removed
                    elif task_type=='regression':
                        if best_val_score>val_score:
                            best_val_score=val_score
                            best_test_score=test_score
                            best_epoch=epoch
                        print('#####################')
                        print("-------------------Epoch {}-------------------".format(epoch))
                        print(f"Epoch time: {time.time() - epoch_start:.2f}s")
                        f.write(f"Epoch:{epoch}\n")
                        print("Train RMSE: {}".format(train_score))
                        print("Val RMSE: {}".format(val_score))
                        print('Test RMSE: {}'.format(test_score))
                        # writer.add_scalar('Train/rmse', train_score, global_step=epoch)
                        # writer.add_scalar('Val/auc', val_score, global_step=epoch)
                        # writer.add_scalar('Test/auc',test_score , global_step=epoch)
                        f.write(f"Train RMSE: {train_score}\t"+f"Val RMSE: {val_score}\t"+f'Test RMSE: {test_score}\t\n')


            ###################################################

        max_score_list.append(best_test_score)
        max_aupr_list.append(best_test_aupr)
        print('best model in epoch ',best_epoch)
        print('best val score is ',best_val_score)
        print('test score in this epoch is',best_test_score)
        # ---------- save best model & metrics for this fold ----------
        if best_state_dict is not None:
            import os, json
            os.makedirs('log/checkpoint', exist_ok=True)
            torch.save(best_state_dict, f'log/checkpoint/fold{seed}_epoch{best_epoch}_best.pth')
            metrics_path = f'results/AMES/best_metrics_fold{seed}.txt'
            import numpy as np
            # convert numpy types to python scalars for JSON
            serializable = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in best_metrics_dict.items()}
            with open(metrics_path, 'w', encoding='utf-8') as mf:
                json.dump(serializable, mf, ensure_ascii=False, indent=2)
        if task_type=='classification':
            print('best val aupr is ',best_val_aupr)
            print('corresponding best test aupr is ',best_test_aupr)

        f.close()
    print("AUROC:\n")
    print(max_score_list)
    print(np.mean(max_score_list),'+-',np.std(max_score_list))
    print("AUPRC:\n")
    print(np.mean(max_aupr_list),'+-',np.std(max_aupr_list))

    try:
        f=open(strpath+datetime.datetime.now().strftime('%m%d_%H%M')+'.txt','a',encoding='utf-8');
        f.write('\n'.join([key+': '+str(value) for key, value in vars(args).items()])+'\n')
        if task_type=="classification":
            f.write("AUROC:")
        f.write(str(np.mean(max_score_list))+'+-'+str(np.std(max_score_list))+'\n')
        for i in max_score_list:
            f.write(str(i)+" ")
        if task_type=="classification":
            f.write("\nAUPRC:")
            f.write(str(np.mean(max_aupr_list))+'+-'+str(np.std(max_aupr_list))+'\n')
        f.close()
    except:
        f=open(args.dataset+'result_'+datetime.datetime.now().strftime('%m%d_%H%M')+'.txt','a',encoding='utf-8');  
        f.write('\n'.join([key+': '+str(value) for key, value in vars(args).items()])+'\n')
        if task_type=="classification":
            f.write("AUROC:")
        f.write(str(np.mean(max_score_list))+'+-'+str(np.std(max_score_list))+'\n')
        for i in max_score_list:
            f.write(str(i)+" ")
        if task_type=="classification":
            f.write("AUPRC:")
            f.write(str(np.mean(max_aupr_list))+'+-'+str(np.std(max_aupr_list))+'\n')
        f.close()
    
def write_record(path, message):
    file_obj = open(path, 'a')
    file_obj.write('{}\n'.format(message))
    file_obj.close()



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', type=str, default='', help='path to pretrained encoder checkpoint; leave empty to train from scratch')
    p.add_argument('--dataset', type=str, default='AMES', help='dataset name (default: AMES)')
    p.add_argument('--seed', type=int, default=0, help='seed used to shuffle dataset')
    p.add_argument('--atom_in_dim', type=int, default=37, help='atom feature init dim')
    p.add_argument('--bond_in_dim', type=int, default=13, help='bond feature init dim')
    p.add_argument('--ss_node_in_dim', type=int, default=50, help='func group node feature init dim')
    p.add_argument('--ss_edge_in_dim', type=int, default=37, help='func group edge feature init dim')
    p.add_argument('--mol_in_dim', type=int, default=167, help='molecule fingerprint init dim')
    # AUCROC 建议较低学习率
    p.add_argument('--learning_rate', type=float, default=4.500748721079455e-06, help='Adam learning rate (Optuna: 4th-ranked)')
    p.add_argument('--epoch', type=int, default=200, help='train epochs')
    # 默认 batch_size 调整为搜索得到的 256
    p.add_argument('--batch_size', type=int, default=224, help='batch size for train dataset (Optuna: 4th-ranked)')
    p.add_argument('--num_neurons', type=list, default=[512],help='num_neurons in MLP')
    p.add_argument('--input_norm', type=str, default='layer', help='input norm')
    # Dropout 调整为搜索得到的最佳值 ~0.0874
    p.add_argument('--drop_rate',  type=float, default=0.0880775253938379, help='dropout rate in MLP (Optuna: 4th-ranked)')
    # 选择损失函数：bce 或 aucroc
    p.add_argument('--loss_type', type=str, choices=['bce', 'aucroc'], default='bce', help='which loss to use')
    # 指纹分支损失缩放系数 beta_fp，默认 1.0
    p.add_argument('--beta_fp', type=float, default=1.0, help='scaling factor for fingerprint BCE loss (Optuna best)')
    # gamma for AUCROC loss
    p.add_argument('--gamma', type=float, default=500.0, help='gamma for AUCROC loss (ignored for bce)')
    # 隐藏维度调整为 256
    p.add_argument('--hid_dim', type=int, default=256, help='node, edge, fg hidden dims in Net (must be even)')
    p.add_argument('--device', type=str, default='cuda:0', help='fitting device')
    p.add_argument('--dist',type=float,default=0.9083218370998977,help='dist loss func hyperparameter lambda (Optuna: 4th-ranked)')
    p.add_argument('--split_ratio',type=list,default=[0.8,0.1,0.1],help='ratio to split dataset')
    p.add_argument('--folds',type=int,default=1,help='k folds validation')
    p.add_argument('--n_jobs',type=int,default=10,help='num of threads for the handle of the dataset')
    p.add_argument('--resdual',type=bool,default=False,help='resdual choice in message passing')
    p.add_argument('--shuffle',type=bool,default=False,help='whether to shuffle the train dataset')
    p.add_argument('--attention',type=bool,default=True,help='whether to use global attention pooling')
    # 消息传递步数调整为 3
    p.add_argument('--step',type=int,default=3,help='message passing steps')
    p.add_argument('--agg_op',type=str,choices=['max','mean','sum'],default='mean',help='aggregations in local augmentation')
    p.add_argument('--mol_FP',type=str,choices=['atom','ss','both','none'],default='both',help='cat mol FingerPrint to Motif or Atom representation'
                   )
    p.add_argument('--gating_func',type=str,choices=['Softmax','Sigmoid','Identity'],default='Sigmoid',help='Gating Activation Function'
                   )
    p.add_argument('--ScaleBlock',type=str,choices=['Share','Contextual'],default='Contextual',help='Self-Rescaling Block'
                   )
    # 多头注意力默认改为 8 头
    p.add_argument('--heads',type=int,default=4,help='Multi-head num')
    # MambaPool is now default; add flag to disable and fall back to attention pool
    p.add_argument('--no_mamba_pool', action='store_true', help='use legacy attention pool instead of default Mamba pool')
    p.add_argument('--mamba_pool_drop', type=float, default=0.09884612426255308, help='dropout probability for MambaPool (Optuna: 4th-ranked)')
    p.add_argument('--motif_drop', type=float, default=0.08854118815196069, help='dropout probability for motif Mamba block (Optuna: 4th-ranked)')
    # 新增：GAT 与 Atom-branch dropout
    p.add_argument('--gat_drop', type=float, default=0.24015293775845986, help='dropout probability for GraphAttentionLayer (Optuna: 4th-ranked)')
    p.add_argument('--atom_drop', type=float, default=0.18437825012000722, help='dropout probability after atom Mamba stack (Optuna: 4th-ranked)')
    # 新增：ConBiMamba 统一 dropout（feed-forward / attention / conv）
    p.add_argument('--conbi_drop', type=float, default=0.1, help='dropout probability inside ConBiMambaBlock')

    p.add_argument('--gin_lr', type=float, default=0.00010441220786845665, help='learning rate for gin_pool (Optuna: 4th-ranked)')
    p.add_argument('--gin_pool_dim', type=int, default=256, choices=[128,256], help='output dim of MultiHeadMambaPool for GIN branch')
    p.add_argument('--bn_recalibrate', action='store_true', help='run BN calibration pass after loading pretrained weights')
    p.add_argument('--weight_decay', type=float, default=3.804588210164724e-05, help='weight decay (L2 regularization, Optuna: 4th-ranked)')
    p.add_argument('--ff_mult', type=int, default=3, help='feed-forward expansion factor inside ConBiMambaBlock')
    # 关闭芳香键随机翻转增广
    p.add_argument('--no_aromatic_flip', action='store_true',
                   help='disable random aromatic bond flip augmentation')
    args = p.parse_args()
    if args.hid_dim % 2 != 0:
        raise ValueError(f"--hid_dim 必须为偶数，目前收到 {args.hid_dim}")
    main(args)
