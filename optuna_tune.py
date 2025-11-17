import os
import re
import csv
import json
import time
import subprocess
import datetime as dt
from pathlib import Path
import sys
import tarfile

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, PatientPruner

# -------------------------- 配置 --------------------------
DATASET = "AMES"
EPOCHS  = 100     # 每个 trial 训练轮数
N_JOBS  = 0       # 预处理已经有缓存
SEED    = 0
LOG_CSV = "optuna_Ames.csv"
STORAGE = "sqlite:///optuna_ames.db"   # 持久化 DB，用于断点续跑
PYTHON  = "python"

# --------------------- util: 解析 stdout ------------------
VAL_AUC_RE  = re.compile(r"Val AUROC:\s*(\d+\.\d+)")
TEST_AUC_RE = re.compile(r"Test AUROC:\s*(\d+\.\d+)")
EPOCH_RE   = re.compile(r"\[Epoch\s+(\d+)\]")
VAL_AUPRC_RE  = re.compile(r"Val AUPRC:\s*(\d+\.\d+)")
TEST_AUPRC_RE = re.compile(r"Test AUPRC:\s*(\d+\.\d+)")
#  Val/Test ACC 同一行，用两条 regex 分别抓取
VAL_ACC_RE = re.compile(r"Val ACC:\s*([0-9\.]+)")
TEST_ACC_RE = re.compile(r"Test ACC:\s*([0-9\.]+)")
TRAIN_AUC_RE = re.compile(r"Train AUROC:\s*([0-9\.]+)")
LOSS_RE = re.compile(r":loss==([0-9\.eE+-]+)")


# (warm-start 逻辑已取消)


def run_once(params: dict, trial) -> dict:
    """Run train_evaluate.py once with given params, return metrics dict."""
    epochs = EPOCHS

    cmd = [
        PYTHON, "train_evaluate.py",
        "--dataset", DATASET,
        "--epoch", str(epochs),
        "--n_jobs", str(N_JOBS),
        "--seed", str(SEED),
        "--batch_size", str(params["batch"]),
        "--learning_rate", str(params["lr"]),
        "--drop_rate", str(params["dropout"]),
        "--dist", str(params["dist"]),


        "--weight_decay", str(params["wd"]),
        "--mamba_pool_drop", str(params["mamba_pool_drop"]),
        "--motif_drop", str(params["motif_drop"]),
        "--gat_drop", str(params["gat_drop"]),
        "--atom_drop", str(params["atom_drop"]),
        "--gin_lr", str(params["gin_lr"]),
    ]

    # 通过环境变量注入 pos_weight 调整因子 β
    env = os.environ.copy()
    env["POS_BETA"] = str(params["beta"])

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env, bufsize=1)
    except FileNotFoundError as e:
        print("Failed to launch training script", file=sys.stderr)
        raise

    stdout = []
    tic = time.time()
    # 实时打印并收集
    reached_threshold = False      # 第一次 >=0.90
    epochs_since_reach = 0         # 观察计数

    # 准备日志目录
    logs_dir = Path("pruned_logs")
    logs_dir.mkdir(exist_ok=True)

    early_stop = False
    epoch_idx = -1  # 当前 epoch 编号（从 0 开始）
    for line in proc.stdout:
        print(line, end="")         # 直接透传到终端
        stdout.append(line.rstrip("\n"))

        # 捕获 [Epoch xx] 行，更新 epoch_idx
        m_epoch = EPOCH_RE.search(line)
        if m_epoch:
            epoch_idx = int(m_epoch.group(1))

        m_test = TEST_AUC_RE.search(line)
        if m_test:
            current_test_auc = float(m_test.group(1))
            if epoch_idx >= 0:
                trial.report(current_test_auc, step=epoch_idx)
                if trial.should_prune():
                    print("[Pruner] Trial pruned at epoch", epoch_idx, "(Test AUROC)")
                early_stop = True
                proc.kill(); proc.wait();
                break

            # 从第 1 个 trial 即启用阈值早停
            if reached_threshold:
                epochs_since_reach += 1
                # 已过观察期并出现低于阈值
                if epochs_since_reach >= 7 and current_test_auc < 0.90:
                    print("[Early Stop] AUROC dropped <0.90 after 7-epoch window, pruning.")
                    # 写日志文件
                    ts = dt.datetime.now().strftime("%m%d_%H%M%S")
                    log_path = logs_dir / f"trial_{trial.number}_{ts}.txt"
                    with open(log_path, "w", encoding="utf-8") as f_log:
                        f_log.write("\n".join(stdout))

                    # 压缩为 tar.gz
                    tar_path = logs_dir / f"trial_{trial.number}_{ts}.tar.gz"
                    with tarfile.open(tar_path, "w:gz") as tar:
                        tar.add(log_path, arcname=log_path.name)

                    print(f"[Early Stop] Logs saved to {tar_path}")

                    early_stop = True
                    proc.kill()
                    proc.wait()
                    break  # 跳出读取循环
            elif current_test_auc >= 0.90:
                reached_threshold = True
                epochs_since_reach = 0

        m_val = VAL_AUC_RE.search(line)
        if m_val and epoch_idx >= 0:
            val_epoch_auc = float(m_val.group(1))
            trial.report(val_epoch_auc, step=epoch_idx)
            if trial.should_prune():
                print("[MedianPruner] Trial pruned at epoch", epoch_idx)

                # ---- 保存日志并打包 ----
                ts = dt.datetime.now().strftime("%m%d_%H%M%S")
                log_path = logs_dir / f"trial_{trial.number}_{ts}.txt"
                with open(log_path, "w", encoding="utf-8") as f_log:
                    f_log.write("\n".join(stdout))

                tar_path = logs_dir / f"trial_{trial.number}_{ts}.tar.gz"
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(log_path, arcname=log_path.name)
                print(f"[MedianPruner] Logs saved to {tar_path}")

                early_stop = True
                proc.kill(); proc.wait();
                break

    proc.wait()
    toc = time.time()

    if early_stop:
        proc_returncode = 0  # treat as completed
    else:
        proc_returncode = proc.returncode

    if proc_returncode != 0:
        raise RuntimeError(f"train_evaluate exited with code {proc_returncode}")

    # 累积每个 epoch 打印的指标，稍后按与 val_auc 对应的索引取值
    val_auc_list, test_auc_list, train_auc_list = [], [], []
    epoch_of_val = []  # 记录对应的 epoch 编号
    val_aupr_list, test_aupr_list = [], []
    val_acc_list, test_acc_list = [], []
    loss_list = []
    for line in stdout:
        m1 = VAL_AUC_RE.search(line)
        m2 = TEST_AUC_RE.search(line)
        m3 = VAL_AUPRC_RE.search(line)
        m4 = TEST_AUPRC_RE.search(line)
        m5 = VAL_ACC_RE.search(line)
        m6 = TEST_ACC_RE.search(line)
        m7 = TRAIN_AUC_RE.search(line)
        m8 = LOSS_RE.search(line)

        if m1:
            val_auc_list.append(float(m1.group(1)))
            epoch_of_val.append(epoch_idx if epoch_idx>=0 else None)
        if m2:
            test_auc_list.append(float(m2.group(1)))
        if m3:
            val_aupr_list.append(float(m3.group(1)))
        if m4:
            test_aupr_list.append(float(m4.group(1)))
        if m5:
            val_acc_list.append(float(m5.group(1)))
        if m6:
            test_acc_list.append(float(m6.group(1)))
        if m7:
            train_auc_list.append(float(m7.group(1)))
        if m8:
            loss_list.append(float(m8.group(1)))
    if not val_auc_list:
        raise RuntimeError("Failed to parse Val AUROC from output.\nLast 20 lines:\n" + "\n".join(stdout[-20:]))

    # 取最佳 Val AUROC 所在 epoch 的其它指标
    best_idx = val_auc_list.index(max(val_auc_list))

    result = {
        # val_auc 直接取最大值；其它指标对齐到 same epoch
        "val_auc": max(val_auc_list),
        "test_auc": test_auc_list[best_idx] if len(test_auc_list) > best_idx else -1,
        "train_auc": train_auc_list[best_idx] if len(train_auc_list) > best_idx else -1,
        "val_aupr": val_aupr_list[best_idx] if len(val_aupr_list) > best_idx else -1,
        "test_aupr": test_aupr_list[best_idx] if len(test_aupr_list) > best_idx else -1,
        "val_acc": val_acc_list[best_idx] if len(val_acc_list) > best_idx else -1,
        "test_acc": test_acc_list[best_idx] if len(test_acc_list) > best_idx else -1,
        "epoch": (epoch_of_val[best_idx] if best_idx < len(epoch_of_val) and epoch_of_val[best_idx] is not None else best_idx),
        "loss": loss_list[best_idx] if len(loss_list) > best_idx else (loss_list[-1] if loss_list else -1),
        "time": toc - tic,
        "early": early_stop,
    }
    return result


# ---------------------- Optuna objective -------------------

def objective(trial: optuna.Trial):
    params = {
        "lr":      trial.suggest_float("lr", 3e-6, 3e-4, log=True),
        "batch":   trial.suggest_int("batch", 192, 320, step=32),
        "dropout": trial.suggest_float("dropout", 0.04, 0.15),
        "dist":    trial.suggest_float("dist", 0.5, 2.0, log=True),

        "beta":    trial.suggest_float("beta", 0.5, 1.5, log=True),
        "wd":      trial.suggest_float("wd", 1e-6, 1e-4, log=True),
        "mamba_pool_drop": trial.suggest_float("mamba_pool_drop", 0.06, 0.2),
        "motif_drop": trial.suggest_float("motif_drop", 0.06, 0.2),
        "gat_drop": trial.suggest_float("gat_drop", 0.1, 0.55),
        "atom_drop": trial.suggest_float("atom_drop", 0.05, 0.2),
        "gin_lr": trial.suggest_float("gin_lr", 1e-5, 5e-4, log=True),
    }


    try:
        metrics = run_once(params, trial)
        state = "early_stop" if metrics.get("early", False) else "ok"
        err_msg = ""
    except optuna.TrialPruned as e:
        # Trial 被剪枝：记录最后已上报的指标（若有），state=pruned
        metrics = {
            "train_auc": -1, "val_auc": -1, "test_auc": -1,
            "val_aupr": -1, "test_aupr": -1,
            "val_acc": -1, "test_acc": -1,
            "epoch": -1,
            "loss": -1, "time": 0,
        }
        state = "pruned"
        err_msg = "pruned"
        # 仍需写 CSV，然后重新抛让 Optuna 记录 PRUNED
        header = ["time"] + list(params.keys()) + [
            "train_auc", "val_auc", "test_auc", "val_aupr", "test_aupr", "val_acc", "test_acc", "epoch", "loss", "sec", "state", "err"
        ]
        row = [dt.datetime.now().isoformat()] + [params[k] for k in params] + [
            metrics["train_auc"], metrics["val_auc"], metrics["test_auc"],
            metrics["val_aupr"], metrics["test_aupr"],
            metrics["val_acc"], metrics["test_acc"], metrics["epoch"],
            metrics["loss"], metrics["time"], state, err_msg
        ]
        new_file = not Path(LOG_CSV).exists()
        with open(LOG_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(header)
            w.writerow(row)
        raise  # 让 Optuna 维持 PRUNED 状态
    except Exception as e:
        # 其它类型失败
        metrics = {
            "train_auc": -1, "val_auc": -1, "test_auc": -1,
            "val_aupr": -1, "test_aupr": -1,
            "val_acc": -1, "test_acc": -1,
            "epoch": -1,
            "loss": -1, "time": 0,
        }
        state = "fail"
        err_msg = str(e).split("\n")[0][:120]

    # log to csv（无论成功或失败）
    header = ["time"] + list(params.keys()) + [
        "train_auc", "val_auc", "test_auc", "val_aupr", "test_aupr", "val_acc", "test_acc", "epoch", "loss", "sec", "state", "err"
    ]
    row = [dt.datetime.now().isoformat()] + [params[k] for k in params] + [
        metrics["train_auc"], metrics["val_auc"], metrics["test_auc"],
        metrics["val_aupr"], metrics["test_aupr"],
        metrics["val_acc"], metrics["test_acc"], metrics["epoch"],
        metrics["loss"], metrics["time"], state, err_msg
    ]
    new_file = not Path(LOG_CSV).exists()
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)

    if state == "fail":
        raise RuntimeError(err_msg)  # 让 Optuna 标记为失败，但避免 bare raise

    # return value to maximize (Test AUROC)
    return metrics["test_auc"]


if __name__ == "__main__":
    sampler = TPESampler(multivariate=True, n_startup_trials=20)
    base_pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=40, interval_steps=1)
    pruner = PatientPruner(base_pruner, patience=15)

    # ---------- callback to ignore pruned trials (Optuna >=4) ----------
    def _ignore_pruned(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        from optuna.trial import TrialState
        if trial.state == TrialState.PRUNED:
            # 清空该 trial 的值，使 TPE 不将其纳入 bad 分布
            study._storage.set_trial_values(trial._trial_id, [])  # type: ignore[attr-defined]


    try:
        study = optuna.load_study(study_name="AMES_AUROC", storage=STORAGE, sampler=sampler, pruner=pruner)
        print(f"[Resume] Loaded existing study with {len(study.trials)} trials.")
    except KeyError:
        study = optuna.create_study(direction="maximize", study_name="AMES_AUROC",
                                    storage=STORAGE, sampler=sampler, pruner=pruner)
        print("[New] Created new study AMES_AUROC")

    # ---- Enqueue default parameter configuration as a baseline trial ----
    default_params = {
        "lr": 8.531564807673896e-6,
        "batch": 256,
        "dropout": 0.08743600355161327,
        "dist": 1.0243053126755768,

        "beta": 1.0,
        "wd": 7.962629641951952e-6,
        "mamba_pool_drop": 0.10,
        "motif_drop": 0.12,
        "gat_drop": 0.5,
        "atom_drop": 0.12,
        "gin_lr": 2e-4,
    }
    # 仅在该参数组合尚未存在时加入队列
    if all(t.params != default_params for t in study.trials):
        study.enqueue_trial(default_params)

    TOTAL_TRIALS = 200
    remaining = TOTAL_TRIALS - len([t for t in study.trials if t.state.is_finished()])
    if remaining <= 0:
        print("Study already has", TOTAL_TRIALS, "completed trials. Nothing to do.")
    else:
        print(f"Starting optimization for {remaining} remaining trials…")
        study.optimize(objective, n_trials=remaining, callbacks=[_ignore_pruned])

    print("Best trial:")
    print("  Value (Test AUROC):", study.best_trial.value)
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}") 