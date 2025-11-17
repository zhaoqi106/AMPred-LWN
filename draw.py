import matplotlib.pyplot as plt
import os

import numpy as np

# 读取结果目录下所有 test*.txt 文件
results_dir = 'results/AMES'
model_name = os.path.basename(results_dir)  # 用于命名输出文件
test_files = sorted([f for f in os.listdir(results_dir) if f.startswith('test') and f.endswith('.txt')])
if not test_files:
    raise FileNotFoundError(f'在 {results_dir} 未找到 test*.txt 结果文件，请确认训练日志文件存在')

Test_all = []  # 每折的测试 AUROC 序列
for tf in test_files:
    with open(os.path.join(results_dir, tf), 'r') as f:
        fold_aucs = []
        for line in f:
            if line.startswith('##########folds:'):
                # 遇到新的折标记，先把上一折保存
                if fold_aucs:
                    Test_all.append(fold_aucs)
                    fold_aucs = []
                continue
            if 'Test AUROC:' in line:
                try:
                    value = float(line.strip().split('Test AUROC:')[-1])
                    fold_aucs.append(value)
                except ValueError:
                    continue
        # 文件结束后再保存最后一折
        if fold_aucs:
            Test_all.append(fold_aucs)

if not Test_all:
    raise RuntimeError('未能从日志文件中解析到任何 AUROC 数据')

# 计算平均长度，使用最短折的长度防止不一致
min_len = min(len(fold) for fold in Test_all)
Test_all = [fold[:min_len] for fold in Test_all]

# 生成 epoch 序列
epochs = list(range(1, min_len + 1))
# 读取 best_metrics_fold*.txt
best_files = sorted([f for f in os.listdir(results_dir) if f.startswith('best_metrics_fold') and f.endswith('.txt')])
best_aucs = []
best_epochs = []
for bf in best_files:
    fold_idx = int(bf.split('best_metrics_fold')[1].split('.')[0])
    try:
        with open(os.path.join(results_dir, bf), 'r') as f:
            import json
            data = json.load(f)
            best_aucs.append((fold_idx, data['AUROC']))
            best_epochs.append((fold_idx, data['epoch'] + 1))  # epoch recorded 0-based? ensure +1
    except Exception:
        continue
# 排序确保与Test_all对应
best_aucs = [x[1] for x in sorted(best_aucs, key=lambda x: x[0])]
best_epochs = [x[1] for x in sorted(best_epochs, key=lambda x: x[0])]
avg_best = np.mean(best_aucs)

# 使用Matplotlib绘图
plt.figure(figsize=(8, 6))
# 颜色改为 Color-blind safe Set2
colors = plt.cm.Set2.colors  # 8 种颜色

# 绘制每折曲线
for idx, fold_auc in enumerate(Test_all):
    best_auc = best_aucs[idx] if idx < len(best_aucs) else max(fold_auc)
    best_epoch = best_epochs[idx] if idx < len(best_epochs) else np.argmax(fold_auc)+1
    color = colors[idx % len(colors)]
    plt.plot(epochs, fold_auc, label=f'Fold {idx + 1} AUC={best_auc:.3f} (Ep{best_epoch})',
             color=color, linewidth=1)
    # 空心圆标注最佳点
    plt.scatter(best_epoch, best_auc, facecolors='none', edgecolors=color, marker='o', s=50, linewidths=1.2, zorder=4)
    plt.axvline(best_epoch, color=color, linestyle=':', alpha=0.25, linewidth=1)

# 绘制黑色平均曲线
mean_curve = np.mean(Test_all, axis=0)
plt.plot(epochs, mean_curve, color='black', linewidth=2, label='Mean AUROC')

# 绘制平均趋势虚线
# start_value = average_auc[0]
# end_value = average_auc[-1]
# plt.plot([epochs[0], epochs[-1]], [start_value, end_value], color='purple', linestyle='--', label=f'Average AUC={end_value:.3f}')

# 绘制平均最佳AUC水平虚线
plt.axhline(avg_best, color='purple', linestyle='--', label=f'Average Best AUC={avg_best:.3f}')

plt.ylabel('AUC', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.xticks(fontsize=15)  # X轴刻度字体大小
plt.yticks(fontsize=15)  # Y轴刻度字体大小
plt.grid(True)
plt.legend(fontsize=11, loc='lower right')
directory = 'fin_result/figures/'
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig('fin_result/figures/' + model_name + '.pdf')

plt.clf()

# plt.plot(epochs, test_loss, label = 'test_auc', color = 'g')
# plt.plot(epochs, train_loss, label = 'train_auc', color = 'r')
# plt.plot(epochs, valid_loss, label = 'valid_auc', color = 'b')
# plt.xlabel('Epoch')
# plt.ylabel('loss')
# plt.title('Train & Test Loss')
# plt.grid(True)
# plt.legend()
# plt.savefig('fin_result/figures/' + model_name + '_loss' + '.pdf')
