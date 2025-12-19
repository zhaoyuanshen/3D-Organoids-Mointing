import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 加载训练历史
try:
    # 首先尝试安全加载
    history = torch.load('training_history.pth', weights_only=True)
except:
    # 如果失败，使用 weights_only=False（确保文件可信）
    history = torch.load('training_history.pth', weights_only=False)

# 查看最佳epoch的指标
print(f"Best F1: {max(history['valid_f1']):.4f}")
print(f"Best Accuracy: {max(history['valid_acc']):.4f}")
print(f"Best Kappa: {max(history['valid_kappa']):.4f}")

# 获取 epoch 数量
num_epochs = len(history['train_loss'])
epochs = range(1, num_epochs + 1)

# 找到最佳epoch
best_f1_epoch = np.argmax(history['valid_f1']) + 1
best_acc_epoch = np.argmax(history['valid_acc']) + 1
best_kappa_epoch = np.argmax(history['valid_kappa']) + 1

print(f"\n最佳F1在第 {best_f1_epoch} 个epoch: {max(history['valid_f1']):.4f}")
print(f"最佳准确率在第 {best_acc_epoch} 个epoch: {max(history['valid_acc']):.4f}")
print(f"最佳Kappa在第 {best_kappa_epoch} 个epoch: {max(history['valid_kappa']):.4f}")

# 绘制训练历史
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 损失曲线
ax1.plot(epochs, history['train_loss'], label='Train Loss')
ax1.plot(epochs, history['valid_loss'], label='Valid Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Loss History')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))


# 指标曲线
ax2.plot(epochs, history['valid_acc'], label='Valid Acc')
ax2.plot(epochs, history['valid_f1'], label='Valid F1')
ax2.plot(epochs, history['valid_kappa'], label='Valid Kappa')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Metric')
ax2.legend()
ax2.set_title('Validation Metrics')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

# 损失图：一般标记验证集损失最小点
best_loss_epoch = np.argmin(history['valid_loss']) + 1
ax1.axvline(best_loss_epoch, color='red', linestyle='--', alpha=0.7)
ax1.text(best_loss_epoch, max(history['valid_loss']), f'Best Loss @ {best_loss_epoch}',
         rotation=90, verticalalignment='bottom', color='red')

# 指标图：分别标记 F1、Acc、Kappa
ax2.axvline(best_f1_epoch, color='red', linestyle='--', alpha=0.7)
ax2.text(best_f1_epoch, max(history['valid_f1']), f'F1@{best_f1_epoch}',
         rotation=90, verticalalignment='bottom', color='red')

# ax2.axvline(best_acc_epoch, color='blue', linestyle='--', alpha=0.7)
# ax2.text(best_acc_epoch, max(history['valid_acc']), f'Acc@{best_acc_epoch}',
#          rotation=90, verticalalignment='bottom', color='blue')

# ax2.axvline(best_kappa_epoch, color='purple', linestyle='--', alpha=0.7)
# ax2.text(best_kappa_epoch, max(history['valid_kappa']), f'Kappa@{best_kappa_epoch}',
#          rotation=90, verticalalignment='bottom', color='purple')

plt.tight_layout()
plt.savefig('training_history_analysis.png', dpi=300)
plt.show()