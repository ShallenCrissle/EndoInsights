import matplotlib.pyplot as plt

# ==========================
# 1. PUT YOUR VALUES HERE
# ==========================

train_loss = [
0.0718, 0.0116, 0.0036, 0.0047, 0.0025, 0.0326, 0.0017, 0.0008, 0.0028, 0.0194,
0.0115, 0.0020, 0.0029, 0.0046, 0.0031, 0.0028, 0.0002, 0.0001, 0.0019, 0.0261,
0.0029, 0.0007, 0.0069, 0.0158, 0.0014, 0.0402, 0.0127, 0.0001, 0.0001, 0.0001
]

val_loss = [
0.0046, 0.0025, 0.0028, 0.0087, 0.0166, 0.0036, 0.0038, 0.0031, 0.0066, 0.0317,
0.0221, 0.0149, 0.0081, 0.0085, 0.0067, 0.0007, 0.0003, 0.0010, 0.0032, 0.0374,
0.0101, 0.0087, 0.0036, 0.0178, 0.0034, 0.0018, 0.0014, 0.0034, 0.0031, 0.0023
]

train_acc = [
97.67, 99.64, 99.91, 99.82, 99.93, 99.09, 99.96, 99.98, 99.89, 99.56,
99.60, 99.96, 99.93, 99.87, 99.89, 99.98, 100.00, 100.00, 99.91, 99.22,
99.93, 100.00, 99.96, 99.53, 99.96, 99.93, 99.67, 100.00, 100.00, 100.00
]

val_acc = [
100.00, 99.91, 99.91, 99.56, 99.38, 99.82, 99.91, 99.91, 99.82, 99.38,
99.47, 99.64, 99.73, 99.82, 99.82, 100.00, 100.00, 100.00, 99.91, 99.02,
99.64, 99.56, 99.82, 99.64, 99.82, 99.91, 99.91, 99.82, 99.82, 99.82
]

best_epoch = val_loss.index(min(val_loss)) + 1

epochs = range(1, best_epoch + 1)

# ==========================
# LOSS CURVE (EARLY STOPPED)
# ==========================
plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss[:best_epoch], label="Train Loss")
plt.plot(epochs, val_loss[:best_epoch], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training vs Validation Loss (Early Stopped at Epoch {best_epoch})")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# ACCURACY CURVE (EARLY STOPPED)
# ==========================
plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc[:best_epoch], label="Train Accuracy")
plt.plot(epochs, val_acc[:best_epoch], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(f"Training vs Validation Accuracy (Early Stopped at Epoch {best_epoch})")
plt.legend()
plt.grid(True)
plt.show()