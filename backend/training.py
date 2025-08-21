# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from backend.initial import train_transforms, val_transforms,model,train_loader,device,optimizer,criterion,val_loader,dataset
import os
import shutil
import json

#Training Loop with 30 epochs
epochs = 30
for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

# Validation
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")
  
# Save checkpoint after every epoch
    checkpoint_path = f"/content/drive/MyDrive/cnnvit_endometriosis_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# Save the final model
final_model_path = "/content/drive/MyDrive/cnnvit_endometriosis_final.pth"
torch.save(model.state_dict(), final_model_path)
print("Final model saved!")

#Save the class mapping
class_to_idx = dataset.class_to_idx
with open("/content/drive/MyDrive/class_to_idx.json", "w") as f:
    json.dump(class_to_idx, f)
print("Class mapping saved!")