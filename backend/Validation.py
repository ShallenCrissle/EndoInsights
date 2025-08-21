import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import timm
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#  Validation transforms
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load dataset and split
dataset = datasets.ImageFolder(r"D:\ZENDO\endo_dataset", transform=val_transforms)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load class mapping
with open(r"D:\ZENDO\model\class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Model definition
class CNNViTHybrid(nn.Module):
    def __init__(self, num_classes):
        super(CNNViTHybrid, self).__init__()
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Identity()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.vit.head = nn.Identity()
        self.fc = nn.Linear(512 + 768, num_classes)

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        return self.fc(combined)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_to_idx)
model = CNNViTHybrid(num_classes=num_classes)
model.load_state_dict(torch.load(r"D:\ZENDO\model\cnnvit_endometriosis_final.pth", map_location=device))
model.to(device)
model.eval()

# Run validation
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy
correct = sum([p == l for p, l in zip(all_preds, all_labels)])
val_acc = 100 * correct / len(all_labels)
print(f"Validation Accuracy: {val_acc:.2f}%")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(idx_to_class.values()))
disp.plot(cmap=plt.cm.Blues)
plt.show()