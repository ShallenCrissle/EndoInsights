#Importing Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
import os
import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

#Performing Data Augmentation to convert to gray-scale 
#Using torchvision transforms for data augmentation
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # grayscale but keep 3 channels
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Rearranging the dataset into a format that PyTorch’s ImageFolder can read
base_dir = r"D:\ZENDO\dataset\GLENDA_v1.0\DS"
target_dir = r"D:\ZENDO\endo_dataset"

os.makedirs(target_dir, exist_ok=True)

for class_name in ["no_pathology", "pathology"]:
    class_source = os.path.join(base_dir, class_name, "frames")
    class_target = os.path.join(target_dir, class_name)
    os.makedirs(class_target, exist_ok=True)

    for video_folder in os.listdir(class_source):
        video_path = os.path.join(class_source, video_folder)
        for img_file in os.listdir(video_path):
            shutil.copy(os.path.join(video_path, img_file), class_target)

print("Dataset ready for ImageFolder!")

# Define transforms (grayscale + image sizing)
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(r"D:\ZENDO\endo_dataset", transform=train_transforms)

# 80/20 train/validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

#Defining the CNN-Vit Hybrid Model
class CNNViTHybrid(nn.Module):
    def __init__(self, num_classes):
        super(CNNViTHybrid, self).__init__()

        # CNN backbone
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # remove final fc layer

        # Vision Transformer backbone
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # remove classifier head

        # Classifier head
        self.fc = nn.Linear(512 + 768, num_classes)  # resnet18 output=512, vit=768

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        out = self.fc(combined)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(os.listdir(r"D:\ZENDO\endo_dataset"))  # pathology, no_pathology
model = CNNViTHybrid(num_classes=num_classes).to(device)

# Define Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define Data Augmentation for training and validation
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 3 channels
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load dataset with transforms
dataset = datasets.ImageFolder(r"D:\ZENDO\endo_dataset", transform=train_transforms)
# 80/20 split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Apply val transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)