import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import timm
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import precision_score

# Transforms
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Dataset
dataset = datasets.ImageFolder(r"D:\ZENDO\endo_dataset", transform=val_transforms)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # batch=1

# Class mapping
with open(r"D:\ZENDO\model\class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# Model
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(class_to_idx)
model = CNNViTHybrid(num_classes=num_classes)
model.load_state_dict(torch.load(r"D:\ZENDO\model\cnnvit_endometriosis_final.pth", map_location=device))
model.to(device)
model.eval()

# Saliency Map Function
def generate_saliency_map(img_tensor, model):
    img_tensor.requires_grad_()
    output = model(img_tensor)
    score, _ = torch.max(output, 1)
    score.backward()
    saliency = img_tensor.grad.data.abs().max(dim=1)[0].cpu().numpy()[0]
    return saliency

# Detection Loop
all_preds, all_labels = [], []
processed = 0

for imgs, labels in val_loader:
    if processed >= 5:
        break

    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs)
    probs = torch.softmax(outputs, dim=1)
    conf, preds = torch.max(probs, 1)

    all_preds.append(preds.item())
    all_labels.append(labels.item())

    predicted_class = idx_to_class[preds.item()]
    is_pathology = predicted_class.lower() == "pathology"

    if is_pathology:
        label_str = "Endometriosis detected"
    else:
        label_str = "No endometriosis detected"

    # Convert image for display
    img_np = imgs.cpu().squeeze().permute(1, 2, 0).numpy()
    img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    total_area = img_color.shape[0] * img_color.shape[1]

    if is_pathology:
        # Saliency + bounding box
        saliency = generate_saliency_map(imgs.clone(), model)
        thresh = (saliency > saliency.mean() + saliency.std()).astype(np.uint8) * 255
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            all_x, all_y, all_w, all_h = [], [], [], []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                all_x.append(x)
                all_y.append(y)
                all_w.append(x + w)
                all_h.append(y + h)

            min_x, min_y = min(all_x), min(all_y)
            max_x, max_y = max(all_w), max(all_h)
            salient_area = (max_x - min_x) * (max_y - min_y)

            cv2.rectangle(img_color, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)  # green box
        else:
            salient_area = 0

        severity_score = conf.item() * (salient_area / total_area)

        print(f"Image {processed+1}: {label_str} | Confidence: {conf.item()*100:.2f}%")
        print(f"Estimated Severity Score: {severity_score:.3f}")

        # Show both grayscale+box and saliency map
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img_color, cmap="gray")
        axs[0].set_title(label_str)
        axs[0].axis("off")

        axs[1].imshow(saliency, cmap="hot")
        axs[1].set_title("Saliency Map")
        axs[1].axis("off")
        plt.show()

    else:
        # No saliency/bbox for negatives
        severity_score = conf.item() * 0  # No salient area
        print(f"Image {processed+1}: {label_str} | Confidence: {conf.item()*100:.2f}%")
        print(f"Estimated Severity Score: {severity_score:.3f}")

        plt.imshow(img_gray, cmap="gray")
        plt.title(label_str)
        plt.axis("off")
        plt.show()

    processed += 1

# Precision
precision = precision_score(all_labels, all_preds, average='binary', pos_label=class_to_idx["pathology"])
print(f"Precision (on first 5 images): {precision:.2f}")
