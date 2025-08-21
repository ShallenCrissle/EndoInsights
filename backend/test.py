import torch
import torch.nn as nn
from torchvision import models, transforms
import timm
import json
import cv2
import numpy as np

# Model Definition
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

# Model Loader
def load_model(model_path, class_map_path):
    """
    Load trained CNN+ViT hybrid model and class mapping.
    """
    with open(class_map_path, "r") as f:
        class_to_idx = json.load(f)

    num_classes = len(class_to_idx)
    model = CNNViTHybrid(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, class_to_idx, device

# Saliency Map Generator
def draw_saliency_map(img_tensor, model):
    """
    Generate saliency map for an image tensor.
    """
    img_tensor.requires_grad_()
    output = model(img_tensor)
    score, _ = torch.max(output, 1)
    score.backward()
    saliency = img_tensor.grad.data.abs().max(dim=1)[0].cpu().numpy()[0]
    return saliency

# Bounding Box Drawer
def draw_bounding_boxes(saliency, image_shape):
    """
    Draw bounding box from saliency map.
    Returns: salient_area, (min_x, min_y, max_x, max_y) or None
    """
    thresh = (saliency > saliency.mean() + saliency.std()).astype(np.uint8) * 255
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, None

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

    return salient_area, (min_x, min_y, max_x, max_y)

# Severity Estimation
def estimate_severity(confidence, salient_area, total_area):
    """
    Calculate severity score as a function of confidence and salient area ratio.
    """
    return confidence * (salient_area / total_area)

# Image Transform (for app)
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
