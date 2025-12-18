import cv2
import numpy as np
import torch
import torch.nn.functional as F

def generate_gradcam(model, input_tensor, target_class, target_layer):
    """
    Generate Grad-CAM heatmap for a given model and input.
    
    Args:
        model: Trained PyTorch model
        input_tensor: Input image tensor (1, C, H, W)
        target_class: Index of predicted/desired class
        target_layer: Layer to compute Grad-CAM on (e.g., model.layer4[-1].conv3 for ResNet50)
    """
    # Forward hook
    activations = []
    def forward_hook(module, inp, out):
        activations.append(out)
    
    # Backward hook
    gradients = []
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item() if target_class is None else target_class
    score = output[:, pred_class]
    
    # Backward pass
    model.zero_grad()
    score.backward(retain_graph=True)
    
    # Get activations & gradients
    acts = activations[0].detach()
    grads = gradients[0].detach()
    
    # Global average pooling of gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)
    
    # Weighted sum of activations
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    
    # Normalize to [0,1]
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    
    # Cleanup
    handle_fwd.remove()
    handle_bwd.remove()
    
    return cam
