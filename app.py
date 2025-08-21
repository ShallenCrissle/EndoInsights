import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from backend.test import load_model, draw_saliency_map, draw_bounding_boxes, estimate_severity, val_transforms

# Load model once
MODEL_PATH = r"D:\ZENDO\model\cnnvit_endometriosis_final.pth"
CLASS_MAP_PATH = r"D:\ZENDO\model\class_to_idx.json"
model, class_to_idx, device = load_model(MODEL_PATH, CLASS_MAP_PATH)
idx_to_class = {v: k for k, v in class_to_idx.items()}

st.title("🩺 Endometriosis Detection")
uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = val_transforms(image).unsqueeze(0).to(device)

    # Inference
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)
    predicted_class = idx_to_class[pred.item()]
    confidence = conf.item()

    # Convert image for OpenCV
    img_np = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    total_area = img_color.shape[0] * img_color.shape[1]

    if predicted_class.lower() == "pathology":
        saliency = draw_saliency_map(img_tensor.clone(), model)
        salient_area, bbox = draw_bounding_boxes(saliency, img_color.shape)

        if bbox:
            cv2.rectangle(img_color, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        severity_score = estimate_severity(confidence, salient_area, total_area)

        # Display results
        st.subheader(f"Prediction: Endometriosis detected ✅")
        st.write(f"Confidence: **{confidence*100:.2f}%**")
        st.write(f"Severity Score: **{severity_score:.3f}**")

        st.image(img_color, caption="Detected Bounding Box", channels="BGR")
        st.image(saliency, caption="Saliency Map", clamp=True)

    else:
        st.subheader(f"Prediction: No Endometriosis ❌")
        st.write(f"Confidence: **{confidence*100:.2f}%**")
        st.image(img_gray, caption="Ultrasound Image", channels="GRAY")
