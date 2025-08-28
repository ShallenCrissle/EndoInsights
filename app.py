# app.py — EndoInsights (Merged Frontend + Backend Inference)
# Run: streamlit run app.py

import streamlit as st
import torch
import cv2
import io
import base64
import numpy as np
from PIL import Image
from backend.test import load_model, draw_saliency_map, draw_bounding_boxes, estimate_severity, val_transforms

# ------------------ UTILS ------------------ #
def get_base64_image(image_file):
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="EndoInsights — Endometriosis Detection",
    page_icon="🩺",
    layout="wide"
)

bg_image = get_base64_image("frontend/static/banner.jpg")

# ------------------ CLINICAL THEME ------------------ #
st.markdown(f"""
<style>
.stApp {{
  background-color: #eaf6ff !important;
  background-image: linear-gradient(180deg, #f3faff, #eaf6ff) !important;
}}
section[data-testid="stSidebar"] {{
  background-color: #dff1fb !important;
  color: #0a3b5a !important;
}}
.card {{
  background: #ffffff;
  padding: 25px 30px;
  border-radius: 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
  margin-bottom: 25px;
  border: 1px solid #eef5fb;
}}
h1, h2, h3, h4 {{
  color: #0a3b5a !important;
  font-family: 'Segoe UI', system-ui, Roboto, 'Open Sans', sans-serif !important;
}}
p, li, label, span, div {{
  color: #143e56;
  font-family: 'Segoe UI', system-ui, Roboto, 'Open Sans', sans-serif !important;
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(90deg, #0f8bb3, #28b4de) !important;
  color: #fff !important;
  border-radius: 12px !important;
  padding: 0.6rem 1.2rem !important;
  font-weight: 600 !important;
}}
.banner {{
  position: relative;
  background: url("data:image/jpg;base64,{bg_image}") no-repeat center center;
  background-size: cover;
  background-position: center;
  height: 250px;
  border-radius: 14px;
  margin-bottom: 22px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}}
.banner-overlay {{
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background: rgba(0,0,0,0.4); /* dark overlay for readability */
  border-radius: 14px;
}}
.banner-text {{
    position: absolute;
    top: 20px;
    left: 30%;
    transform: translateX(-60%);
    color: white;
    font-size: 36px;
    font-weight: bold;
}}

.banner-text h2 {{
  margin: 0;
  font-size: 34px;
  font-weight: 800;
  letter-spacing: 0.5px;
}}
.banner-text p {{
  margin: 8px 0 0 0;
  font-size: 14px;
  font-weight: 600;
  opacity: 0.95;
}}
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------ #
st.sidebar.title("EndoInsights")
st.sidebar.caption("Clinical AI for Endometriosis")
st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.markdown("**Quick Links**\n- Home\n- Upload Scan\n- Results\n- About")

# ------------------ HEADER ------------------ #
st.markdown(f"""
<div class="banner">
  <div class="banner-overlay"></div>
  <div class="banner-text">
    <h2>EndoInsights — <br> Endometriosis Detection</h2>
    <p>AI-assisted clinical decision support for ultrasound and laparoscopic scans.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------ TABS ------------------ #
tabs = st.tabs(["🏠 Home", "📂 Upload Scan", "📊 Results", "ℹ️ About"])
ss = st.session_state
for k in ["uploaded_image", "diagnosis_label", "confidence_pct", "severity_score",
          "probs_dict", "bbox_image_bgr", "saliency_image"]:
    ss.setdefault(k, None)

# ------------------ LOAD MODEL ------------------ #
MODEL_PATH = r"D:\ZENDO\model\cnnvit_endometriosis_final.pth"
CLASS_MAP_PATH = r"D:\ZENDO\model\class_to_idx.json"

model, class_to_idx, device = load_model(MODEL_PATH, CLASS_MAP_PATH)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ------------------ HOME TAB ------------------ #
with tabs[0]:
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Welcome to EndoInsights")
    st.write("""
EndoInsights is a **state-of-the-art AI platform** designed to assist clinicians in detecting **endometriosis** 
from ultrasound and laparoscopic images.  
With a hybrid **CNN + Vision Transformer model**, it provides **high-confidence predictions**, **severity estimation**, 
and **visual explainability** for each scan.  

**How it works:**
- Upload a patient scan (ultrasound or laparoscopic image).
- Run AI analysis to detect potential pathology.
- Review probabilities, severity score, and heatmaps for explainability.

EndoInsights aims to enhance clinical decision-making with **transparent and interpretable AI insights**.
""")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ UPLOAD TAB ------------------ #
with tabs[1]:
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Patient Scan")
    uploaded = st.file_uploader("Choose a scan (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        image_bytes = uploaded.read()
        ss.uploaded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(ss.uploaded_image, caption="Uploaded Scan", use_container_width=True)

        st.success("Scan uploaded successfully.")

        run = st.button("🔍 Run Analysis", type="primary")
        if run:
            # Real inference
            img_tensor = val_transforms(ss.uploaded_image).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            predicted_class = idx_to_class[pred.item()]
            confidence = conf.item()

            ss.diagnosis_label = predicted_class
            ss.confidence_pct = confidence * 100
            ss.probs_dict = {idx_to_class[i]: p.item() for i, p in enumerate(probs[0])}

            img_np = np.array(ss.uploaded_image)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            total_area = img_color.shape[0] * img_color.shape[1]

            if predicted_class.lower() == "pathology":
                saliency = draw_saliency_map(img_tensor.clone(), model)
                salient_area, bbox = draw_bounding_boxes(saliency, img_color.shape)

                if bbox:
                    cv2.rectangle(img_color, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                severity_score = estimate_severity(confidence, salient_area, total_area)
                ss.severity_score = severity_score
                ss.bbox_image_bgr = img_color
                ss.saliency_image = saliency
            else:
                ss.severity_score = None
                ss.bbox_image_bgr = img_color
                ss.saliency_image = None
            st.success("Analysis complete. Check the Results tab.")
    else:
        st.info("Upload a scan to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ RESULTS TAB ------------------ #
with tabs[2]:
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("AI-Based Diagnostic Report")

    if ss.diagnosis_label is None:
        st.warning("No results yet. Upload a scan and run analysis in the Upload tab.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Diagnosis", ss.diagnosis_label)
        if ss.confidence_pct is not None:
            m2.metric("Confidence", f"{ss.confidence_pct:.1f}%")
        if ss.severity_score is not None:
            m3.metric("Severity Score", f"{ss.severity_score:.2f}")

        st.markdown("**Class Probabilities**")
        for cls, p in ss.probs_dict.items():
            pct = int(round(p * 100))
            st.write(f"{cls}: **{pct}%**")
            st.progress(pct)

        st.markdown("**Explainability Visuals**")
        c1, c2 = st.columns(2)
        with c1:
            if ss.bbox_image_bgr is not None:
                st.image(ss.bbox_image_bgr, channels="BGR", caption="Bounding Box Overlay")
        with c2:
            if ss.saliency_image is not None:
                st.image(ss.saliency_image, caption="Saliency Map", clamp=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ ABOUT TAB ------------------ #
with tabs[3]:
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About EndoInsights")
    st.write("""
**EndoInsights** is a **research-focused clinical AI platform** designed to assist in the detection and assessment 
of endometriosis. It leverages a hybrid **Convolutional Neural Network + Vision Transformer** model to analyze 
ultrasound and laparoscopic images with high precision.

**Key Features:**
- AI-powered detection of endometriosis and related pathologies.
- Severity scoring based on lesion area and confidence.
- Explainable heatmaps for clinicians to understand model focus.
- Clean, intuitive, and medically themed UI for rapid adoption.

Our goal is to **enhance clinical decision-making** while maintaining transparency, interpretability, 
and ease of use for healthcare professionals.
""")
    st.markdown('</div>', unsafe_allow_html=True)
