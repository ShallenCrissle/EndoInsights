# app.py — EndoInsights (Frontend-Only, Clinical UI)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
import numpy as np
def get_base64_image(image_file):
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="EndoInsights — Endometriosis Detection",
    page_icon="🩺",
    layout="wide"
)
bg_image = get_base64_image("static/banner.jpg")
# ------------------ CLINICAL THEME (CSS) ------------------ #
st.markdown(f"""
<style>
/* Force light, clinical background */
.stApp {{
  background-color: #eaf6ff !important; /* very light medical blue */
  background-image: linear-gradient(180deg, #f3faff, #eaf6ff) !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background-color: #dff1fb !important; /* pale blue */
  color: #0a3b5a !important;
}}

/* Cards */
.card {{
  background: #ffffff;
  padding: 20px;
  border-radius: 14px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
  margin-bottom: 18px;
  border: 1px solid #eef5fb;
}}

/* Headers & text */
h1, h2, h3, h4 {{
  color: #0a3b5a !important; /* deep clinical blue */
  font-family: 'Segoe UI', system-ui, -apple-system, Roboto, 'Open Sans', sans-serif !important;
}}
p, li, label, span, div {{
  color: #143e56;
  font-family: 'Segoe UI', system-ui, -apple-system, Roboto, 'Open Sans', sans-serif !important;
}}

/* Primary buttons */
.stButton > button[kind="primary"] {{
  background: linear-gradient(90deg, #0f8bb3, #28b4de) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.6rem 1.1rem !important;
  font-weight: 600 !important;
}}
.stButton > button[kind="primary"]:hover {{
  filter: brightness(0.95);
}}

/* File uploader */
[data-testid="stFileUploader"] {{
  background-color: #f7fbff !important;
  border: 2px dashed #9dd7f2 !important;
  border-radius: 12px !important;
  padding: 1rem !important;
}}

/* Tabs look */
.stTabs [data-baseweb="tab-list"] {{
  gap: 6px;
}}
.stTabs [data-baseweb="tab"] {{
  background: #dff1fb;
  border-radius: 10px 10px 0 0;
}}
.stTabs [aria-selected="true"] {{
  background: #ffffff !important;
  color: #0a3b5a !important;
  border: 1px solid #d9ecf9 !important;
  border-bottom: 0 !important;
}}

/* Metrics */
[data-testid="stMetricValue"] {{
  color: #0a3b5a !important;
}}

/* Subtle separators */
.hr {{
  height: 1px;
  background: linear-gradient(90deg, transparent, #cfe6f7, transparent);
  border: 0;
  margin: 12px 0 4px 0;
}}

/* Banner styles */
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
st.sidebar.caption("Clinical AI for Endometriosis (Frontend UI)")

# Demo mode
demo_mode = st.sidebar.toggle("Demo mode (show sample output)", value=True)

st.sidebar.markdown("<hr class='hr'/>", unsafe_allow_html=True)
st.sidebar.markdown("**Quick links**")
st.sidebar.markdown("- Home\n- Upload\n- Results\n- About")

# ------------------ HEADER BANNER ------------------ #
st.markdown("""
<div class="banner">
  <div class="banner-overlay"></div>
  <div class="banner-text">
    <h2> EndoInsights — <br> Endometriosis Detection</h2>
    <p>Clinical decision-support UI for analyzing ultrasound/laparoscopic images with AI.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------ NAVIGATION (TABS) ------------------ #
tabs = st.tabs(["🏠 Home", "📂 Upload Scan", "📊 Results", "ℹ️ About"])

# ------------------ SESSION STATE KEYS ------------------ #
ss = st.session_state
for k in ["uploaded_image", "diagnosis_label", "confidence_pct", "severity_score",
          "probs_dict", "bbox_image_bgr", "saliency_image"]:
    ss.setdefault(k, None)

# ------------------ HOME TAB ------------------ #
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Welcome")
    st.write(
        "This interface presents AI-assisted analysis for **endometriosis detection**. "
        "Upload a scan and review probability-based results, severity estimation, and visual explanations."
    )
    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
    colA, colB, colC = st.columns([1.2, 1, 1])
    with colA:
        st.markdown("**Workflow**")
        st.write("1) Upload ultrasound/laparoscopic image\n2) Run analysis\n3) Review results & visuals")
    with colB:
        st.markdown("**Outputs**")
        st.write("- Diagnosis label\n- Confidence (%)\n- Severity score\n- Class probabilities")
    with colC:
        st.markdown("**Explainability**")
        st.write("- Bounding box on suspicious region\n- Saliency/attention map")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ UPLOAD TAB ------------------ #
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Patient Scan")
    uploaded = st.file_uploader("Choose a scan (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        image_bytes = uploaded.read()
        ss.uploaded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(ss.uploaded_image, caption="Uploaded Scan", use_column_width=True)
        st.success("Scan uploaded successfully.")

        run = st.button("🔍 Run Analysis", type="primary")
        if run:
            if demo_mode:
                ss.diagnosis_label = "Pathology"
                ss.confidence_pct = 84.3
                ss.severity_score = 0.62
                ss.probs_dict = {
                    "Endometriosis (Pathology)": 0.843,
                    "No Pathology": 0.157
                }
                np_img = np.array(ss.uploaded_image)
                bbox = np_img.copy()
                h, w, _ = bbox.shape
                x1, y1, x2, y2 = int(w*0.28), int(h*0.28), int(w*0.72), int(h*0.65)
                overlay = bbox.copy()
                overlay[y1:y2, x1:x2] = [0, 180, 255]
                bbox = (0.65*overlay + 0.35*bbox).astype(np.uint8)
                ss.bbox_image_bgr = bbox[:, :, ::-1]

                gray = np.mean(np_img, axis=2)
                heat = np.stack([gray*0.8, gray*0.9, gray], axis=2)
                heat = np.clip(heat, 0, 255).astype(np.uint8)
                ss.saliency_image = heat
                st.success("Analysis complete (demo). Check the Results tab.")
            else:
                st.info("Frontend ready. Connect the backend to populate session_state.")
    else:
        st.info("Upload a scan to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ RESULTS TAB ------------------ #
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("AI-Based Diagnostic Report")

    if ss.diagnosis_label is None:
        st.warning("No results yet. Upload a scan and run analysis in the **Upload Scan** tab.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Diagnosis", ss.diagnosis_label)
        if ss.confidence_pct is not None:
            m2.metric("Confidence", f"{ss.confidence_pct:.1f}%")
        if ss.severity_score is not None:
            m3.metric("Severity Score", f"{ss.severity_score:.2f}")

        st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

        if ss.probs_dict:
            st.markdown("**Class Probabilities**")
            for cls, p in ss.probs_dict.items():
                pct = int(round(p * 100))
                st.write(f"{cls}: **{pct}%**")
                st.progress(pct)

            df = pd.DataFrame(
                {"Class": list(ss.probs_dict.keys()),
                 "Probability": [f"{v*100:.1f}%" for v in ss.probs_dict.values()]}
            )
            st.table(df)

        st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
        st.markdown("**Explainability Visuals**")
        c1, c2 = st.columns(2)
        with c1:
            if ss.bbox_image_bgr is not None:
                st.image(ss.bbox_image_bgr, channels="BGR", caption="Detected Region (Bounding Box Overlay)", use_column_width=True)
            elif ss.uploaded_image is not None:
                st.image(ss.uploaded_image, caption="Scan (No BBox available)", use_column_width=True)
        with c2:
            if ss.saliency_image is not None:
                if ss.saliency_image.ndim == 2:
                    st.image(ss.saliency_image, caption="Saliency Map", use_column_width=True)
                else:
                    st.image(ss.saliency_image, caption="Saliency Map", channels="RGB", use_column_width=True)
            else:
                st.info("Saliency map not available.")

        st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
        st.caption("⚠️ Research prototype. Not a substitute for professional medical diagnosis.")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ ABOUT TAB ------------------ #
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About EndoInsights")
    st.write("""
**EndoInsights** leverages a Hybrid **CNN + Vision Transformer (ViT)** pipeline to assist with ultrasound/laparoscopic
image analysis for endometriosis detection. This UI focuses on clinical clarity:

- **Clean, medical palette** with professional typography  
- **Clear sections**: Upload → Results → Explainability  
- **Metrics & probabilities** for transparent decision support  

**Integration note (for your teammate):**  
Import backend functions and populate `st.session_state` keys after inference:

```python
st.session_state.diagnosis_label = "Pathology"  # or "No Pathology"
st.session_state.confidence_pct = 84.3          # float 0–100
st.session_state.severity_score = 0.62          # float 0–1
st.session_state.probs_dict = {
    "Endometriosis (Pathology)": 0.843,
    "No Pathology": 0.157
}
st.session_state.bbox_image_bgr = bbox_numpy_bgr  # HxWx3 (BGR)
st.session_state.saliency_image = saliency_numpy   # HxW or HxWx3 (RGB or gray)
""")
st.markdown('</div>', unsafe_allow_html=True)