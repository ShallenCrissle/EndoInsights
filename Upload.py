# app.py — EndoInsights (Full polished UI + backend)
# Run: streamlit run app.py

import streamlit as st
import torch
import cv2
import io
import os
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.test import load_model, draw_saliency_map, draw_bounding_boxes, estimate_severity, val_transforms
from backend.gradcam import generate_gradcam
from dotenv import load_dotenv
import os
import requests
load_dotenv()  # Load environment variables from .env
from groq import Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq()

# GROQ_API_KEY = "your_groq_api_key_here"


# ------------------ HELPERS ------------------ #
import cv2
import numpy as np

def calculate_lesion_dimensions(bbox, pixel_to_cm=None):
    """
    Calculates lesion width, height, and area from bounding box.
    bbox = (x_min, y_min, x_max, y_max)
    If pixel_to_cm is given, converts dimensions to cm.
    """
    x_min, y_min, x_max, y_max = bbox
    width_px = x_max - x_min
    height_px = y_max - y_min
    area_px = width_px * height_px

    if pixel_to_cm:
        width_cm = width_px * pixel_to_cm
        height_cm = height_px * pixel_to_cm
        area_cm2 = area_px * (pixel_to_cm ** 2)
        return width_cm, height_cm, area_cm2
    else:
        return width_px, height_px, area_px

def image_file_to_base64(path_or_pil):
    if isinstance(path_or_pil, str):
        if not os.path.exists(path_or_pil):
            return None
        img = Image.open(path_or_pil).convert("RGB")
    else:
        img = path_or_pil.convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def make_placeholder(initials: str = "UI", size=(320, 320), bgcolor=(220, 235, 245)):
    img = Image.new("RGB", size, bgcolor)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", int(size[0] * 0.28))
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), initials, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size[0]-w)/2, (size[1]-h)/2 -10), initials, fill=(20,60,90), font=font)
    return img

def get_image_b64_or_placeholder(filepath: str, name_for_initials: str):
    b64 = image_file_to_base64(filepath)
    if b64:
        return b64
    initials = "".join([part[0] for part in name_for_initials.split()][:2]).upper()
    placeholder = make_placeholder(initials)
    return image_file_to_base64(placeholder)

def safe_mailto(to_email: str, subject: str, body: str):
    import urllib.parse
    return f"mailto:{to_email}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="EndoInsights — Endometriosis Detection",
    page_icon="🩺",
    layout="wide"
)

# ------------------ STATIC ASSETS ------------------ #
BANNER_PATH = "frontend/static/banner.jpg"
HOME_ILLUST = "frontend/static/banner.png"
WORKFLOW_IMG = "frontend/static/banner.png"
UPLOAD_BANNER = "frontend/static/banner.png"
ABOUT_BANNER = "frontend/static/banner.png"


bg_b64 = get_image_b64_or_placeholder(BANNER_PATH, "EN")

# ------------------ CSS ------------------ #
st.markdown("""
<style>
 

/* === GLOBAL TOP PADDING TO AVOID NAVBAR OVERLAP === */
.main, .block-container, .stAppViewContainer, .stApp {
    padding-top: 120px !important;
}

/* === NAVBAR (Same as About Page) === */
.navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100px;
    background: black;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0px 60px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    z-index: 9999 !important;
}

/* Title Style */
.navbar-title {
    font-size: 54px;
    font-weight: 1900;
    color: white !important;
    letter-spacing: 8px;
    margin: 0px !important;
}

/* Link Container */
.nav-links {
    display: flex;
    align-items: center;
}

/* Navbar Links */
.nav-links a {
    color: white !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    margin: 0 16px !important;
    font-size: 20px !important;
}

/* Hover Effect */
.nav-links a:hover {
    color: #58c2ff !important;
    text-shadow: 0 0 10px #58c2ff;
}

/* Active Link Styling (optional) */
.nav-links a:active {
    color: #90e0ff !important;
}

       
.stApp {{
  background-color: #eaf6ff !important;
  background-image: linear-gradient(180deg, #f6fbff, #eaf6ff) !important;
  color: #0b3b52;
  font-family: 'Segoe UI', Roboto, Arial, sans-serif;
}}

section[data-testid="stSidebar"] {{
  background-color: #e7f6fb !important;
  color: #05364a !important;
  border-radius: 10px;
}}

.card {{
  background: #ffffff;
  padding: 18px 22px;
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(10,40,60,0.06);
  margin-bottom: 20px;
  margin-top: 0px;
  border: 1px solid #eef7fb;
}}

h1,h2,h3,h4 {{
  color: #0b3b52 !important;
}}

.stButton > button[kind="primary"] {{
  background: linear-gradient(90deg,#0f8bb3,#28b4de) !important;
  color: #fff !important;
  border-radius: 10px !important;
  padding: 0.6rem 1.1rem !important;
  font-weight: 600 !important;
}}
.stButton > button[kind="primary"]:hover {{
  filter: brightness(0.95);
}}

[data-testid="stFileUploader"] {{
  background-color: #f7fbff !important;
  border: 2px dashed #bfe8fb !important;
  border-radius: 12px !important;
  padding: 12px !important;
}}

.hero {{
  position: relative;
  background: url("data:image/jpg;base64,{bg_b64}") no-repeat center center;
  background-size: cover;
  height: 240px;
  border-radius: 14px;
  margin-bottom: 18px;
  box-shadow: 0 6px 18px rgba(10,40,60,0.06);
}}
.hero-overlay {{
  position: absolute;
  width: 100%;
  height: 100%;
  background: linear-gradient(180deg, rgba(2,27,44,0.25), rgba(2,27,44,0.08));
  border-radius: 14px;
}}
.hero-text {{
  position: absolute;
  left: 6%;
  top: 28%;
  color: #ffffff;
}}
.hero-text h2 {{
  font-size: 34px;
  margin: 0;
  letter-spacing: 0.6px;
}}
.hero-text p {{
  margin-top: 6px;
  font-weight: 600;
  opacity: 0.95;
}}

.profile-card {{
  background: #fff;
  border-radius: 12px;
  padding: 14px;
  text-align: center;
  box-shadow: 0 6px 18px rgba(3,25,40,0.06);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.profile-card:hover {{
  transform: translateY(-6px);
  box-shadow: 0 12px 28px rgba(3,25,40,0.1);
}}
.profile-card img {{
  width: 110px;
  height: 110px;
  border-radius: 50%;
  object-fit: cover;
  margin-bottom: 8px;
}}
.profile-card h4 {{ margin-bottom: 4px; }}
.profile-card p.role {{
  margin-top: 0;
  color: #25607a;
  font-weight: 600;
  font-size: 0.95rem;
}}

.hr {{
  height: 1px;
  background: linear-gradient(90deg, transparent, #cfe6f7, transparent);
  border: 0;
  margin: 12px 0 18px 0;
}}

</style>
<div class="navbar">
        <div class="navbar-title">EndoInsights</div>  
    </div>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* --- Force all image captions (Grad-CAM, Bounding Box, etc.) to black --- */
[data-testid="stImageCaption"] *,
[data-testid="stImage"] small,
[data-testid="stImage"] p,
figure p,
figcaption,
.stImage figcaption,
div[role="img"] + div p,
div[data-testid*="caption"] *,
small,
.st-emotion-cache-17z3ip7,
.st-emotion-cache-1lbz3im,
.st-emotion-cache-1xarl3l {
    color: #000 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)




# ------------------ HERO ------------------ #
st.markdown("""
<div class="banner">
  <div class="banner-overlay"></div>
  <div class="banner-text">
    <h2> EndoInsights — <br> Endometriosis Detection</h2>
    <p>Clinical decision-support UI for analyzing ultrasound/laparoscopic images with AI.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------ TABS ------------------ #
tabs = st.tabs([ "📂 Upload Scan", "📊 Results"])
ss = st.session_state
for k in ["uploaded_image","diagnosis_label","confidence_pct","severity_score","probs_dict","bbox_image_bgr","saliency_image","gradcam_image"]:
    ss.setdefault(k, None)

# ------------------ MODEL LOAD ------------------ #
MODEL_PATH = r"D:\ZENDO\model\cnnvit_endometriosis_final.pth"
CLASS_MAP_PATH = r"D:\ZENDO\model\class_to_idx.json"
model, class_to_idx, device = load_model(MODEL_PATH, CLASS_MAP_PATH)
idx_to_class = {v:k for k,v in class_to_idx.items()}

# ------------------ UPLOAD TAB ------------------ #
with tabs[0]:
    if os.path.exists(UPLOAD_BANNER):
        st.image(UPLOAD_BANNER, use_container_width=True)
    st.markdown("<h3>Upload Patient Scan</h3>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a scan (JPG, JPEG, PNG)", type=["jpg","jpeg","png"])
    
    if uploaded:
        image_bytes = uploaded.read()
        try:
            ss.uploaded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
            ss.uploaded_image = None
        
        if ss.uploaded_image:
            st.image(ss.uploaded_image, caption="Uploaded Scan", use_container_width=True)
            run = st.button("🔍 Run Analysis", type="primary")
            
            if run:
                # --- Inference ---
                img_tensor = val_transforms(ss.uploaded_image).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs,1)
                
                ss.diagnosis_label = idx_to_class[pred.item()]
                ss.confidence_pct = conf.item()*100
                ss.probs_dict = {idx_to_class[i]: p.item() for i,p in enumerate(probs[0])}

                # --- Prepare image for drawing ---
                img_np = np.array(ss.uploaded_image)
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                total_area = img_color.shape[0] * img_color.shape[1]
                ss.total_area = total_area

                if ss.diagnosis_label.lower() == "pathology":
                    # --- GradCAM ---
                    target_layer = model.cnn.layer4[-1].conv2
                    gradcam_map = generate_gradcam(model,img_tensor,target_class=None,target_layer=target_layer)
                    gradcam_resized = cv2.resize(gradcam_map,(img_color.shape[1],img_color.shape[0]))
                    gradcam_norm = cv2.normalize(gradcam_resized,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

                    # --- Bounding box on salient area ---
                    _, thresh = cv2.threshold(gradcam_norm,127,255,cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    salient_area = 0
                    if contours:
                        largest = max(contours,key=cv2.contourArea)
                        x,y,w,h = cv2.boundingRect(largest)
                        cv2.rectangle(img_color,(x,y),(x+w,y+h),(0,255,0),2)
                        salient_area = cv2.contourArea(largest)
                        ss.bbox_coords = (x, y, x + w, y + h)

                    ss.salient_area = salient_area
                    ss.bbox_image_bgr = img_color

                    # --- Severity score ---
                    ss.severity_score = estimate_severity(conf.item(), salient_area, total_area)

                    # --- GradCAM overlay ---
                    heatmap = cv2.applyColorMap(gradcam_norm, cv2.COLORMAP_JET)
                    ss.gradcam_image = cv2.addWeighted(img_color,0.6,heatmap,0.4,0)

                    # --- Red-black saliency map ---
                    saliency = draw_saliency_map(img_tensor.clone(), model)
                    saliency_norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    saliency_red_black = np.zeros((saliency_norm.shape[0], saliency_norm.shape[1], 3), dtype=np.uint8)
                    saliency_red_black[...,0] = saliency_norm  # red channel

                    # resize to original image size
                    saliency_resized = cv2.resize(saliency_red_black, (img_color.shape[1], img_color.shape[0]))

                    # combine with original image
                    ss.saliency_image = cv2.addWeighted(img_color, 0.6, saliency_resized, 0.4, 0)

                
                else:
                    ss.severity_score = None
                    ss.bbox_image_bgr = img_color
                    ss.gradcam_image = None
                    ss.saliency_image = None

                st.success("Analysis complete. Check the Results tab.")
    else:
        st.info("Upload a scan to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)

import requests
def generate_llm_prompt(diagnosis, confidence, severity_score, stage, bbox_coords=None, lesion_area=None):
    prompt = f"""
You are a clinical AI assistant analyzing endometriosis ultrasound scans.
Patient scan analysis results:
- Diagnosis: {diagnosis}
- Confidence: {confidence:.1f}%
- Severity score: {severity_score:.2f}
- Stage: {stage}
"""
    if bbox_coords and lesion_area:
        prompt += f"- Lesion bounding box: {bbox_coords}\n- Lesion area (px²): {lesion_area}\n"

    prompt += """
Based on these results, provide detailed next steps and recommendations for the clinician:
- Suggested follow-up investigations
- Possible treatment options or monitoring advice
- Patient guidance if applicable
Provide the response in a clear, actionable, and professional manner.
"""
    return prompt

def get_llm_recommendation(prompt):
    if not GROQ_API_KEY:
        return "GROQ API key not found. Please set it in your .env file."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_completion_tokens=500,
            top_p=1,
            reasoning_effort="medium",
            stream=True
        )

        # Use a single placeholder, not multiple text_area calls
        placeholder = st.empty()
        recommendation = ""

        for i, chunk in enumerate(completion):
            text = chunk.choices[0].delta.content or ""
            recommendation += text
            # Give a unique key using iteration count
            # placeholder.text_area(
            #     "AI Recommendation",
            #     value=recommendation,
            #     height=200,
            #     key=f"llm_recommendation_{i}"
            # )
        return recommendation

    except Exception as e:
        return f"Error fetching recommendation: {e}"


def severity_to_stage(score):
    """
    Convert numeric severity score to stage label.
    """
    if score < 20:
        return "Stage I – Minimal"
    elif score < 40:
        return "Stage II – Mild"
    elif score < 70:
        return "Stage III – Moderate"
    else:
        return "Stage IV – Severe"

# ------------------ RESULTS TAB ------------------ #
with tabs[1]:
    st.markdown("<h3>AI Diagnostic Report</h3>", unsafe_allow_html=True)
    if ss.diagnosis_label is None:
        st.warning("No results yet. Upload a scan and run analysis in the Upload tab.")
    else:
        c1, c2, c3 = st.columns([1.2, 1, 1])
        c1.metric("Diagnosis", ss.diagnosis_label)
        if ss.confidence_pct is not None:
            c2.metric("Confidence", f"{ss.confidence_pct:.1f}%")
        if ss.severity_score is not None:
            c3.metric("Severity Score", f"{ss.severity_score:.2f}")
        if ss.severity_score:
            stage = severity_to_stage(ss.severity_score)
            if ss.severity_score:
                stage = severity_to_stage(ss.severity_score)
                # Prepare bbox info
                bbox = getattr(ss, "bbox_coords", None)
                lesion_area = getattr(ss, "salient_area", None)
                
                dynamic_prompt = generate_llm_prompt(
                    ss.diagnosis_label,
                    ss.confidence_pct,
                    ss.severity_score,
                    stage,
                    bbox_coords=bbox,
                    lesion_area=lesion_area
                )
                llm_text = get_llm_recommendation(dynamic_prompt)
                st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
                st.markdown(f"**Recommendations — {stage}**")
                st.info(llm_text)
                
        # --- Severity Calculation Breakdown ---
        if ss.diagnosis_label.lower() == "pathology" and ss.salient_area is not None and ss.total_area is not None:
            ss.severity_score = estimate_severity(
            ss.confidence_pct / 100 if ss.confidence_pct else 0,
            ss.salient_area,
            ss.total_area
        )
            ss.salient_area = salient_area
            ss.total_area = total_area
            conf_val = conf.item()
            salient_ratio = salient_area / total_area if total_area else 0

            # --- Console output only ---
            print("\n=== Severity Calculation Breakdown ===")
            print(f"Confidence: {conf_val:.4f} ({ss.confidence_pct:.2f}%)")
            print(f"Salient Area: {int(ss.salient_area)} px")
            print(f"Total Area: {int(ss.total_area)} px")
            print(f"Salient Ratio: {salient_ratio:.6f}")
            print("Severity Score = Confidence × Salient_Ratio × 100")
            print(f"              = {conf_val:.4f} × {salient_ratio:.6f} × 100")
            print(f"              = {ss.severity_score:.2f}\n")

        st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

        if ss.probs_dict:
            st.markdown("**Class Probabilities**")
            for cls, p in ss.probs_dict.items():
                pct = int(round(p * 100))
                st.write(f"{cls}: **{pct}%**")
                st.progress(pct)

        st.markdown("<hr class='hr'/>", unsafe_allow_html=True)
        st.markdown("**Explainability Visuals**")
        colA, colB = st.columns(2)
        with colA:
            if ss.bbox_image_bgr is not None:
                st.image(ss.bbox_image_bgr, channels="BGR", caption="Bounding Box Overlay", use_container_width=True)
                bbox = getattr(ss, "bbox_coords", None)
                  # (x_min, y_min, x_max, y_max)
                 
                if bbox:
                    pixel_to_cm = None  # e.g., 1/50 if you know the scale (1 cm = 50 px)
                    width, height, area = calculate_lesion_dimensions(bbox, pixel_to_cm)

                    if pixel_to_cm:
                        st.markdown(f"**Estimated Lesion Dimensions:** {width:.2f} × {height:.2f} cm")
                        st.markdown(f"**Estimated Area:** {area:.2f} cm²")
                    else:
                        st.markdown(f"**Estimated Lesion Dimensions (unscaled):** {width:.0f} × {height:.0f} px")
                        st.markdown(f"**Estimated Area (unscaled):** {area:.0f} px²")
                        st.caption("Note: Measurements shown in pixels (unscaled). Physical calibration not available for this scan.")
                        def estimate_severity(area_px):
                            """Estimate lesion severity based on unscaled pixel area."""
                            if area_px < 5000:
                                return "Mild (Stage I–II)"
                            elif 5000 <= area_px < 30000:
                                return "Moderate (Stage II–III)"
                            else:
                                return "Severe (Stage III–IV)"

                        severity = estimate_severity(area)
                        # st.markdown(f"**Estimated Severity:** {severity}")
                        st.caption("Note: Approximation based on lesion size. True stage depends on location, depth, and adhesions.")
                                        # If you know scale, e.g., 1 cm = 50 pixels => pixel_to_cm = 1/50
                        # --- Lesion location estimation (heuristic) ---
                        def estimate_location(bbox, image_shape):
                            x_min, y_min, x_max, y_max = bbox
                            h, w = image_shape[:2]
                            y_center = (y_min + y_max) / 2

                            if y_center < h * 0.33:
                                return "Peritoneum (upper field)"
                            elif y_center < h * 0.66:
                                return "Ovary / Fallopian Region"
                            else:
                                return "Rectovaginal / Pouch of Douglas area"

                        location = estimate_location(bbox, ss.bbox_image_bgr.shape)
                        st.markdown(f"**Estimated Lesion Location:** {location}")
                        st.caption("Note: Estimated from lesion position. Actual location should be clinically verified.")

            elif ss.uploaded_image is not None:
                st.image(ss.uploaded_image, caption="Uploaded Scan", use_container_width=True)
        with colB:
            if ss.gradcam_image is not None:
                st.image(ss.gradcam_image, channels="BGR", caption="Grad-CAM Overlay", use_container_width=True)
            elif ss.saliency_image is not None:
                st.image(ss.saliency_image, caption="Saliency Map", use_container_width=True)
            else:
                st.info("No explainability visuals available.")
 
