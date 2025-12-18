# about_page.py -- Streamlit About Page for EndoInsights (final polished version)
# Run: streamlit run about_page.py

import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os
import urllib.parse
import base64

st.set_page_config(
    page_title="EndoInsights — Endometriosis Detection",
    page_icon="🩺",
    layout="wide"
)

# --- Navbar + Global Styles ---
st.markdown(
    """
    <style>
     /* Force top padding for Streamlit content (space for navbar) */
    .main, .block-container, .stAppViewContainer, .stApp {
        padding-top: 100px !important;
        margin-top: 20 !important;
    }
    
    /* --- NAVBAR --- */
    .navbar {
        position: fixed;
        top: 00;
        left: 0;
        width: 100%;
        # z-index: 9999;
        background: black;
        height: 100px;
        # padding: 10px 60px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }

    .navbar-title {
        margin: 12px !important;
        font-size: 54px;
        font-weight: 1900;
        color: white !important; 
        letter-spacing: 8px;
    }

    .nav-links a {
        color: white;
        text-decoration: none;
        font-weight: 500;
        margin: 0 16px;
        font-size: 20px;
        transition: 0.2s ease;
    }

    .nav-links a:hover {
        color: #38bdf8;
        text-decoration: underline;
    }
    

    /* --- GLOBAL TEXT --- */
    html, body, [class*="css"] {
        color: black !important;
    }
    .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, span, div {
        color: black !important;
    }

    /* --- CONTENT SPACING --- */
    .page-content {
        # margin-top: 0px;
        # padding: 0px 40px;
    }

    /* --- IMAGES --- */
    .circle-img img {
        width: 180px !important;
        height: 180px !important;
        border-radius: 50%;
        object-fit: cover;
        object-position: center;
        border: 3px solid #38bdf8;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
        
        transition: transform 0.3s ease;
    }
    .circle-img img:hover {transform: scale(1.05);}

    .contact-btn {
        background: pink;
        color: black;
        border: none;
        border-radius: 8px;
        padding: 6px 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .contact-btn:hover {opacity: 0.9; transform: translateY(-2px);}

    .team-card {
         background: linear-gradient(90deg, #0a192f, #123a56);
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
        transition: all 0.3s ease;
    }
    .team-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    }
    </style>

    <div class="navbar">
        <div class="navbar-title" >EndoInsights</div>
    </div>

    <div class="page-content">
    """,
    unsafe_allow_html=True,
)

# --- Utility: load image (URL or local) ---
@st.cache_data(show_spinner=False)
def load_image_any(source, timeout=8):
    if not source:
        return None
    if isinstance(source, str) and source.lower().startswith(("http://", "https://")):
        try:
            resp = requests.get(source, timeout=timeout)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None
    if isinstance(source, str) and os.path.exists(os.path.expanduser(source)):
        try:
            return Image.open(os.path.expanduser(source)).convert("RGB")
        except Exception:
            return None
    return None

# --- Project Overview ---
st.markdown("## Project Overview")
with st.container():
    col1, col2, col3 = st.columns([1.8, 1, 1])
    with col1:
        st.markdown("#### What this project does")
        st.markdown(
            """
            - Detects and highlights suspicious lesions in ultrasound/laparoscopy images.  
            - Combines CNN + Vision Transformer for robust feature extraction.  
            - Helps clinicians get quick, explainable second opinions.
            """
        )
    with col2:
        st.metric(label="Model Type", value="Hybrid CNN + ViT")
        st.metric(label="Primary Input", value="Glenda Dataset")
    with col3:
        st.metric(label="Status", value="Prototype")
        st.metric(label="Last update", value="Nov 06, 2025")

st.write("---")

# --- Visual Gallery ---
st.markdown("##  Visual Gallery ")
image_urls = [
    "frontend/static/9.jpg", "frontend/static/10.jpg", "frontend/static/11.jpg", "frontend/static/12.jpg",
    "frontend/static/5.jpg", "frontend/static/6.jpg", "frontend/static/4.jpg", "frontend/static/8.jpg",
]
placeholder = Image.new("RGB", (800, 450), color=(220, 220, 220))

# Display images 4 per row, 2 rows total
for row in range(2):
    cols = st.columns(4)
    for i in range(4):
        idx = row * 4 + i
        if idx < len(image_urls):
            img = load_image_any(image_urls[idx])
            with cols[i]:
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.image(placeholder, caption=f"Image {idx+1}", use_container_width=True)

st.write("---")

# --- Project Highlights ---
st.markdown("##  Project Highlights & Timeline")
left, right = st.columns([2, 1])
with left:
    st.markdown("**Design** — Data collection, annotation, model architecture exploration (CNN → ViT hybrid)")
    st.markdown("**Development** — Preprocessing pipelines, training scripts, evaluation dashboards.")
    st.markdown("**Evaluation** — Cross-val, holdout test, clinician-in-the-loop qualitative checks.")
    st.markdown("**Next** — UI polish, prospective clinical validation, regulatory planning.")
with right:
    st.info("Want the full write-up? Click below to download a demo PDF.")
    pdf_bytes = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
    st.download_button("📄 Download project summary (demo)", data=pdf_bytes, file_name="SYNOPSIS.pdf", mime="SYNOPSIS/pdf")

st.write("---")

# --- Team Section ---
st.markdown("## Team & Contributors")
team = [
    {"name": "Amrutha G Y", "role": "Backend Developer", "email": "amruthagy16@gmail.com", "img": "frontend/static/Amru.jpeg"},
    {"name": "Shallen Crissle Sequeira", "role": "Backend Developer", "email": "shallensequeira1204@gmail.com", "img": "frontend/static/ShallenSequeira.jpeg"},
    {"name": "Sparsha S Shriyan", "role": "Backend Developer", "email": "sparshashriyan@gmail.com", "img": "frontend/static/Sparsha.jpeg"},
    {"name": "Vaishnavi Poojari", "role": "UI/UX Designer", "email": "vaish3229@gmail.com", "img": "frontend/static/Vaish.jpeg"},
]
cols = st.columns(4)
for idx, member in enumerate(team):
    with cols[idx]:
        img = load_image_any(member["img"]) or Image.new("RGB", (160, 160), color=(25, 30, 35))
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        st.markdown(f"""
            <div class='team-card'>
                <div class='circle-img'>
                    <img src='data:image/png;base64,{img_b64}' alt='{member["name"]}'/>
                </div>
                <strong style='font-size:0.9rem; color:white;'>{member["name"]}</strong><br>
                <em style='font-size:0.9rem; color:white;'>{member["role"]}</em><br>
                <p style='font-size:0.9rem; color:white;'>📧 <a href='mailto:{member["email"]}' style='color:white;'>{member["email"]}</a></p>
                <a href='mailto:{member["email"]}?subject=Query%20about%20EndoInsights'>
                    <button class='contact-btn'>📩 Contact</button>
                </a>
            </div>
        """, unsafe_allow_html=True)

st.write("---")

# --- Contact Form (Black Background + White Input Boxes, Correctly Styled) ---
st.markdown("""
    <style>
    /* --- Contact Section Wrapper --- */
    .contact-wrapper {
        background: black;
        color: white;
        border-radius: 20px;
        padding: 50px 60px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.6);
        margin: 50px auto;
        width: 90%;
    }

    .contact-wrapper h2 {
        color: #38bdf8;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }

    .contact-wrapper p {
        color: #cccccc;
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
    }

    /* --- Input Boxes --- */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #666 !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus,
    .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56,189,248,0.3);
    }

    /* --- Button --- */
    .stButton button {
        background: linear-gradient(90deg, #38bdf8, #0ea5e9);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 22px;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(56,189,248,0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Now we use st.container() so everything inside is visually enclosed
with st.container():
    # st.markdown("<div class='contact-wrapper'>", unsafe_allow_html=True)
    st.markdown("##  Contact the Team")
    st.markdown("<p>We’d love to hear your thoughts, suggestions, or questions about EndoInsights!</p>", unsafe_allow_html=True)

    with st.form("contact_form style=' color:black;'", clear_on_submit=True):
        nm = st.text_input("Your Name", placeholder="Enter your full name")
        em = st.text_input("Your Email", placeholder="example@email.com")
        msg = st.text_area("Message", placeholder="Write your message here...")
        target = st.selectbox("Send to", [m["name"] for m in team], index=1)
        submitted = st.form_submit_button("Send Message ")
        if submitted:
            recipient = next((m["email"] for m in team if m["name"] == target), team[1]["email"])
            subject = urllib.parse.quote(f"EndoInsights Query from {nm or 'Anonymous'}")
            body = urllib.parse.quote(f"From: {nm or 'Anonymous'}\nEmail: {em or 'Not provided'}\n\nMessage:\n{msg or '(no message)'}")
            mailto_link = f"mailto:{recipient}?subject={subject}&body={body}"
            st.success("✨ Your message is ready to send via your email app.")
            st.markdown(f"[Click here if it didn’t open automatically]({mailto_link})")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")



# --- Explainability ---
st.markdown("## How EndoInsights Works (high-level)")
st.markdown(
    """
    1. **Preprocessing** — standardize image sizes, remove identifiable metadata, normalize intensity.  
    2. **Chunking** — break large images into tiles to preserve resolution where needed.  
    3. **Embedding** — convolutional layers extract local texture; Transformer layers model global context.  
    4. **Explainability** — Grad-CAM style heatmaps + pixel-level lesion overlays for clinician review.
    """
)

st.markdown("<div style='color:black;'>Made by the team — prototype for research & collaboration.</div>", unsafe_allow_html=True)
