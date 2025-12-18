import streamlit as st
from PIL import Image
st.set_page_config(
    page_title="EndoInsights — Endometriosis Detection",
    page_icon="🩺",
    layout="wide"
)

# ---------------- NAVBAR + GLOBAL STYLES ----------------
st.markdown("""
    <style>
    /* Professional and Vibrant Color Palette Definition */
    :root {
        --color-primary: #1A2C3E; /* Dark Navy for Text/Primary */
        --color-accent: #00A7A5; /* Vibrant Teal/Cyan */
        --color-background: #FFFFFF; /* Clean White */
        --color-card-bg: #F8F9FA; /* Lightest Grey for professional cards */
        --color-neon-blue: #38bdf8; /* Original Neon Blue for Navbar hover */
        
        /* Medical Ombre Colors for Feature Cards */
        --color-med-soft-green: #A5D6A7;
        --color-med-soft-blue: #81D4FA;
        --color-med-dark-teal: #00796B;
    }

    /* Add smooth animations and Inter font globally */
    * {
        transition: 0.3s ease;
        # font-family: 'Inter', sans-serif;
    }

    /* Page top spacing and background */
    .main, .block-container, .stAppViewContainer, .stApp {
        padding-top: 100px !important;
        background-color: var(--color-background);
    }

    /* NAVBAR (SYNCHRONIZED WITH ABOUT PAGE) */
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
    /* END OF NAV BAR STYLES */

    /* Set default text color to professional navy, overriding Streamlit defaults */
    html, body, [class*="css"] * {
        color: var(--color-primary) !important;
    }

    /* Heading Style for main content sections */
    .professional-heading {
        color: var(--color-primary) !important;
        font-weight: 800;
        letter-spacing: 1px;
        font-size: 32px;
        margin-bottom: 20px;
    }

    /* Feature Card styling with Medical Ombre Hover Effect */
    .feature-card {
        background: var(--color-card-bg); /* Light base */
        border-radius: 12px;
        padding: 24px;
        /* Start with a soft, clean border */
        border: 2px solid #E0E0E0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border 0.3s ease;
        min-height: 180px; /* Ensure uniform height */
    }

    .feature-card:hover {
        transform: translateY(-5px);
        /* Ombre effect using diffused, multi-color box shadow */
        border-color: var(--color-med-soft-blue); /* Border color shifts */
        box-shadow: 
            0 0 5px var(--color-med-soft-green), /* Soft Green Layer */
            0 0 15px var(--color-med-soft-blue), /* Soft Blue Layer */
            0 8px 25px rgba(0, 167, 165, 0.2); /* Base Teal Accent */
    }

    /* Professional section dividers */
    .divider {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--color-accent), transparent);
        margin: 40px 0;
    }

    </style>

    <div class="navbar">
        <div class="navbar-title">EndoInsights</div>
        <div class="nav-links">
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION (Professional/Vibrant) ----------------
st.markdown(f"""
    <div style="
        width: 100%;
        padding: 60px 40px;
        border-radius: 16px;
        background: linear-gradient(135deg, #1A2C3E, #00A7A5); /* Dark Navy to Vibrant Teal */
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(0, 167, 165, 0.4);
    ">
        <h1 style="font-size: 48px; margin-bottom: 10px; color: white; font-weight: 900;">
            Precision Diagnostics for Endometriosis
        </h1>
        <p style="font-size: 20px; color:#e0e0e0;">
            Leveraging advanced AI vision to provide clear, actionable insights from medical imagery, accelerating diagnosis and improving patient care.
        </p>
    </div>
""", unsafe_allow_html=True)

st.write("")


# ---------------- FEATURE CARDS SECTION (Ombre Medical Style) ----------------
st.markdown("""<div class='divider'></div>""", unsafe_allow_html=True)
st.markdown("<h2 class='professional-heading'>Key Differentiating Features</h2>", unsafe_allow_html=True)

cols = st.columns(4)
features = [
    ("Clinical Accuracy", "AI models trained on diverse, validated datasets for high diagnostic precision."),
    ("Clarity & Localization", "Segmentation maps highlight affected regions with high-fidelity detail."),
    ("Seamless Integration", "Designed for rapid deployment into existing clinical workflows."),
    ("Research-Driven", "Built on modern deep learning architectures and published research standards."),
]

for col, (title, desc) in zip(cols, features):
    with col:
        st.markdown(f"""
            <div class="feature-card">
               <h4 style="color:var(--color-med-dark-teal); margin-bottom: 8px; font-weight: 700;">{title}</h4>
               <p style="font-size:16px; color:var(--color-primary) !important;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)


st.markdown("""<div class='divider'></div>""", unsafe_allow_html=True)


# ---------------- MODEL EXPLANATION (Professional) ----------------
st.markdown("<h2 class='professional-heading'>The Science Behind the Vision</h2>", unsafe_allow_html=True)

st.markdown("""
<p style="font-size: 18px; line-height: 1.6; margin-bottom: 20px;">
    EndoInsights utilizes a sophisticated <b>Convolutional Neural Network (CNN)</b> architecture, purpose-built for surgical and microscopic image analysis. This model is engineered not just to classify, but to precisely localize and segment suspected endometriotic lesions. Our focus is on transparency, providing clinicians with both the definitive classification and a visual heatmap of the AI's area of attention.
</p>
""", unsafe_allow_html=True)

try:
    # This block handles the image display or the placeholder if the image file is missing.
    arch = Image.open("frontend/static/AI.png")
    st.image(arch, caption="AI Architecture (Concept Art)", use_container_width=True)
except:
    # Placeholder for AI Architecture Diagram
    st.markdown(f"""
    <div style="
        width: 100%;
        height: 300px;
        background-color: #E0E0E0;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px dashed #B0B0B0;
        margin-top: 20px;
        margin-bottom: 20px;
    ">
        <p style="color: #6A6A6A !important; font-size: 20px;">[Placeholder for AI Architecture Diagram]</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""<div class='divider'></div>""", unsafe_allow_html=True)


# ---------------- FOOTER (Professional) ----------------
st.markdown(f"""
    <div style="text-align:center; color:var(--color-primary); padding:20px; font-size: 14px;">
        EndoInsights is a research platform. © 2025. | <span style="color: var(--color-accent); font-weight: 600;">Accelerating medical insight through technology.</span>
    </div>
""", unsafe_allow_html=True)