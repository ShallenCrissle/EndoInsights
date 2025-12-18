import streamlit as st
import json
from datetime import datetime

st.set_page_config(
    page_title="EndoInsights — Feedback",
    page_icon="🩺",
    layout="wide"
)

# ---------------- NAVBAR + GLOBAL STYLES ----------------
st.markdown("""
    <style>
    :root {
        --color-primary: #1A2C3E;
        --color-accent: #00A7A5;
        --color-background: #FFFFFF;
        --color-card-bg: #F8F9FA;
        --color-neon-blue: #38bdf8;
    }
    * { transition: 0.3s ease; }
    .main, .block-container, .stAppViewContainer, .stApp { padding-top: 100px !important; background-color: var(--color-background); }
    .navbar { position: fixed; top: 0; left: 0; width: 100%; background: black; height: 100px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
    .navbar-title { margin: 12px !important; font-size: 54px; font-weight: 1900; color: white !important; letter-spacing: 8px; }
    .nav-links a { color: white; text-decoration: none; font-weight: 500; margin: 0 16px; font-size: 20px; transition: 0.2s ease; }
    .nav-links a:hover { color: #38bdf8; text-decoration: underline; }
    html, body, [class*="css"] * { color: var(--color-primary) !important; }
    .feature-card { background: var(--color-card-bg); border-radius: 12px; padding: 24px; border: 2px solid #E0E0E0; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05); transition: transform 0.3s ease, box-shadow 0.3s ease, border 0.3s ease; min-height: 180px; }
    .feature-card:hover { transform: translateY(-5px); border-color: var(--color-med-soft-blue); box-shadow: 0 0 5px #A5D6A7, 0 0 15px #81D4FA, 0 8px 25px rgba(0, 167, 165, 0.2); }
    </style>

    <div class="navbar">
        <div class="navbar-title">EndoInsights</div>
        <div class="nav-links">
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown(f"""
    <div style="
        width: 100%;
        padding: 60px 40px;
        border-radius: 16px;
        background: linear-gradient(135deg, #00A7A5, #1A2C3E); 
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(0, 167, 165, 0.4);
    ">
        <h1 style="font-size: 48px; margin-bottom: 10px; font-weight: 900;">
            We Hear You!
        </h1>
        <p style="font-size: 20px; color:#e0e0e0;">
            Your feedback helps us improve EndoInsights and the AI-powered diagnosis experience.
        </p>
    </div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- FEEDBACK FORM ----------------
st.markdown("### Share Your Feedback")
st.write("Please answer the following questions to help us improve the app:")

# Questions
results_type = st.selectbox(
    "What type of results did you receive?",
    ["Pathology", "Non-Pathology"]
)

effectiveness = st.slider(
    "How effective were the results?", 0, 10, 5
)

ai_diagnosis_usefulness = st.radio(
    "Was the AI diagnosis helpful?", 
    ["Yes, very helpful", "Somewhat helpful", "Not helpful"]
)

app_experience = st.selectbox(
    "Overall, how did you find the app experience?",
    ["Excellent", "Good", "Average", "Poor"]
)

additional_comments = st.text_area(
    "Any other comments or suggestions?"
)

# Submit button
if st.button("Submit Feedback"):
    feedback_data = {
        "timestamp": str(datetime.now()),
        "results_type": results_type,
        "effectiveness": effectiveness,
        "ai_diagnosis_usefulness": ai_diagnosis_usefulness,
        "app_experience": app_experience,
        "additional_comments": additional_comments
    }

    # Save to file
    try:
        with open("feedback.json", "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
        st.success("Thank you! Your feedback has been submitted.")
    except Exception as e:
        st.error(f"Oops! Could not save feedback: {e}")
