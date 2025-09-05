###########################################
# =========================================
# Library 
# =========================================
import streamlit as st
import base64
# =========================================
# Page Information 
st.set_page_config(
    page_title="Medical AI Diagnosis HUB",
    page_icon="🏥",
    layout="wide",  
    initial_sidebar_state="collapsed", 
)
# =========================================
# 🔒 Hide Streamlit elements
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
# =========================================
# Image Url
def set_bg_from_url(image_url):
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .main-button {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 10vh;
        }}
        .stButton > button{{
            font-size: 30px ;
            padding: 0.75em 2em;
            border-radius: 12px;
            background-color: #000;
            color: #FFF;
            border: none;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
            cursor: pointer;
            transition: 0.3s;
            margin-left: 40%;

        }}
        .stButton > button:hover {{
            background-color: #f0f0f0;
            color: #111;
            
        }}
    
        </style>
        """,
        unsafe_allow_html=True
    )
image_path = "images/img4.png"
set_bg_from_url(image_path)  

# st.title("Medical AI HUB")

st.markdown("<h1 style='text-align: center; color: white; font-size: 70px;'>Medical AI Diagnosis HUB</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; color: white; font-size: 30px;'>Stroke Risk Prediction <br> Liver Cirrhosis Stage Classification <br> Brain Tumor Classification</h3>", unsafe_allow_html=True)

st.markdown("<div class='main-button';  style='text-align: center;' >", unsafe_allow_html=True)
if st.button("Start Diagnosis 🧬"):
    st.switch_page("pages/Medical.py")
st.markdown("</div>", unsafe_allow_html=True)
