# =========================================
# Gihub request from drive
import requests
import gdown
import zipfile
import os
# =========================================
# Library 
# =========================================
import streamlit as st
# =========================================
# Pytorch 
import torch
import torchvision.transforms as transforms
# =========================================
# Sklearn 
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import set_config
import pandas as pd
import sklearn.utils
# =========================================
# Image
from PIL import Image
# =========================================
# =========================================

import joblib
import base64
###########################################
###########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================
# Page Information 
st.set_page_config(
    page_title="Medical AI Diagnosis HUB",
    page_icon="🏥",
    layout="wide",  
)
# =========================================
# Hide Streamlit elements
hide_streamlit_style = """ <style> 
                # MainMenu {visibility: hidden;} 
                header {visibility: hidden;} 
                footer {visibility: hidden;} 
                .block-container { padding-top: 0rem; padding-bottom: 0.5 rem; }
                [data-testid="stSidebar"] 
                {display: none;} 
               h1, h2, h3, h4, h5, h6, label, p {
                    color: white !important;
                }
                
                /* Tabs text */
                [data-baseweb="tab"] p {
                    color: white !important;
                    font-size: 15px !important;
                    font-weight: bold !important;
                }
                

                </style> """

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
###########################################
# -----------------------------
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


        </style>
        """,
        unsafe_allow_html=True
    )
image_path = "images/img4.png"

set_bg_from_url(image_path)  
###########################################
# Model 1
stroke_model = joblib.load("strok_svm.pkl")
###########################################
###########################################
# Model 2
def binary_transform(X):
    return X.replace({
        'Y': 1, 
        'N': 0, 
        'F': 1, 
        'M': 0, 
        'Placebo': 0,
        'D-penicillamine': 1
    })
binary_transformer = FunctionTransformer(binary_transform)
def transform_input(df):
    return df.replace({
        'Y': 1, 'N': 0,
        'F': 1, 'M': 0,
        'Placebo': 0, 'D-penicillamine': 1
    })

###########################################
liver_model = joblib.load("xgb_pipeline_with_le.pkl")
pipeline = liver_model["pipeline"]
le = liver_model["label_encoder"]

###########################################
###########################################
# Model 3

url = "https://github.com/Mohamed66Hemdan/Medical-AI-Diagnosis-Application/releases/download/v1.0/mri_model.pth"
output = "mri_model.pth"
response = requests.get(url)
with open(output, "wb") as f:
    f.write(response.content)
torch.serialization.add_safe_globals([
    torch.nn.Sequential,
    torch.nn.Conv2d,
    torch.nn.ReLU,
    torch.nn.MaxPool2d,
    torch.nn.Flatten,
    torch.nn.Linear,
    torch.nn.Dropout
])
brain_model = torch.load(output, map_location=device)
brain_model.eval()

###########################################
###########################################

stroke_classes = ["Low Risk", "High Risk"]
liver_classes  = ["Stage 1", "Stage 2", "Stage 3"]
brain_classes  = ["Healthy", "Tumor"]
# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏥 Medical AI Diagnosis HUB")
tab1, tab2, tab3 = st.tabs(["💓 Stroke Risk", "🧬 Liver Cirrhosis", "🧠 Brain Tumor"])

# -----------------------------
# Tab 1 - Stroke Risk (Structured Data)
# -----------------------------
with tab1:
    st.subheader("💓 Stroke Risk Prediction ")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 1)  # الافتراضي 1
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)  # Default Male
        chest_pain = st.selectbox("Chest Pain", [0, 1], index=0)
        high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], index=0)
        irregular_heartbeat = st.selectbox("Irregular Heartbeat", [0, 1], index=0)
        shortness_of_breath = st.selectbox("Shortness of Breath", [0, 1], index=0)

    with col2:
        fatigue_weakness = st.selectbox("Fatigue / Weakness", [0, 1], index=0)
        dizziness = st.selectbox("Dizziness", [0, 1], index=0)
        swelling_edema = st.selectbox("Swelling / Edema", [0, 1], index=0)
        neck_jaw_pain = st.selectbox("Neck / Jaw Pain", [0, 1], index=0)
        excessive_sweating = st.selectbox("Excessive Sweating", [0, 1], index=0)
        persistent_cough = st.selectbox("Persistent Cough", [0, 1], index=0)
    
    with col3:
        nausea_vomiting = st.selectbox("Nausea / Vomiting", [0, 1], index=0)
        chest_discomfort = st.selectbox("Chest Discomfort", [0, 1], index=0)
        cold_hands_feet = st.selectbox("Cold Hands / Feet", [0, 1], index=0)
        snoring_sleep_apnea = st.selectbox("Snoring / Sleep Apnea", [0, 1], index=0)
        anxiety_doom = st.selectbox("Anxiety / Doom", [0, 1], index=0)


    if st.button("Predict Stroke Risk", key="single"):
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "chest_pain": chest_pain,
            "high_blood_pressure": high_blood_pressure,
            "irregular_heartbeat": irregular_heartbeat,
            "shortness_of_breath": shortness_of_breath,
            "fatigue_weakness": fatigue_weakness,
            "dizziness": dizziness,
            "swelling_edema": swelling_edema,
            "neck_jaw_pain": neck_jaw_pain,
            "excessive_sweating": excessive_sweating,
            "persistent_cough": persistent_cough,
            "nausea_vomiting": nausea_vomiting,
            "chest_discomfort": chest_discomfort,
            "cold_hands_feet": cold_hands_feet,
            "snoring_sleep_apnea": snoring_sleep_apnea,
            "anxiety_doom": anxiety_doom
        }])

        pred = stroke_model.predict(input_df)[0]
        proba = stroke_model.predict_proba(input_df)[0][1]

        st.success(f"✅ Predicted Risk: {'At Risk' if pred==1 else 'Not At Risk'}")
        st.info(f"📊 Probability of Stroke Risk: {proba:.2%}")


# -----------------------------
# Tab 2 - Liver Cirrhosis (Structured Data)
# -----------------------------
with tab2:
    st.subheader("🧬 Liver Cirrhosis Stage Classification")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (days)", 0, 40000, 0)  # default 0
        sex = st.selectbox("Sex", ["M", "F"], index=0)  # default "M"
        ascites = st.selectbox("Ascites", ["Y", "N"], index=0)  # default "Y"
        hepatomegaly = st.selectbox("Hepatomegaly", ["Y", "N"], index=0)  # default "Y"
        spiders = st.selectbox("Spiders", ["Y", "N"], index=0)  # default "Y"
        n_days = st.number_input("N_Days", 0, 5000, 0)  # default 0

    with col2:
        drug = st.selectbox("Drug", ["D-penicillamine", "Placebo"], index=0)  # default first
        status = st.selectbox("Status", ["C", "CL", "D"], index=0)  # default "C"
        edema = st.selectbox("Edema", ["Y", "N"], index=0)  # default "Y"
        albumin = st.number_input("Albumin", 0.0, 6.0, 0.0)  # default 0.0
        bilirubin = st.number_input("Bilirubin", 0.0, 50.0, 0.0)  # default 0.0
        tryglicerides = st.number_input("Triglycerides", 0, 1000, 0)  # default 0
    
    with col3:
        platelets = st.number_input("Platelets", 0, 1000, 0)  # default 0
        prothrombin = st.number_input("Prothrombin", 0.0, 30.0, 0.0)  # default 0.0
        cholesterol = st.number_input("Cholesterol", 0.0, 1000.0, 0.0)  # default 0.0
        copper = st.number_input("Copper", 0.0, 500.0, 0.0)  # default 0.0
        alk_phos = st.number_input("Alk Phos", 0.0, 2000.0, 0.0)  # default 0.0
        sgot = st.number_input("SGOT", 0, 500, 0)  # default 0

    if st.button("Predict Cirrhosis Stage"):
        new_patient = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "Ascites": ascites,
            "Hepatomegaly": hepatomegaly,
            "Spiders": spiders,
            "Drug": drug,
            "Status": status,
            "Edema": edema,
            "Albumin": albumin,
            "Bilirubin": bilirubin,
            "Platelets": platelets,
            "Prothrombin": prothrombin,
            "Cholesterol": cholesterol,
            "Copper": copper,
            "Alk_Phos": alk_phos,
            "SGOT": sgot,
            "Tryglicerides": tryglicerides,
            "N_Days": n_days
        }])


        # Prediction
        pred_enc = pipeline.predict(new_patient)
        pred_label = le.inverse_transform(pred_enc)

        st.success(f"✅ Predicted Stage: **{pred_label[0]}**")
# -----------------------------
# Tab 3 - Brain Tumor (Image)
# -----------------------------
with tab3:
    st.subheader("🧠 Brain Tumor Classification")
    
    uploaded_file = st.file_uploader(
        "📤 Upload MRI or CT Image",
        type=["png", "jpg", "jpeg"],
        help="Upload an MRI or CT Image image (PNG, JPG, JPEG). Max size 200MB."
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
    
        # Style Image 
        st.markdown(
            f"""
            <div style="text-align:center; margin-top:-10px;">
                <img src="data:image/png;base64,{base64.b64encode(uploaded_file.getvalue()).decode()}" 
                     alt="Uploaded Image" 
                     style="border-radius:12px; 
                            box-shadow:0px 4px 15px rgba(0,0,0,0.2); 
                            width:400px;"/>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Prepare Model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_img = transform(img).unsqueeze(0).to(device)

        # Button Predication
        st.markdown("<div style='margin-top:25px; text-align:center;'>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Tumor")
        st.markdown("</div>", unsafe_allow_html=True)

        if predict_btn:
            with torch.inference_mode():
                logits = brain_model(input_img)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, 1)
                pred = brain_classes[idx.item()]
                confidence = conf.item()

        
            if pred.lower() == "tumor":
                bg_color = "#e74c3c"
                icon = "⚠️"
            else:
                bg_color = "#2ecc71"
                icon = "✅"

            # Output Massage
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; 
                            background-color:{bg_color}; color:white; 
                            text-align:center; font-size:18px; font-weight:bold; 
                            margin-top:15px; width:600px; margin-left:auto; margin-right:auto;">
                    {icon}  {pred} <br>
                    🔎 Confidence: {confidence*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
# 














































