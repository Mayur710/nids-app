import streamlit as st 
import joblib
import pandas as pd 
import numpy as np
from genai import generate_explanation, ask_security_chatbot
import gdown
import os

def download_models():
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('preprocessed_data'):
        os.makedirs('preprocessed_data')
    
    # Only download if files don't exist
    files = {
        'models/rf.pkl': '19fC9xLkp239JKbds1qXrB0mtAgNn6ebN',
        'models/dt.pkl': '1ukDtHq89wykJGetld-_Cjzd6T1elbCBA',
        'models/xgb.pkl': '1I4s1VB2vXdoZqISBr7_K7Zrc6VjD4pri',
        'models/voting.pkl': '1Sl-EqB_oRdI5vZqmey8PCWhHdTwaAVxv',
        'preprocessed_data/scaler.pkl': '1W1cpn_1OJ8hPR8pPr9FqnKvbSakPWB3u',
        'preprocessed_data/pca.pkl': '1mSMAw41uUeee7nBqXSK3pk2Py93Hra-C',
        'preprocessed_data/label_encoder.pkl': '1adfN8Om7MNZi-TbVwaaor_Az7rneML7J',
        'preprocessed_data/feature_columns.pkl': '1Is09beVasUAhS7IgQFCQEEcFqqC9hT8-',
    }
    
    for path, file_id in files.items():
        if not os.path.exists(path):
            gdown.download(
                f'https://drive.google.com/uc?id={file_id}',
                path,
                quiet=False
            )

download_models()
@st.cache_resource
def load_models():
    #here we have to load pkl and npy files for ML models 
    #load rf.pkl
    feature_columns = joblib.load("preprocessed_data/feature_columns.pkl")
    ml_model = joblib.load("models/rf.pkl")
    scaler = joblib.load("preprocessed_data/scaler.pkl")
    pca = joblib.load("preprocessed_data/pca.pkl")
    label_encoder = joblib.load("preprocessed_data/label_encoder.pkl")
    return ml_model, scaler, pca, label_encoder, feature_columns

ml_model, scaler, pca, label_encoder, feature_columns = load_models()
st.title("Intelligent Network Intrusion Detection System")
st.write("Upload a network traffic CSV file to detect attacks")

#section 1 where we have to upload csv and run ML models    
st.header("Upload network traffic data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if(uploaded_file is not None):
    
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()
    st.write("Data preview:")
    st.dataframe(data.head())
    #preprocess the data using the scaler and pca we loaded
    if "Label" in data.columns:
        features = data.drop(columns=["Label"])
    else:
        features = data.copy()
    features = features[feature_columns]
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()
    scaled_features = scaler.transform(features)
    pca_features = pca.transform(scaled_features)
    #predict using the ml model we loaded
    predictions = ml_model.predict(pca_features)
    
    predicted_labels = label_encoder.inverse_transform(predictions)
    features["predicted_label"] = predicted_labels
    st.write(features["predicted_label"].value_counts())
    st.write("Predictions:")
    st.dataframe(features[["predicted_label"]].head())

    st.header("Detection Results")
    #attack distribution bar chart
    st.subheader("Attack Distribution")
    attack_counts = features["predicted_label"].value_counts()
    st.bar_chart(attack_counts)

    total_flows = len(features)
    malicious = len(features[features["predicted_label"] != "BENIGN"])
    benign = len(features[features["predicted_label"] == "BENIGN"])

    st.metric("Total Flows", total_flows)
    st.metric("Malicious Flows", malicious)
    st.metric("Benign Flows", benign)
    st.header("AI analysis of detected attacks")
    attack_types = features["predicted_label"].unique()
    attack_types = [a for a in attack_types if a != "BENIGN"]
    if len(attack_types) == 0:
        st.write("No attacks detected in this traffic sample.")

    else:
        for attack in attack_types:
            with st.expander(f"AI Explanation: {attack}"):
                with st.spinner("Generating explanation..."):
                    explanation = generate_explanation(
                        attack_type=attack,
                        confidence=round(attack_counts[attack] / total_flows * 100, 2),
                        top_features="packet rate, flow duration , flag counts"  # Placeholder for actual top features
                    )
                    st.write(explanation)   

st.header("Ask the Security Chatbot")
st.write("Ask any question to security chatbot")
user_question = st.text_input("Your question:....")
if user_question:
    with st.spinner("Thinking..."):
        chatbot_response = ask_security_chatbot(user_question)
        st.write(chatbot_response)