import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

# --- Display the image safely and resized ---
if os.path.exists("heart_banner.png"):
    img = Image.open("heart_banner.png")
    img = img.resize((600, 400))  # Resize to look normal
    st.image(img, caption="🫀 Heart Checkup")
else:
    st.warning("⚠️ Image not found. Please add 'heart_banner.png' to the app folder.")

st.title("💓 Heart Disease Risk Predictor")
st.write("Enter the patient's medical details below to predict the likelihood of heart disease.")

# --- Load trained model ---
model = tf.keras.models.load_model("heart_model.keras", compile=False)


# --- Normalization values (cp removed, so shape is 14) ---
x_min = np.array([28.0, 0.0, 0.0, 0.0, 0.0, 60.0, 0.0, -2.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
x_max = np.array([77.0, 1.0, 200.0, 603.0, 1.0, 1.0, 202.0, 1.0, 6.2, 3.0, 1.0, 1.0, 1.0, 1.0])

# --- Input Form ---
with st.form("input_form"):
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    restecg = st.selectbox("Resting ECG Results", ["LV Hypertrophy", "Normal"])
    thalch = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise-Induced Angina", ["True", "False"])
    oldpeak = st.slider("ST Depression Induced", 0.0, 6.2, 1.0)
    ca = st.slider("Number of Major Vessels Colored", 0, 3, 0)
    slope = st.selectbox("Slope of Peak Exercise", ["Flat", "Downsloping", "Upsloping"])
    thal = st.selectbox("Thalassemia", ["Normal", "Reversable Defect", "Fixed Defect"])
    submit = st.form_submit_button("Predict")

# --- Process and Predict ---
if submit:
    sex = 0 if sex == "Male" else 1
    fbs = 1 if fbs == "True" else 0
    restecg = 0 if restecg == "LV Hypertrophy" else 1
    exang = 1 if exang == "True" else 0
    slope_map = {"Flat": [1, 0], "Downsloping": [0, 0], "Upsloping": [0, 1]}
    thal_map = {"Normal": [1, 0], "Reversable Defect": [0, 1], "Fixed Defect": [0, 0]}
    slope_flat, slope_upsloping = slope_map[slope]
    thal_normal, thal_reversable = thal_map[thal]

    input_data = np.array([[age, sex, trestbps, chol, fbs, restecg, thalch,
                            exang, oldpeak, ca, thal_normal, thal_reversable,
                            slope_flat, slope_upsloping]], dtype=np.float32)

    input_data = (input_data - x_min) / (x_max - x_min)

    prediction = model.predict(input_data)[0][0]

    # Display confidence meter
    st.subheader("🧠 Confidence Meter")
    fig, ax = plt.subplots(figsize=(5, 0.5))
    ax.barh([0], [prediction], color="red" if prediction > 0.5 else "green")
    ax.set_xlim(0, 1)
    ax.axis('off')
    st.pyplot(fig)

    # Show result
    st.markdown(f"### 🩺 Prediction Probability: **{prediction:.2%}**")

    if prediction > 0.5:
        st.error("🚨 You are at **high risk** of heart disease.")

        st.markdown("#### ❤️ Health Tips:")
        st.markdown("""
        - 🥗 Eat more fruits, vegetables, and whole grains  
        - 🚶‍♂️ Exercise regularly (30 minutes/day)  
        - 🚭 Avoid smoking and limit alcohol  
        - 💧 Stay hydrated  
        - 🧘‍♀️ Manage stress with relaxation techniques  
        - 🩺 Schedule regular checkups
        """)
    else:
        st.success("✅ You are at **low risk** of heart disease.")
        jokes = [
            "Why did the heart go to school? To get to the heart of the subject!",
            "You must be cardiac tissue, because you’ve got a lot of heart ❤️",
            "Heart says gym, brain says pizza. You chose well!"
        ]
        st.markdown("### 😄 Here's a joke to make your heart smile:")
        st.info(random.choice(jokes))
