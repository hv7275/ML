import streamlit as st
import pandas as pd
import joblib

# ---- Load files correctly ----
model = joblib.load('NaivBayes_Heart.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

st.title("❤️ Heart Disease Prediction")
st.write("Provide the details below:")

# UI inputs
age = st.slider('Age', 18, 100, 40)
sex = st.selectbox('Sex', ['M', 'F'])
chestPain = st.selectbox('Chest Pain Type', ["ATA", 'NAP', 'TA', 'ASY'])
restingBP = st.number_input('Resting Blood Pressure (mm Hg)', 80, 200, 120)  # FIXED max
cholesterol = st.number_input('Cholesterol (mg/dL)', 100, 600, 200)           # FIXED max
fastingBS = st.selectbox('Fasting Blood Sugar >120 mg/dL', [0, 1])
restECG = st.selectbox('Resting ECG', ["Normal", "ST", "LVH"])
maxHR = st.slider('Max Heart Rate', 60, 220, 150)
exerciseAngina = st.selectbox('Exercise-Induced Angina', ['Y', 'N'])
oldPeak = st.slider('Oldpeak (ST Depression)', 0.0, 6.0, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    # 1) Create raw input DF
    raw = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chestPain,
        "RestingBP": restingBP,
        "Cholesterol": cholesterol,
        "FastingBS": fastingBS,
        "RestingECG": restECG,
        "MaxHR": maxHR,
        "ExerciseAngina": exerciseAngina,
        "Oldpeak": oldPeak,
        "ST_Slope": st_slope
    }])

    # 2) Numeric + categorical
    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'FastingBS']

    # 3) Scale numeric
    num_scaled = scaler.transform(raw[numeric_cols])
    num_df = pd.DataFrame(num_scaled, columns=numeric_cols)

    # 4) Encode categorical USING TRAINED ENCODER
    cat_encoded = encoder.transform(raw[categorical_cols])
    cat_df = pd.DataFrame(
        cat_encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # 5) Combine everything
    # Order matches training: scaled numerics then one-hot categoricals
    final_input = pd.concat([num_df, cat_df], axis=1)

    # 7) Predict
    pred = model.predict(final_input)[0]

    if pred == 1:
        st.error("⚠️ High risk of Heart Disease!")
    else:
        st.success("✅ Low risk of Heart Disease.")
