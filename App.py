import streamlit as st
import pandas as pd
import joblib

# Muat model dan scaler
try:
    model = joblib.load('random_forest_obesity_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan. Pastikan file 'random_forest_obesity_model.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
    st.stop()

# Definisikan kolom numerik dan kategorikal sesuai preprocessing di Colab
numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

st.title('Prediksi Tingkat Obesitas')
st.write('Masukkan nilai fitur berikut untuk memprediksi kategori obesitas:')

# Input fitur numerik
ing = {}
for col in numeric_cols:
    ing[col] = st.sidebar.number_input(
        label=col,
        min_value=0.0,
        format="%.2f",
        value=1.0 if col != 'Age' else 25.0
    )

# Input fitur kategorikal
options = {
    'Gender': ['Male', 'Female'],
    'family_history_with_overweight': ['yes', 'no'],
    'FAVC': ['yes', 'no'],
    'CAEC': ['Never', 'Sometimes', 'Frequently', 'Always'],
    'SMOKE': ['yes', 'no'],
    'SCC': ['yes', 'no'],
    'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],
    'MTRANS': ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking']
}
for col, opts in options.items():
    ing[col] = st.sidebar.selectbox(label=col, options=opts)

if st.button('Prediksi'):
    # Buat DataFrame input
    input_df = pd.DataFrame([ing])

    # Scaling numerik
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # One-hot encoding kategory
    dummies = pd.get_dummies(input_df[categorical_cols], drop_first=True)

    # Gabungkan semua fitur
    X_input = pd.concat([input_df[numeric_cols], dummies], axis=1)
    X_input = X_input.reindex(columns=feature_columns, fill_value=0)

    # Prediksi
    pred = model.predict(X_input)[0]
    st.subheader('Hasil Prediksi')
    st.write(f"Kategori obesitas: **{pred}**")
