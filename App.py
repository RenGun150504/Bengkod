import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Muat model dan scaler
try:
    model = joblib.load('model_akhir.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan. Pastikan file 'random_forest_obesity_model.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
    st.stop()

# Judul dan deskripsi aplikasi
st.title("Aplikasi Prediksi Tingkat Obesitas")
st.markdown("Muhammad Satriya Pratama Manggala Kusuma-(A11.2022.14225)")
st.subheader("Masukkan data di bawah ini untuk memprediksi tingkat obesitas Anda.")

col1, col2 = st.columns(2)

with col1:
    st.header("Atribut Fisik")
    age = st.slider("Usia (Tahun)", 10, 70, 25)
    height = st.slider("Tinggi Badan (Meter)", 1.40, 2.00, 1.75, 0.01)
    weight = st.slider("Berat Badan (Kg)", 30.0, 200.0, 70.0, 0.5)
    gender = st.selectbox("Jenis Kelamin", ['Pria', 'Wanita'])
    family_history = st.selectbox("Riwayat obesitas dalam keluarga?", ['Ya', 'Tidak'])

with col2:
    st.header("Kebiasaan Gaya Hidup")
    favc = st.selectbox("Sering makan makanan berkalori tinggi (FAVC)?", ['Ya', 'Tidak'])
    fcvc = st.slider("Frekuensi makan sayur (FCVC)", 1, 3, 2)
    ncp = st.slider("Jumlah makan utama per hari (NCP)", 1, 4, 3)
    caec = st.selectbox("Makan di antara waktu makan utama (CAEC)?", ['Tidak', 'Kadang-kadang', 'Sering', 'Selalu'])
    smoke = st.selectbox("Apakah Anda merokok (SMOKE)?", ['Ya', 'Tidak'])

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.header("Aktivitas dan Konsumsi Lainnya")
    ch2o = st.slider("Konsumsi air harian (Liter) (CH2O)", 1, 3, 2)
    scc = st.selectbox("Memantau konsumsi kalori (SCC)?", ['Ya', 'Tidak'])
    faf = st.slider("Frekuensi aktivitas fisik per minggu (FAF)", 0, 3, 1)
    tue = st.slider("Waktu penggunaan gawai per hari (Jam) (TUE)", 0, 2, 1)
    calc = st.selectbox("Konsumsi alkohol (CALC)?", ['Tidak', 'Kadang-kadang', 'Sering', 'Selalu'])
    mtrans = st.selectbox("Transportasi yang digunakan (MTRANS)", ['Mobil', 'Motor', 'Sepeda', 'Transportasi Umum', 'Jalan Kaki'])

if st.button("Prediksi Tingkat Obesitas", type="primary"):
    input_data = {
        'Age': age, 'Height': height, 'Weight': weight, 'FCVC': fcvc, 'NCP': ncp,
        'CH2O': ch2o, 'FAF': faf, 'TUE': tue,
        'Gender': 'Male' if gender == 'Pria' else 'Female',
        'family_history_with_overweight': 'yes' if family_history == 'Ya' else 'no',
        'FAVC': 'yes' if favc == 'Ya' else 'no',
        'CAEC': caec.replace('Sering', 'Frequently').replace('Kadang-kadang', 'Sometimes').replace('Selalu', 'Always').replace('Tidak', 'no'),
        'SMOKE': 'yes' if smoke == 'Ya' else 'no',
        'SCC': 'yes' if scc == 'Ya' else 'no',
        'CALC': calc.replace('Sering', 'Frequently').replace('Kadang-kadang', 'Sometimes').replace('Selalu', 'Always').replace('Tidak', 'no'),
        'MTRANS': mtrans.replace('Mobil', 'Automobile').replace('Motor', 'Motorbike').replace('Sepeda', 'Bike').replace('Transportasi Umum', 'Public_Transportation').replace('Jalan Kaki', 'Walking')
    }

    input_df_raw = pd.DataFrame([input_data])
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df_raw[numerical_cols] = scaler.transform(input_df_raw[numerical_cols])

    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    input_df_encoded = pd.get_dummies(input_df_raw, columns=categorical_cols)

    training_columns = [
        'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
        'Gender_Female', 'Gender_Male',
        'family_history_with_overweight_no', 'family_history_with_overweight_yes',
        'FAVC_no', 'FAVC_yes',
        'CAEC_Always', 'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no',
        'SMOKE_no', 'SMOKE_yes',
        'SCC_no', 'SCC_yes',
        'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 'CALC_no',
        'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
        'MTRANS_Public_Transportation', 'MTRANS_Walking'
    ]

    input_final = input_df_encoded.reindex(columns=training_columns, fill_value=0)

    prediction = model.predict(input_final)
    prediction_proba = model.predict_proba(input_final)

    st.subheader("Hasil Prediksi")

    result = prediction[0].replace("_", " ")
    
    st.success(f"Anda diprediksi memiliki tingkat: **{result}**")
    
    st.write("Probabilitas untuk setiap kategori:")
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=model.classes_,
        index=['Probabilitas']
    ).T
    st.bar_chart(proba_df)
