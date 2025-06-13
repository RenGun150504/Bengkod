import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load('random_forest_obesity_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan. Pastikan file 'random_forest_obesity_model.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
    st.stop()

# Map options for categorical inputs (display -> model value)
gender_map = {'Laki-laki': 'Male', 'Perempuan': 'Female'}
family_history_map = {'Ya': 'yes', 'Tidak': 'no'}
favc_map = {'Ya': 'yes', 'Tidak': 'no'}
caec_map = {'Tidak Pernah': 'Never', 'Kadang-kadang': 'Sometimes', 'Sering': 'Frequently', 'Selalu': 'Always'}
smoke_map = {'Merokok': 'yes', 'Tidak Merokok': 'no'}
scc_map = {'Mencatat Kalori': 'yes', 'Tidak Mencatat': 'no'}
calc_map = {'Tidak Minum Alkohol': 'no', 'Kadang-kadang': 'Sometimes', 'Sering': 'Frequently', 'Selalu': 'Always'}
mtrans_map = {
    'Jalan Kaki': 'Walking',
    'Sepeda': 'Bike',
    'Motor': 'Motorbike',
    'Mobil': 'Automobile',
    'Transportasi Umum': 'Public_Transportation'
}

st.title('Prediksi Tingkat Obesitas')
st.write('Isi data berikut untuk memprediksi kategori obesitas:')

# Sidebar Input
st.sidebar.header('1. Identitas')
age = st.sidebar.number_input('Usia (tahun)', min_value=0, value=25)
height = st.sidebar.number_input('Tinggi Badan (cm)', min_value=0.0, format="%.1f", value=170.0)
weight = st.sidebar.number_input('Berat Badan (kg)', min_value=0.0, format="%.1f", value=70.0)
gender = st.sidebar.selectbox('Jenis Kelamin', options=list(gender_map.keys()))
family_history = st.sidebar.selectbox('Riwayat Keluarga kelebihan berat badan', options=list(family_history_map.keys()))

st.sidebar.header('2. Kebiasaan & Gaya Hidup')
favc = st.sidebar.selectbox('Sering Makan Makanan Tinggi Kalori', options=list(favc_map.keys()))
fcvc = st.sidebar.number_input('Frekuensi Konsumsi Sayur (porsi/hari)', min_value=0.0, format="%.1f", value=2.0)
cp = st.sidebar.number_input('Jumlah Makan Utama per Hari', min_value=1, max_value=6, value=3)
caec = st.sidebar.selectbox('Camilan di Luar Jam Makan', options=list(caec_map.keys()))
smoke = st.sidebar.selectbox('Kebiasaan Merokok', options=list(smoke_map.keys()))

st.sidebar.header('3. Informasi Lainnya')
ch2o = st.sidebar.number_input('Konsumsi Air Putih (liter/hari)', min_value=0.0, format="%.1f", value=1.5)
scc = st.sidebar.selectbox('Mencatat Asupan Kalori', options=list(scc_map.keys()))
faf = st.sidebar.number_input('Frekuensi Olahraga (kali/minggu)', min_value=0, value=3)
tue = st.sidebar.number_input('Waktu Menggunakan Gadget (jam/hari)', min_value=0.0, format="%.1f", value=5.0)
calc = st.sidebar.selectbox('Konsumsi Alkohol', options=list(calc_map.keys()))
mtrans = st.sidebar.selectbox('Moda Transportasi Utama', options=list(mtrans_map.keys()))

if st.sidebar.button('Prediksi'):
    # Prepare input dictionary
    data = {
        'Age': age,
        'Height': height,
        'Weight': weight,
        'FCVC': fcvc,
        'NCP': cp,
        'CH2O': ch2o,
        'FAF': faf,
        'TUE': tue,
        'Gender': gender_map[gender],
        'family_history_with_overweight': family_history_map[family_history],
        'FAVC': favc_map[favc],
        'CAEC': caec_map[caec],
        'SMOKE': smoke_map[smoke],
        'SCC': scc_map[scc],
        'CALC': calc_map[calc],
        'MTRANS': mtrans_map[mtrans]
    }
    input_df = pd.DataFrame([data])

    # Scale numeric features
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # One-hot encoding
    cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    dummies = pd.get_dummies(input_df[cat_cols], drop_first=True)
    X_input = pd.concat([input_df[num_cols], dummies], axis=1)
    X_input = X_input.reindex(columns=feature_columns, fill_value=0)

    # Prediction
    pred = model.predict(X_input)[0]
    st.subheader('Hasil Prediksi')
    st.write(f"Kategori obesitas: **{pred}**")
else:
    st.info('Isi semua input di sidebar lalu klik Prediksi.')
