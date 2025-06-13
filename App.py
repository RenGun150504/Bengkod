import streamlit as st
import joblib
import numpy as np

# Judul Aplikasi
st.title('Aplikasi Prediksi Obesitas')
st.write('Aplikasi ini memprediksi tingkat obesitas berdasarkan fitur yang Anda masukkan.')

# Memuat model dan scaler yang telah disimpan
try:
    model = joblib.load('model_akhir.pkl')  # Pastikan path ini benar
    scaler = joblib.load('scaler.pkl')      # Pastikan path ini benar
except FileNotFoundError:
    st.error("Error: Model atau scaler tidak ditemukan. Pastikan file 'model_akhir.pkl' dan 'scaler.pkl' berada di direktori yang sama.")
    st.stop() # Hentikan aplikasi jika file tidak ditemukan

st.header('Masukkan Data Anda:')

# Contoh input fitur (Anda perlu menyesuaikannya dengan fitur-fitur aktual Anda)
# Asumsi: Fitur-fitur numerik yang perlu di-scale
# Ganti 'Fitur_1', 'Fitur_2', dst. dengan nama fitur sebenarnya dari dataset Anda
# Dan sesuaikan jenis input (st.number_input, st.selectbox, dll.) sesuai kebutuhan

gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
age = st.slider('Usia', 1, 100, 25)
height = st.number_input('Tinggi Badan (cm)', min_value=50.0, max_value=250.0, value=170.0)
weight = st.number_input('Berat Badan (kg)', min_value=10.0, max_value=200.0, value=70.0)
# Tambahkan fitur lainnya sesuai dengan dataset Anda
# Contoh:
# physical_activity_level = st.slider('Tingkat Aktivitas Fisik (kali/minggu)', 0, 7, 3)
# vegetable_consumption = st.slider('Konsumsi Sayuran (porsi/hari)', 0, 5, 2)
# ... dan seterusnya untuk semua fitur yang digunakan oleh model Anda

# Mengubah input kategorikal menjadi numerik (jika ada)
# Contoh untuk gender:
gender_encoded = 1 if gender == 'Wanita' else 0 # Sesuaikan encoding ini dengan yang Anda gunakan saat pelatihan

# Mengumpulkan semua input fitur ke dalam array numpy
# Pastikan urutan fitur ini SESUAI dengan urutan fitur yang digunakan saat melatih model
# Contoh:
# input_data = np.array([[gender_encoded, age, height, weight, physical_activity_level, vegetable_consumption, ...]])
input_data = np.array([[gender_encoded, age, height, weight]]) # Ganti dengan semua fitur Anda

# Scaling input data
try:
    scaled_input_data = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error saat scaling data: {e}. Pastikan fitur input sesuai dengan yang digunakan saat melatih scaler.")
    st.stop()


if st.button('Prediksi Obesitas'):
    # Melakukan prediksi
    prediction = model.predict(scaled_input_data)
    prediction_proba = model.predict_proba(scaled_input_data)

    st.subheader('Hasil Prediksi:')

    # Asumsi: Model Anda memprediksi kelas obesitas.
    # Anda mungkin perlu menyesuaikan pemetaan ini berdasarkan output kelas model Anda.
    # Contoh pemetaan kelas:
    class_mapping = {
        0: 'Kekurangan Berat Badan',
        1: 'Berat Badan Normal',
        2: 'Kelebihan Berat Badan',
        3: 'Obesitas Tipe I',
        4: 'Obesitas Tipe II',
        5: 'Obesitas Tipe III'
    }
    
    predicted_class_label = class_mapping.get(prediction[0], 'Kelas Tidak Dikenal')

    st.success(f'Berdasarkan data yang Anda masukkan, prediksi tingkat obesitas adalah: **{predicted_class_label}**')

    st.write('Probabilitas untuk setiap kelas:')
    for i, prob in enumerate(prediction_proba[0]):
        st.write(f'- {class_mapping.get(i, f"Kelas {i}")}: {prob:.2f}')

st.markdown('---')
st.markdown('Aplikasi ini dibuat oleh Rendra Gunawan (A11.2022.14235).')
