import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title='Prediksi Tingkat Obesitas',
    layout='centered', # Menggunakan layout centered untuk tampilan yang lebih fokus
    initial_sidebar_state='collapsed'
)

# --- Fungsi untuk Memuat Model dan Scaler (menggunakan cache) ---
@st.cache_resource
def load_resources():
    try:
        # Perhatikan: Sesuaikan path jika model dan scaler tidak berada di direktori yang sama
        model = joblib.load('model_akhir.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Pastikan file 'model_akhir.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model atau scaler: {e}")
        st.stop()

model, scaler = load_resources()

# --- Definisi Pemetaan untuk Dropdown ---
gender_map = {'Laki-laki': 'Male', 'Perempuan': 'Female'}
maps = {
    'Riwayat Keluarga': {'Ya': 'yes', 'Tidak': 'no'},
    'FAVC': {'Ya': 'yes', 'Tidak': 'no'},
    'CAEC': {'Tidak Pernah': 'Never', 'Kadang-kadang': 'Sometimes', 'Sering': 'Frequently', 'Selalu': 'Always'},
    'SMOKE': {'Merokok': 'yes', 'Tidak Merokok': 'no'},
    'SCC': {'Mencatat': 'yes', 'Tidak Mencatat': 'no'},
    'CALC': {'Tidak Minum Alkohol': 'no', 'Kadang-kadang': 'Sometimes', 'Sering': 'Frequently', 'Selalu': 'Always'},
    'MTRANS': {'Jalan Kaki': 'Walking', 'Sepeda': 'Bike', 'Motor': 'Motorbike', 'Mobil': 'Automobile', 'Transportasi Umum': 'Public_Transportation'}
}

# --- Daftar Kolom Numerik dan Kategorikal (penting untuk pre-processing) ---
# PASTIKAN NAMA DAN URUTAN KOLOM INI SESUAI DENGAN YANG DIGUNAKAN SAAT PELATIHAN MODEL ANDA!
NUM_COLS = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
CAT_COLS = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

# --- Class Mapping untuk Hasil Prediksi ---
# SESUAIKAN INI DENGAN LABEL KELAS OUTPUT DARI MODEL ANDA!
CLASS_MAPPING_OBESITY = {
    'Normal_Weight': 'Berat Badan Normal',
    'Overweight_Level_I': 'Kelebihan Berat Badan Tingkat I',
    'Overweight_Level_II': 'Kelebihan Berat Badan Tingkat II',
    'Obesity_Type_I': 'Obesitas Tipe I',
    'Obesity_Type_II': 'Obesitas Tipe II',
    'Obesity_Type_III': 'Obesitas Tipe III',
    'Insufficient_Weight': 'Kekurangan Berat Badan'
}

# --- Styling CSS Injeksi ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.8em; /* Sedikit lebih kecil untuk layout centered */
        font-weight: bold;
        color: #2E8B57; /* Warna hijau laut */
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .subheader {
        font-size: 1.2em;
        color: #36454F; /* Charcoal */
        text-align: center;
        margin-bottom: 1.5em;
    }
    .stExpander {
        border: 1px solid #D3D3D3;
        border-radius: 10px;
        box-shadow: 1px 1px 8px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #4CAF50; /* Hijau */
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-size: 1.1em;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 3px 5px rgba(0,0,0,0.1);
        width: 100%; /* Tombol lebar penuh */
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-1px);
        box-shadow: 0 5px 7px rgba(0,0,0,0.15);
    }
    .stSuccess {
        background-color: #e6ffe6; /* Light green for success */
        color: #2E8B57;
        padding: 18px; /* Padding lebih besar */
        border-radius: 12px; /* Radius lebih bulat */
        border: 1px solid #4CAF50;
        font-size: 1.5em; /* Ukuran font lebih besar */
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 25px; /* Jarak dari atas */
        font-weight: bold;
    }
    .stInfo {
        background-color: #e0f2f7; /* Light blue for info */
        color: #2b6a7a;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #62ADD3;
        font-size: 0.95em;
        margin-top: 15px;
    }
</style>
<h1 class="main-header">🩺 Prediksi Tingkat Obesitas Individu</h1>
<p class="subheader">Dapatkan gambaran singkat kategori obesitas Anda berdasarkan data yang akurat.</p>
""", unsafe_allow_html=True)

# --- Area Input Form ---
st.subheader('Masukkan Informasi Anda:')

with st.expander('**1. Data Diri & Riwayat Kesehatan** 👤', expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input('Usia (tahun)', min_value=1, max_value=100, value=25, step=1, help='Usia Anda dalam tahun.')
    with c2:
        gender = st.selectbox('Jenis Kelamin', options=list(gender_map.keys()), help='Pilih jenis kelamin Anda.')
    with c3:
        family_history = st.selectbox('Riwayat Keluarga kelebihan berat badan', options=list(maps['Riwayat Keluarga'].keys()), help='Apakah ada riwayat obesitas dalam keluarga Anda?')

    c4, c5 = st.columns(2)
    with c4:
        height = st.number_input('Tinggi Badan (cm)', min_value=50.0, max_value=250.0, value=170.0, step=0.5, help='Tinggi badan Anda dalam centimeter.')
    with c5:
        weight = st.number_input('Berat Badan (kg)', min_value=10.0, max_value=250.0, value=70.0, step=0.5, help='Berat badan Anda dalam kilogram.')

with st.expander('**2. Kebiasaan Makan & Gaya Hidup** 🍎🍜', expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        favc = st.selectbox('Sering mengonsumsi makanan tinggi kalori?', options=list(maps['FAVC'].keys()), help='Seberapa sering Anda makan makanan tinggi kalori?')
        fcvc = st.slider('Konsumsi Sayur (porsi/hari)', min_value=0.0, max_value=5.0, value=2.0, step=0.5, help='Berapa porsi sayuran yang Anda makan per hari?')
    with c2:
        cp = st.slider('Jumlah Makan Utama/hari', min_value=1, max_value=6, value=3, step=1, help='Berapa kali Anda makan makanan utama dalam sehari?')
        caec = st.selectbox('Camilan di Luar Jam Makan', options=list(maps['CAEC'].keys()), help='Seberapa sering Anda ngemil di luar jam makan utama?')
    with c3:
        smoke = st.selectbox('Merokok?', options=list(
