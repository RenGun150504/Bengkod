import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title='Prediksi Tingkat Obesitas',
    layout='centered',
    initial_sidebar_state='collapsed',
    menu_items={
        'Get Help': 'https://www.streamlit.io/docs',
        'Report a bug': "https://www.github.com/streamlit/streamlit/issues",
        'About': "# Aplikasi Prediksi Obesitas. Dibuat untuk proyek data science."
    }
)

# --- Styling CSS Kustom (Modern & Minimalis) ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6; /* Warna latar belakang lembut */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .css-1d391kg { /* Target the main content area */
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    h1 {
        color: #2c3e50; /* Darker blue-grey for headers */
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5em;
    }
    h2 {
        color: #34495e;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 1em;
    }
    .stButton>button {
        background-color: #3498db; /* Biru cerah */
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
        width: 100%;
        box_shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #2980b9; /* Biru lebih gelap saat hover */
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stSuccess {
        background-color: #e8f5e9; /* Light green for success */
        color: #2e7d32; /* Dark green text */
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 2rem;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .stExpander > div > div > p { /* Target expander title */
        font-weight: 600;
        color: #4a4a4a;
    }
    .stInfo {
        background-color: #e3f2fd; /* Light blue for info */
        color: #1976d2; /* Dark blue text */
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 0.75rem;
        margin-top: 1.5rem;
        font-size: 0.9rem;
    }
    /* Style for text inputs/select boxes */
    .st-dg { /* Selectbox label */
        font-weight: 500;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- Fungsi untuk Memuat Model dan Scaler (dan mendapatkan feature names dari model) ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model_akhir.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # --- Kunci Perbaikan: Mengambil feature names langsung dari model ---
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_.tolist()
            st.success("✅ Model, scaler, dan daftar fitur berhasil dimuat dari model!")
        else:
            st.error("❌ Peringatan: Model tidak memiliki atribut 'feature_names_in_'. Prediksi mungkin tidak akurat jika urutan/nama fitur berbeda.")
            st.warning("Direkomendasikan untuk menyimpan daftar fitur secara eksplisit dari Colab.")
            # Sebagai fallback, jika model tidak memiliki feature_names_in_, kita perlu menebak.
            # Ini sangat berisiko dan bisa menyebabkan error lagi.
            # Berdasarkan notebook Anda, ada 23 fitur: 8 numerik + 15 one-hot encoded
            # Anda HARUS mengganti ini dengan urutan yang BENAR dari fitur Anda
            # jika model Anda tidak memiliki feature_names_in_.
            # Contoh (ini masih placeholder, sesuaikan dengan output X_train.columns.tolist() terakhir):
            model_features = [
                'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
                'Gender_Male',
                'family_history_with_overweight_yes',
                'FAVC_yes',
                'CAEC_Sometimes', 'CAEC_Frequently', 'CAEC_Always', 'CAEC_no', # Tambahkan semua kemungkinan dummy
                'SMOKE_yes',
                'SCC_yes',
                'CALC_Sometimes', 'CALC_Frequently', 'CALC_Always', 'CALC_no', # Tambahkan semua kemungkinan dummy
                'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking', 'MTRANS_Bike', 'MTRANS_Automobile' # Tambahkan semua kemungkinan dummy
            ]
            st.info("Menggunakan daftar fitur default. Mohon pastikan ini sesuai dengan model Anda.")
            
        return model, scaler, model_features
    except FileNotFoundError as e:
        st.error(f"❌ Error: File sumber daya tidak ditemukan. Pastikan 'model_akhir.pkl' dan 'scaler.pkl' ada di direktori yang sama dengan aplikasi. Detail: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error saat memuat sumber daya: {e}")
        st.exception(e) # Menampilkan traceback lengkap di UI untuk debugging
        st.stop()

model, scaler, MODEL_FEATURES = load_resources()

# --- Pemetaan untuk Dropdown UI (Indonesia ke Bahasa Inggris untuk Model) ---
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

# --- Daftar Kolom Numerik dan Kategorikal (sesuai data asli sebelum encoding) ---
NUM_COLS = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
CAT_COLS = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# --- Class Mapping untuk Hasil Prediksi ---
CLASS_MAPPING_OBESITY = {
    'Normal_Weight': 'Berat Badan Normal',
    'Overweight_Level_I': 'Kelebihan Berat Badan Tingkat I',
    'Overweight_Level_II': 'Kelebihan Berat Badan Tingkat II',
    'Obesity_Type_I': 'Obesitas Tipe I',
    'Obesity_Type_II': 'Obesitas Tipe II',
    'Obesity_Type_III': 'Obesitas Tipe III',
    'Insufficient_Weight': 'Kekurangan Berat Badan'
}

# --- Header Aplikasi ---
st.markdown("<h1>🩺 Prediksi Tingkat Obesitas Individu</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Dapatkan gambaran singkat kategori obesitas Anda.</p>", unsafe_allow_html=True)

st.markdown("---") # Garis pemisah

# --- Area Input Form ---
st.subheader('📥 Masukkan Data Anda')

# Membuat section yang jelas
st.markdown("<h3><small>Data Diri & Riwayat</small></h3>", unsafe_allow_html=True)
with st.expander('Klik untuk Data Diri', expanded=True):
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

st.markdown("<h3><small>Kebiasaan Makan & Gaya Hidup</small></h3>", unsafe_allow_html=True)
with st.expander('Klik untuk Kebiasaan', expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        favc = st.selectbox('Sering mengonsumsi makanan tinggi kalori?', options=list(maps['FAVC'].keys()), help='Seberapa sering Anda makan makanan tinggi kalori?')
        fcvc = st.slider('Konsumsi Sayur (porsi/hari)', min_value=0.0, max_value=5.0, value=2.0, step=0.5, help='Berapa porsi sayuran yang Anda makan per hari?')
    with c2:
        cp = st.slider('Jumlah Makan Utama/hari', min_value=1, max_value=6, value=3, step=1, help='Berapa kali Anda makan makanan utama dalam sehari?')
        caec = st.selectbox('Camilan di Luar Jam Makan', options=list(maps['CAEC'].keys()), help='Seberapa sering Anda ngemil di luar jam makan utama?')
    
    smoke = st.selectbox('Merokok?', options=list(maps['SMOKE'].keys()), help='Apakah Anda seorang perokok?')
    calc = st.selectbox('Konsumsi Alkohol', options=list(maps['CALC'].keys()), help='Seberapa sering Anda mengonsumsi alkohol?')

st.markdown("<h3><small>Aktivitas & Lainnya</small></h3>", unsafe_allow_html=True)
with st.expander('Klik untuk Aktivitas & Lainnya', expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        ch2o = st.slider('Konsumsi Air Putih (liter/hari)', min_value=0.0, max_value=5.0, value=1.5, step=0.1, help='Berapa liter air putih yang Anda minum per hari?')
        faf = st.slider('Olahraga (kali/minggu)', min_value=0, max_value=7, value=3, step=1, help='Berapa kali Anda berolahraga dalam seminggu?')
    with c2:
        tue = st.slider('Waktu Gadget (jam/hari)', min_value=0.0, max_value=15.0, value=5.0, step=0.5, help='Berapa jam Anda menghabiskan waktu di depan gadget/layar per hari?')
        mtrans = st.selectbox('Moda Transportasi Utama', options=list(maps['MTRANS'].keys()), help='Bagaimana Anda bepergian sehari-hari?')
    
    scc = st.selectbox('Mencatat Asupan Kalori?', options=list(maps['SCC'].keys()), help='Apakah Anda memiliki kebiasaan mencatat asupan kalori?')

st.markdown("---") # Garis pemisah sebelum tombol
predict_button = st.button('🚀 Prediksi Kategori Obesitas Anda')

# --- Logika Prediksi dan Tampilan Hasil ---
if predict_button:
    with st.spinner('Menganalisis data dan membuat prediksi...'):
        try:
            # Membangun dictionary raw data
            raw_data = {
                'Age': age, 'Height': height, 'Weight': weight,
                'FCVC': fcvc, 'NCP': cp, 'CH2O': ch2o,
                'FAF': faf, 'TUE': tue,
                'Gender': gender_map[gender],
                'family_history_with_overweight': maps['Riwayat Keluarga'][family_history],
                'FAVC': maps['FAVC'][favc], 'CAEC': maps['CAEC'][caec],
                'SMOKE': maps['SMOKE'][smoke], 'SCC': maps['SCC'][scc],
                'CALC': maps['CALC'][calc], 'MTRANS': maps['MTRANS'][mtrans]
            }
            df = pd.DataFrame([raw_data])

            # Scaling fitur numerik
            df[NUM_COLS] = scaler.transform(df[NUM_COLS])

            # One-Hot Encoding untuk fitur kategorikal
            # Penting: drop_first=True harus konsisten dengan pelatihan model Anda
            dummies = pd.get_dummies(df[CAT_COLS], drop_first=True)
            
            # Gabungkan kolom numerik dan dummy
            X_processed = pd.concat([df[NUM_COLS], dummies], axis=1)

            # --- Kunci Perbaikan: Reindexasi Menggunakan MODEL_FEATURES yang Dimuat atau Diambil ---
            # Ini akan memastikan X_final memiliki semua kolom yang diharapkan model,
            # dalam urutan yang benar, dan mengisi 0 untuk kolom dummy yang tidak muncul
            # di input saat ini.
            X_final = X_processed.reindex(columns=MODEL_FEATURES, fill_value=0)

            # --- Debugging Tambahan (opsional, bisa di-uncomment jika ada masalah) ---
            # st.write("Kolom X_final (setelah reindex):", X_final.columns.tolist())
            # st.write("Data X_final (siap prediksi):", X_final)
            # if set(MODEL_FEATURES) != set(X_final.columns.tolist()):
            #    st.error("Ketidakcocokan kolom terdeteksi setelah reindex!")
            #    missing_in_X_final = set(MODEL_FEATURES) - set(X_final.columns.tolist())
            #    st.error(f"Kolom hilang di X_final: {list(missing_in_X_final)}")
            #    extra_in_X_final = set(X_final.columns.tolist()) - set(MODEL_FEATURES)
            #    st.error(f"Kolom ekstra di X_final: {list(extra_in_X_final)}")
            # elif MODEL_FEATURES != X_final.columns.tolist():
            #    st.warning("Urutan kolom berbeda! Mungkin tidak masalah untuk beberapa model tapi baiknya cocok.")


            # Lakukan prediksi
            prediction = model.predict(X_final)[0]
            
            # Tampilkan hasil kesimpulan
            st.markdown(f"""
            <div class="stSuccess">
                Kategori Obesitas Anda: <strong>{CLASS_MAPPING_OBESITY.get(prediction, prediction)}</strong>
            </div>
            """, unsafe_allow_html=True)

            st.balloons() # Efek balon saat sukses

            st.info("""
            *Penting: Prediksi ini adalah hasil dari model machine learning berdasarkan data yang Anda berikan. Untuk diagnosis medis yang akurat dan rencana penanganan yang tepat, sangat disarankan untuk berkonsultasi dengan profesional kesehatan.*
            """)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan dalam proses prediksi: {e}")
            st.warning("Mohon periksa kembali semua data yang Anda masukkan dan pastikan model serta scaler dimuat dengan benar.")
            st.exception(e) # Menampilkan traceback lengkap di UI untuk debugging

# --- Footer Aplikasi ---
st.markdown("""
<hr style="border:1px solid #f0f2f6; margin-top: 3rem;">
<p style="text-align: center; color: #7f8c8d; font-size: 0.8em;">
    Dibuat oleh Rendra Gunawan (A11.2022.14235). <br> Untuk prediksi obesitas.
</p>
""", unsafe_allow_html=True)
