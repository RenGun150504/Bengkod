import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
@st.cache_resource
def load_resources():
    model = joblib.load('model_akhir.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_resources()

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

st.set_page_config(page_title='Prediksi Obesitas', layout='wide')

st.markdown("""
# Prediksi Tingkat Obesitas
Masukkan data diri dan kebiasaan untuk mendapatkan prediksi kategori obesitas.
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    with st.expander('1. Data Diri', expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input('Usia (tahun)', min_value=0, value=25, step=1)
            height = st.number_input('Tinggi Badan (cm)', min_value=0.0, value=170.0, step=1.0)
        with c2:
            weight = st.number_input('Berat Badan (kg)', min_value=0.0, value=70.0, step=1.0)
            gender = st.selectbox('Jenis Kelamin', options=list(gender_map.keys()))
        with c3:
            family_history = st.selectbox('Riwayat Keluarga kelebihan berat badan', options=list(maps['Riwayat Keluarga'].keys()))

    with st.expander('2. Kebiasaan & Gaya Hidup'):
        c1, c2, c3 = st.columns(3)
        with c1:
            favc = st.selectbox('Makanan Tinggi Kalori', options=list(maps['FAVC'].keys()))
            fcvc = st.number_input('Konsumsi Sayur (porsi/hari)', min_value=0.0, value=2.0, step=1.0)
        with c2:
            cp = st.slider('Jumlah Makan Utama/hari', min_value=1, max_value=6, value=3)
            caec = st.selectbox('Camilan di Luar Jam Makan', options=list(maps['CAEC'].keys()))
        with c3:
            smoke = st.selectbox('Merokok', options=list(maps['SMOKE'].keys()))

    with st.expander('3. Informasi Lainnya'):
        c1, c2, c3 = st.columns(3)
        with c1:
            ch2o = st.number_input('Air Putih (L/hari)', min_value=0.0, value=1.5, step=1.0)
            scc = st.selectbox('Mencatat Asupan Kalori', options=list(maps['SCC'].keys()))
        with c2:
            faf = st.number_input('Olahraga (kali/minggu)', min_value=0, value=3, step=1)
            tue = st.number_input('Waktu Gadget (jam/hari)', min_value=0.0, value=5.0, step=1.0)
        with c3:
            calc = st.selectbox('Konsumsi Alkohol', options=list(maps['CALC'].keys()))
            mtrans = st.selectbox('Moda Transportasi', options=list(maps['MTRANS'].keys()))

    predict_button = st.button('🔍 Prediksi')
with col2:
    st.image('https://www.clipartmax.com/png/full/377-3774794_fat-loss-png-obesity-clip-art.png', caption='Obesitas', use_column_width=True)

if predict_button:
    with st.spinner('Memproses prediksi...'):
        raw = {
            'Age': age, 'Height': height, 'Weight': weight,
            'FCVC': fcvc, 'NCP': cp, 'CH2O': ch2o,
            'FAF': faf, 'TUE': tue,
            'Gender': gender_map[gender],
            'family_history_with_overweight': maps['Riwayat Keluarga'][family_history],
            'FAVC': maps['FAVC'][favc], 'CAEC': maps['CAEC'][caec],
            'SMOKE': maps['SMOKE'][smoke], 'SCC': maps['SCC'][scc],
            'CALC': maps['CALC'][calc], 'MTRANS': maps['MTRANS'][mtrans]
        }
        df = pd.DataFrame([raw])
        num_cols = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
        df[num_cols] = scaler.transform(df[num_cols])
        cat_cols = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']
        dummies = pd.get_dummies(df[cat_cols], drop_first=True)
        X = pd.concat([df[num_cols], dummies], axis=1)
        result = model.predict(X)[0]
    st.success(f'Kategori obesitas: **{result}**')
    st.balloons()
