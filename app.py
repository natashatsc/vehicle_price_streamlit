import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

st.title("Prediksi Harga Mobil Second")

# Load model
model = joblib.load("catboost.pkl")

# Input dari user
st.write("### Masukkan informasi mobil Anda untuk prediksi harga jual:")

year = st.number_input("Tahun mobil", min_value=1990, max_value=2025, value=2018)
km_driven = st.number_input("Kilometer ditempuh", min_value=0, value=45000)
fuel = st.selectbox("Jenis bahan bakar", ["Petrol", "Diesel", "CNG", "LPG"])
seller_type = st.selectbox("Tipe penjual", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmisi", ["Manual", "Automatic"])
owner = st.selectbox("Status kepemilikan", [
    "First Owner", 
    "Second Owner", 
    "Third Owner", 
    "Fourth & Above Owner", 
    "Test Drive Car"
])
mileage = st.number_input("Mileage (km/ltr/kg)", min_value=0.0, value=18.5)
engine = st.number_input("Kapasitas mesin (cc)", min_value=500, max_value=5000, value=1200)
max_power = st.number_input("Tenaga maksimal (HP)", min_value=20.0, max_value=300.0, value=80.0)
seats = st.number_input("Jumlah kursi", min_value=2, max_value=10, value=5)

# Data user ke DataFrame
user_input = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner],
    'mileage(km/ltr/kg)': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats],
})

# Encode kolom kategorikal (agar sesuai dengan model)
for col in user_input.select_dtypes(include='object').columns:
    le = LabelEncoder()
    user_input[col] = le.fit_transform(user_input[col])

# Tombol prediksi
if st.button("Prediksi Harga Jual"):
    try:
        # Prediksi dalam rupee
        prediction_rupee = model.predict(user_input)[0]

        # Konversi ke rupiah (kurs kira-kira 1 INR = 185 IDR)
        kurs_inr_to_idr = 185
        prediction_rupiah = prediction_rupee * kurs_inr_to_idr

        # Tampilkan hasil
        st.success(f"💰 Prediksi harga jual mobil Anda: **Rp {prediction_rupiah:,.0f}**")
        st.caption(f"(≈ {prediction_rupee:,.0f} Rupee India, dikonversi ke Rupiah dengan kurs 1 INR = {kurs_inr_to_idr} IDR)")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
