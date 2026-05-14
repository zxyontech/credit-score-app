import streamlit as st
import pickle
import pandas as pd

# Load model dan scaler
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Konfigurasi halaman
st.set_page_config(
    page_title="Credit Score Prediction",
    page_icon="💳",
    layout="centered"
)

# Header
st.title("💳 Credit Score Prediction")
st.markdown("### Prediksi kelayakan kredit berdasarkan profil keuangan Anda")
st.divider()

# Input Form
st.subheader("📋 Masukkan Data Keuangan")

col1, col2 = st.columns(2)

with col1:
    outstanding_debt = st.number_input("Hutang Tersisa ($)", min_value=0, value=500)
    interest_rate = st.number_input("Suku Bunga (%)", min_value=0, max_value=50, value=10)
    delay_from_due_date = st.number_input("Keterlambatan Bayar (hari)", min_value=0, max_value=100, value=5)
    changed_credit_limit = st.number_input("Perubahan Limit Kredit", min_value=0, max_value=50, value=5)
    credit_mix = st.selectbox("Credit Mix", options=[0, 1, 2, 3],
                               format_func=lambda x: ['Unknown', 'Bad', 'Standard', 'Good'][x])

with col2:
    num_credit_inquiries = st.number_input("Jumlah Pengecekan Kredit", min_value=0, max_value=20, value=3)
    num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, max_value=15, value=2)
    annual_income = st.number_input("Pendapatan Tahunan ($)", min_value=0, value=30000)
    total_emi = st.number_input("Total Cicilan per Bulan ($)", min_value=0, value=50)
    credit_utilization = st.number_input("Rasio Penggunaan Kredit (%)", min_value=0, max_value=100, value=30)

st.divider()

# Tombol Prediksi
if st.button("🔍 Prediksi Credit Score", use_container_width=True):
    # Susun input data sesuai urutan 10 fitur
    input_data = pd.DataFrame([[
        outstanding_debt, interest_rate, delay_from_due_date,
        changed_credit_limit, credit_mix, num_credit_inquiries,
        num_credit_card, annual_income, total_emi, credit_utilization
    ]], columns=[
        'Outstanding_Debt', 'Interest_Rate', 'Delay_from_due_date',
        'Changed_Credit_Limit', 'Credit_Mix', 'Num_Credit_Inquiries',
        'Num_Credit_Card', 'Annual_Income', 'Total_EMI_per_month',
        'Credit_Utilization_Ratio'
    ])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # Tampilkan hasil
    st.subheader("🎯 Hasil Prediksi")

    if prediction == 'Good':
        st.success("## ✅ Credit Score: GOOD")
        st.markdown("**Keuangan Anda sehat dan terpercaya!** Kemungkinan besar disetujui untuk pinjaman.")
    elif prediction == 'Standard':
        st.warning("## ⚠️ Credit Score: STANDARD")
        st.markdown("**Keuangan Anda cukup baik.** Pinjaman mungkin disetujui dengan syarat tertentu.")
    else:
        st.error("## ❌ Credit Score: POOR")
        st.markdown("**Keuangan Anda berisiko.** Disarankan untuk memperbaiki kebiasaan keuangan.")

    # Probabilitas
    st.divider()
    st.subheader("📊 Probabilitas per Kelas")
    classes = model.classes_
    prob_df = pd.DataFrame({
        'Kelas': classes,
        'Probabilitas': [f"{p:.2%}" for p in probability]
    })
    st.dataframe(prob_df, use_container_width=True)