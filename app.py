import streamlit as st
import pickle
import numpy as np
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
    age = st.number_input("Usia", min_value=18, max_value=100, value=25)
    annual_income = st.number_input("Pendapatan Tahunan ($)", min_value=0, value=30000)
    monthly_salary = st.number_input("Gaji Bulanan ($)", min_value=0, value=2500)
    num_bank_accounts = st.number_input("Jumlah Rekening Bank", min_value=0, max_value=20, value=2)
    num_credit_card = st.number_input("Jumlah Kartu Kredit", min_value=0, max_value=15, value=2)
    interest_rate = st.number_input("Suku Bunga (%)", min_value=0, max_value=50, value=10)
    num_of_loan = st.number_input("Jumlah Pinjaman", min_value=0, max_value=20, value=2)
    delay_from_due_date = st.number_input("Rata-rata Keterlambatan Bayar (hari)", min_value=0, max_value=100, value=5)
    num_delayed_payment = st.number_input("Jumlah Pembayaran Terlambat", min_value=0, max_value=50, value=3)
    changed_credit_limit = st.number_input("Perubahan Limit Kredit", min_value=0, max_value=50, value=5)

with col2:
    num_credit_inquiries = st.number_input("Jumlah Pengecekan Kredit", min_value=0, max_value=20, value=3)
    credit_mix = st.selectbox("Credit Mix", options=[0, 1, 2, 3], 
                               format_func=lambda x: ['Unknown', 'Bad', 'Standard', 'Good'][x])
    outstanding_debt = st.number_input("Hutang Tersisa ($)", min_value=0, value=500)
    credit_utilization = st.number_input("Rasio Penggunaan Kredit (%)", min_value=0, max_value=100, value=30)
    payment_min = st.selectbox("Bayar Minimum?", options=[0, 1],
                                format_func=lambda x: ['Tidak', 'Ya'][x])
    total_emi = st.number_input("Total Cicilan per Bulan ($)", min_value=0, value=50)
    amount_invested = st.number_input("Investasi per Bulan ($)", min_value=0, value=100)
    payment_behaviour = st.selectbox("Perilaku Pembayaran", options=[0, 1, 2, 3, 4, 5, 6],
                                      format_func=lambda x: [
                                          'High spent Large value',
                                          'High spent Medium value', 
                                          'High spent Small value',
                                          'Low spent Large value',
                                          'Low spent Medium value',
                                          'Low spent Small value',
                                          'Unknown'][x])
    monthly_balance = st.number_input("Saldo Bulanan ($)", min_value=0, value=300)
    occupation = st.selectbox("Pekerjaan", options=list(range(0, 16)),
                               format_func=lambda x: [
                                   'Accountant', 'Architect', 'Developer',
                                   'Doctor', 'Engineer', 'Entrepreneur',
                                   'Journalist', 'Lawyer', 'Manager',
                                   'Mechanic', 'Media Manager', 'Musician',
                                   'Scientist', 'Teacher', 'Writer', 'Other'][x])

st.divider()

# Tombol Prediksi
if st.button("🔍 Prediksi Credit Score", use_container_width=True):
    # Susun input data
    input_data = pd.DataFrame([[
        age, occupation, annual_income, monthly_salary,
        num_bank_accounts, num_credit_card, interest_rate,
        num_of_loan, delay_from_due_date, num_delayed_payment,
        changed_credit_limit, num_credit_inquiries, credit_mix,
        outstanding_debt, credit_utilization, payment_min,
        total_emi, amount_invested, payment_behaviour, monthly_balance
    ]], columns=[
        'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
        'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Payment_of_Min_Amount',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Payment_Behaviour',
        'Monthly_Balance'
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