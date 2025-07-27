import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import holidays

st.title("Prediksi Jumlah Transaksi dengan Prophet")

# === Load dan siapkan data ===
df = pd.read_csv('transaksi.csv')
df['tanggal'] = pd.to_datetime(df['tanggal'])
df = df.rename(columns={'tanggal': 'ds', 'jumlah_transaksi': 'y'})
df['y'] = df['y'].str.replace(',', '').astype(float)

df['day_of_week'] = df['ds'].dt.day_name()
df['is_weekend'] = df['ds'].dt.weekday >= 5

tahun_list = df['ds'].dt.year.unique()
indo_holidays = holidays.Indonesia(years=tahun_list)

# Buat kolom nama_libur, isi nama hari libur jika tanggal termasuk hari libur, else kosong string
df['nama_libur'] = df['ds'].dt.date.map(lambda d: indo_holidays.get(d, ""))

df['is_libur'] = df['nama_libur'] != ""

# Tampilkan data awal dengan nama_libur
st.subheader("Data Transaksi")
st.dataframe(df, height=300)

model = Prophet()
model.add_regressor('is_weekend')
model.add_regressor('is_libur')

model.fit(df)

future = model.make_future_dataframe(periods=7)
future['day_of_week'] = future['ds'].dt.day_name()
future['is_weekend'] = future['ds'].dt.weekday >= 5
future['nama_libur'] = future['ds'].dt.date.map(lambda d: indo_holidays.get(d, ""))
future['is_libur'] = future['nama_libur'] != ""

forecast = model.predict(future)

# Plot hasil prediksi dan tampilkan di streamlit
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Tampilkan prediksi 7 hari ke depan
predicted = forecast[['ds', 'yhat']].tail(7).copy()
predicted['pct_change'] = predicted['yhat'].pct_change() * 100
predicted['status'] = predicted['pct_change'].apply(
    lambda x: f"Naik {x:.2f}%" if x > 0 else f"Turun {abs(x):.2f}%" if x < 0 else "Stabil"
)

# Tambahkan nama_libur di predicted dari future agar muncul di tabel Streamlit
predicted = predicted.merge(future[['ds', 'nama_libur']], on='ds', how='left')

st.subheader("Prediksi 7 Hari ke Depan")
st.dataframe(predicted[['ds', 'yhat', 'nama_libur', 'status']])

# Deteksi anomali
df_merged = df.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
df_merged['anomaly'] = (df_merged['y'] > df_merged['yhat_upper']) | (df_merged['y'] < df_merged['yhat_lower'])
anomali = df_merged[df_merged['anomaly'] == True]

st.subheader("Hari Anomali")
st.dataframe(anomali[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']])

# Statistik hari libur/weekend
st.subheader("Rata-rata Transaksi")
st.write(f"Libur: {df[df['is_libur']]['y'].mean():.2f}")
st.write(f"Weekend: {df[df['is_weekend']]['y'].mean():.2f}")
st.write(f"Weekday: {df[~df['is_weekend']]['y'].mean():.2f}")
