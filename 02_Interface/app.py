import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from detection import PeopleCounter
import os

# Inisialisasi session state
if 'counter' not in st.session_state:
    st.session_state.counter = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'data' not in st.session_state:
    st.session_state.data = []
if 'prev_count_in' not in st.session_state:
    st.session_state.prev_count_in = 0
if 'prev_count_out' not in st.session_state:
    st.session_state.prev_count_out = 0

# Judul aplikasi
st.title("People Flow - Sistem Penghitung Orang")

# Sidebar untuk pengaturan
st.sidebar.header("Pengaturan")
capacity_option = st.sidebar.radio("Batas Kapasitas", ["Tidak Ada Batas Maksimal", "Ada Batas Maksimal"])
max_capacity = None
if capacity_option == "Ada Batas Maksimal":
    max_capacity = st.sidebar.number_input("Kapasitas Maksimum", min_value=1, value=10, step=1)
start_button = st.sidebar.button("Mulai")
stop_button = st.sidebar.button("Berhenti")

# Placeholder untuk video, informasi, dan warning
video_placeholder = st.empty()
info_placeholder = st.empty()
warning_placeholder = st.empty()
report_placeholder = st.empty()

# Fungsi untuk menyimpan laporan
def save_report(data):
    if not data:
        return
    if not os.path.exists("data/reports"):
        os.makedirs("data/reports")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Agregasi data per menit
    aggregated_data = {}
    for time_str, count_in, count_out in data:
        if time_str not in aggregated_data:
            aggregated_data[time_str] = {'Masuk': 0, 'Keluar': 0}
        aggregated_data[time_str]['Masuk'] += count_in
        aggregated_data[time_str]['Keluar'] += count_out
    
    # Buat DataFrame dari data yang diagregasi
    report_data = [[time_str, counts['Masuk'], counts['Keluar']] 
                   for time_str, counts in aggregated_data.items()]
    df = pd.DataFrame(report_data, columns=["Waktu (jj:mm)", "Masuk", "Keluar"])
    df.to_csv(f"data/reports/report_{timestamp}.csv", index=False)

# Logika utama
if start_button and not st.session_state.running:
    st.session_state.running = True
    st.session_state.counter = PeopleCounter(max_capacity)
    st.session_state.data = []
    st.session_state.prev_count_in = 0
    st.session_state.prev_count_out = 0

if stop_button and st.session_state.running:
    st.session_state.running = False
    if st.session_state.data:
        save_report(st.session_state.data)
    st.session_state.counter = None
    st.session_state.prev_count_in = 0
    st.session_state.prev_count_out = 0

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengakses webcam.")
            break

        frame, count_in, count_out, status, is_full = st.session_state.counter.process_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        # Update informasi
        info_text = f"Masuk: {count_in} | Keluar: {count_out} | Status: {status}"
        info_placeholder.text(info_text)

        # Tampilkan warning jika ruangan penuh
        if is_full:
            warning_placeholder.error("⚠️ **RUANGAN PENUH!** Harap tunggu hingga ada orang keluar.")
        else:
            warning_placeholder.empty()

        # Simpan data hanya jika ada perubahan
        if count_in != st.session_state.prev_count_in or count_out != st.session_state.prev_count_out:
            time_str = datetime.now().strftime("%H:%M")
            # Simpan selisih perubahan
            in_change = count_in - st.session_state.prev_count_in
            out_change = count_out - st.session_state.prev_count_out
            st.session_state.data.append([time_str, in_change, out_change])
            st.session_state.prev_count_in = count_in
            st.session_state.prev_count_out = count_out

        # Hentikan jika tombol stop ditekan
        if not st.session_state.running:
            break

    cap.release()

# Tampilkan laporan
if os.path.exists("data/reports"):
    report_files = os.listdir("data/reports")
    if report_files:
        st.header("Laporan")
        for file in report_files:
            df = pd.read_csv(f"data/reports/{file}")
            report_placeholder.subheader(f"Laporan: {file}")
            report_placeholder.dataframe(df)