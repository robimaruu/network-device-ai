import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
from teachablemachine import TeachableMachine

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="AI Network Device Recognition", page_icon="🌐", layout="centered")

st.title("🌐 AI Network Device Recognition")
st.write("Arahkan kamera HP atau unggah foto untuk mengenali jenis dan fungsi alat jaringan komputer.")

# 2. Dictionary Deskripsi Perangkat Jaringan
DEVICE_INFO = {
    "Router": {
        "fungsi": "Menghubungkan dua atau lebih jaringan yang berbeda (misalnya menghubungkan jaringan LAN rumah ke jaringan Internet).",
        "ciri": "Memiliki port WAN/LAN dan sering dilengkapi antena Wi-Fi.",
        "tips": "Pastikan konfigurasi IP Address dan Gateway sudah benar saat instalasi."
    },
    "Switch": {
        "fungsi": "Menghubungkan banyak komputer/perangkat dalam satu jaringan lokal (LAN) yang sama.",
        "ciri": "Memiliki deretan port RJ45 yang banyak (8, 16, 24, atau 48 port).",
        "tips": "Gunakan kabel UTP Straight-Through untuk menghubungkan PC ke Switch."
    },
    "Access Point": {
        "fungsi": "Memancarkan sinyal nirkabel (Wi-Fi) agar perangkat seperti HP dan Laptop bisa terhubung ke jaringan kabel.",
        "ciri": "Bentuk fisik cenderung pipih/bulat dan biasa dipasang di langit-langit atau dinding.",
        "tips": "Atur SSID dan password WPA2/WPA3 untuk keamanan Wi-Fi."
    },
    "Konektor RJ45": {
        "fungsi": "Konektor fisik yang dipasang di ujung kabel UTP untuk dicolokkan ke port LAN komputer atau router.",
        "ciri": "Terbuat dari plastik bening dengan 8 pin tembaga di dalamnya.",
        "tips": "Gunakan tang crimping dan pastikan urutan warna kabel (T568A/T568B) sudah pas."
    }
}

# 3. Loader Model Menggunakan TeachableMachine Wrapper
@st.cache_resource
def load_ai_model():
    model_path = 'keras_model.h5'
    labels_path = 'labels.txt'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File '{model_path}' tidak ditemukan di lokasi repositori.")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"File '{labels_path}' tidak ditemukan di lokasi repositori.")

    # Inisialisasi Teachable Machine Classifier
    model = TeachableMachine(model_path=model_path, labels_file_path=labels_path)
    return model

try:
    model = load_ai_model()
    
    # 4. Input Gambar (Camera/Upload)
    st.subheader("📷 Ambil Gambar / Unggah File")
    camera_img = st.camera_input("Ambil Foto Langsung dari Kamera HP")
    file_img = st.file_uploader("Atau Upload Foto dari Galeri", type=["jpg", "png", "jpeg"])

    target_image = camera_img if camera_img is not None else file_img

    if target_image is not None:
        # Simpan sementara gambar untuk diklasifikasikan
        image = Image.open(target_image).convert('RGB')
        st.image(image, caption='Gambar Perangkat yang Diuji', use_container_width=True)
        
        temp_img_path = "temp_input.jpg"
        image.save(temp_img_path)

        # Prediksi AI via Teachable Machine
        with st.spinner('Menganalisis perangkat...'):
            result = model.classify_image(temp_img_path)
            
            raw_class_name = result['class_name']
            # Ambil label murni (menghilangkan angka indeks seperti "0 Router" -> "Router")
            detected_label = raw_class_name.split(' ', 1)[-1] if ' ' in raw_class_name else raw_class_name
            confidence_score = float(result['class_confidence']) * 100

        # Hapus file sementara
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        # Tampilkan Hasil Analisis
        st.divider()
        st.success(f"### 🚀 Hasil Identifikasi: **{detected_label}**")
        st.info(f"**Tingkat Akurasi AI (Confidence):** {confidence_score:.2f}%")

        # Tampilkan Detail Fungsi Alat jika terdaftar
        if detected_label in DEVICE_INFO:
            info = DEVICE_INFO[detected_label]
            st.markdown(f"**💡 Fungsi Utama:**\n{info['fungsi']}")
            st.markdown(f"**🔍 Ciri Fisik:**\n{info['ciri']}")
            st.markdown(f"**🛠️ Tips Penggunaan:**\n{info['tips']}")
        else:
            st.warning(f"Penjelasan untuk '{detected_label}' belum dikonfigurasi di DEVICE_INFO.")

except Exception as e:
    st.error(f"⚠️ Detail Pesan Kesalahan: {e}")
