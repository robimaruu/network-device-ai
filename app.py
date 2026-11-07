import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os

# Import TFLite interpreter
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

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

# 3. Loader Model TFLite
@st.cache_resource
def load_tflite_model():
    model_path = 'model.tflite'
    labels_path = 'labels.txt'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File '{model_path}' tidak ditemukan di lokasi repositori.")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"File '{labels_path}' tidak ditemukan di lokasi repositori.")

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    with open(labels_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]

    return interpreter, class_names

try:
    interpreter, class_names = load_tflite_model()
    
    # 4. Input Gambar (Camera/Upload)
    st.subheader("📷 Ambil Gambar / Unggah File")
    camera_img = st.camera_input("Ambil Foto Langsung dari Kamera HP")
    file_img = st.file_uploader("Atau Upload Foto dari Galeri", type=["jpg", "png", "jpeg"])

    target_image = camera_img if camera_img is not None else file_img

    if target_image is not None:
        image = Image.open(target_image).convert('RGB')
        st.image(image, caption='Gambar Perangkat yang Diuji', use_container_width=True)

        # Preprocessing Gambar untuk TFLite (224x224, Normalisasi -1 s.d. 1)
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1.0
        input_data = np.expand_dims(normalized_image_array, axis=0)

        # Prediksi AI Menggunakan TFLite
        with st.spinner('Menganalisis perangkat...'):
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            index = np.argmax(output_data[0])
            raw_class_name = class_names[index]
            detected_label = raw_class_name.split(' ', 1)[-1] if ' ' in raw_class_name else raw_class_name
            confidence_score = float(output_data[0][index]) * 100

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

except Exception as e:
    st.error(f"⚠️ Detail Pesan Kesalahan: {e}")
