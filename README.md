# Proyek Analisis Data 📝  

Ini adalah proyek **Analisis Data** menggunakan **Python, Pandas, Matplotlib, Seaborn, dan Streamlit**.  
Dashboard ini dibuat untuk menampilkan **insight dari dataset e-commerce**.

---

## 📂 Struktur Folder  
📦 Proyek-Analisis-Data
┣ 📂 dashboard/ # Folder utama untuk Streamlit
┃ ┣ 📜 dashboard.py # File utama aplikasi Streamlit
┣ 📂 data/ # Folder penyimpanan dataset
┃ ┣ 📜 customers_dataset.csv # Dataset pelanggan
┃ ┣ 📜 order_items_dataset.csv # Dataset barang
┃ ┣ 📜 order_payments_dataset.csv # Dataset pembayaran
┃ ┣ 📜 order_reviews_dataset.csv # Dataset review
┃ ┣ 📜 orders_dataset.csv # Dataset orderan
┃ ┣ 📜 product_category_name_translation_dataset.csv # Dataset nama kategori produk
┃ ┣ 📜 products_dataset.csv # Dataset produk
┃ ┣ 📜 sellers_dataset.csv # Dataset seller
┣ 📜 notebook.ipynb # Notebook proyek
┣ 📜 README.md # Dokumentasi proyek
┣ 📜 requirements.txt # Daftar library yang digunakan
┣ 📜 url.txt # Link Deploy

---

## 🚀 Cara Menjalankan Proyek  
1. **Clone repository ini ke lokal**  
   ```bash
   git clone https://github.com/garie0/Proyek-Analisis-Data.git
   cd Proyek-Analisis-Data

2. **Buat virtual environment & install dependensi**
    ```bash
    python -m venv venv  
    source venv/bin/activate  # Mac/Linux  
    venv\Scripts\activate  # Windows  
    pip install -r requirements.txt  

3. **Jalankan dashboard Streamlit**
    ```bash
    streamlit run dashboard.py

## 📊 Fitur Dashboard
✅ Analisis RFM - Mengelompokkan pelanggan berdasarkan Recency, Frequency, Monetary.
✅ Geospatial Analysis - Menganalisis lokasi pelanggan dengan visualisasi peta interaktif.
✅ Visualisasi Data - Menampilkan tren pesanan, metode pembayaran, dan rating ulasan.

## 📥 Download Dataset
Karena file main_data.csv dan geolocation_dataset.csv terlalu besar untuk GitHub,
silakan unduh dari Google Drive berikut:
Download main_data.csv
Download geolocation_dataset.csv
https://drive.google.com/drive/folders/1udnMk_qYvzO3oPXnZxRvpwzx002ai2Vs?usp=sharing

## 🛠 Teknologi yang Digunakan
Python 🐍
Pandas 📊
Matplotlib & Seaborn 📈
Streamlit 🎛️
Folium 🗺️

## 📌 Kontributor
👤 Gigih Agung Prasetyo
📧 Email: gigihagung0@gmail.com
🔗 GitHub: garie0

## 🌟 License
Proyek ini berlisensi MIT License – bebas digunakan, dimodifikasi, dan disebarluaskan.