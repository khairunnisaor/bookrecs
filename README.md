# Laporan Proyek Machine Learning - Oryza Khairunnisa
Rekomendasi Buku Menggunakan Metode Content-based Filtering

## Project Overview

Dalam era informasi digital saat ini, beban informasi berlebih menjadi tantangan signifikan, terutama dalam mencari buku yang relevan di tengah jutaan judul yang tersedia. Sistem rekomendasi hadir sebagai solusi krusial untuk mempersonalisasi pencarian buku, membantu pengguna menemukan bacaan yang sesuai minat mereka secara efisien. Proyek ini berfokus pada metode content-based filtering, yang bekerja dengan menganalisis atribut buku yang telah disukai pengguna (misalnya, genre, penulis, deskripsi) dan merekomendasikan buku lain dengan atribut serupa. Pendekatan ini memungkinkan rekomendasi yang sangat personal dan relevan, meningkatkan pengalaman pembaca tanpa bergantung pada data interaksi pengguna lain.

Penyelesaian proyek sistem rekomendasi buku berbasis content-based filtering ini sangat penting karena akan meningkatkan pengalaman pengguna dengan menyajikan rekomendasi yang relevan, menghemat waktu pencarian, dan berpotensi memperkenalkan pembaca pada judul baru yang sesuai preferensi mereka. Selain memberikan nilai komersial bagi platform penjualan buku melalui peningkatan engagement pengguna, proyek ini juga memiliki nilai akademis sebagai studi kasus yang baik untuk penerapan algoritma machine learning dan pemrosesan bahasa alami (NLP). Secara fundamental, ini menjadi langkah awal yang kuat untuk pengembangan sistem rekomendasi yang lebih canggih di masa depan.

Untuk mendukung proyek ini, riset dan referensi terkait mencakup dasar-dasar sistem rekomendasi seperti yang dibahas dalam Ricci et al., Recommender Systems Handbook (2011) yang menjelaskan berbagai pendekatan termasuk content-based filtering. Studi oleh Mathew et al., 2024 secara spesifik mengulas implementasi sistem rekomendasi buku berbasis konten, seringkali memanfaatkan teknik seperti TF-IDF untuk ekstraksi fitur dari teks dan Cosine Similarity untuk mengukur kemiripan antar buku. Dataset umum seperti Goodreads Dataset juga menjadi sumber daya penting dalam pengembangan dan pengujian sistem semacam ini.

Referensi: P. Mathew, B. Kuriakose and V. Hegde, "Book Recommendation System through content based and collaborative filtering method," 2016 International Conference on Data Mining and Advanced Computing (SAPIENCE), Ernakulam, India, 2016, pp. 47-52.

## Business Understanding

### Problem Statements
1. Bagaimana cara menghadapi masalah *information overload* dalam menentukan atau mencari buku yang mungkin cocok dengan selera pembaca?
2. Sistem seperti apa yang dapat diimplementasikan untuk mengefisiensi pencarian buku yang biasanya dilakukan secara manual dengan membaca sinopsis?

### Goals
1. Merancang sebuah sistem yang efisien, efektif, dan dapat memberikan saran kepada pembaca berdasarkan informasi yang terkandung di dalam masing-masing buku sehingga dapat mengurangi waktu dan upaya yang diperlukan dalam proses pencarian manual.
2. Merancang dan membangun sebuah sistem rekomendasi buku menggunakan metode content-based filtering untuk mempersonalisasi saran buku bagi pengguna.
3. Memastikan bahwa buku yang direkomendasikan memiliki kemiripan atribut (misalnya, genre, penulis, sinopsis) yang tinggi dengan buku yang sebelumnya disukai pengguna.


## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

### Kesimpulan
Kelebihan Informasi Buku: Pengguna menghadapi kesulitan signifikan dalam menemukan buku yang sesuai dengan minat dan preferensi mereka di tengah jutaan judul yang tersedia di berbagai platform, menyebabkan beban informasi berlebih (information overload).
Pencarian Manual yang Tidak Efisien: Proses pencarian buku secara manual oleh pengguna seringkali memakan waktu, tidak efisien, dan cenderung tidak menghasilkan rekomendasi yang relevan atau personal.
Potensi Minat Terlewat: Pembaca mungkin melewatkan buku-buku relevan yang sangat sesuai dengan selera mereka karena kurangnya sistem yang efektif untuk mempersonalisasi rekomendasi.


**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
