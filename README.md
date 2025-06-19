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
Dataset ini menyediakan informasi lengkap tentang berbagai buku dan karakteristik fundamental setiap buku, mulai dari penulis, penerbit, isi ringkasnya, hingga kategori tematik dan waktu penerbitannya. Data ini sangat serbaguna dan dapat digunakan untuk berbagai keperluan. Data ini ideal untuk membangun sistem rekomendasi buku, di mana informasi konten dapat dianalisis untuk menyarankan judul yang relevan kepada pembaca berdasarkan preferensi mereka. Selain itu, dataset ini dapat dimanfaatkan untuk analisis tren literatur, seperti mengidentifikasi genre yang populer pada periode tertentu, tren penulis, atau pola penerbitan. Kemungkinan lain termasuk pengembangan alat pencarian buku yang lebih advance, klasifikasi buku, atau bahkan riset pasar untuk industri penerbitan.

Dataset ini dikumpulkan dari dua dataset, yaitu [Book Recommendation Dataset](https://www.kaggle.com/datasets/athu1105/book-genre-prediction) dan [Book Genre Prediction](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Kedua dataset tersebut digabungkan untuk mendapatkan informasi yang lebih lengkap, yang mana masing-masing terdiri dari 271.360 dan 4.657 baris.

Sumber Dataset:
* [Book Recommendation Dataset](https://www.kaggle.com/datasets/athu1105/book-genre-prediction)
* [Book Genre Prediction](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

### Variabel-variabel pada kedua dataset tersebut:
Book Recommendation Dataset
* ISBN: Kode unik buku
* Book Title: Judul buku
* Book Author: Nama penulis buku
* Publication Year: Tahun buku diterbitkan
* Publisher: Penerbit

Book Genre Prediction
* ID
* Title: Judul buku
* Genre: Genre cerita/isi dari buku
* Summary: Rangkuman singkat tentang buku tersebut

Dalam menggabungkan kedua dataset, digunakan judul buku sebagai primary dan foreign key satu sama lain, karena dataset Book Genre tidak memiliki kolom ISBN. Hal ini dilakukan dengan cara sebagai berikut:
```python
book_all = pd.merge(book[['booktitle', 'bookauthor', 'publisher', 'publicationyear']], book_genre[['title', 'genre', 'summary']],  left_on='booktitle', right_on='title', how='inner')
book_all = book_all.drop(columns='title')
```
Karena dataset Book Genre jauh lebih sedikit dari Book Recommendation, jumlah baris pada data berkurang mengikuti jumlah baris dari dataset Book Genre.

### Pengecekan Kondisi Dataset
Pada tahap ini dilakukan pemahaman tentang kelengkapan dataset secara umum. Hal ini dilakukan dengan:
```python
book_all.info()
```
Output:
```
RangeIndex: 4898 entries, 0 to 4897
Data columns (total 6 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   booktitle        4898 non-null   object
 1   bookauthor       4898 non-null   object
 2   publisher        4898 non-null   object
 3   publicationyear  4898 non-null   object
 4   genre            4898 non-null   object
 5   summary          4898 non-null   object
dtypes: object(6)
memory usage: 229.7+ KB
```
Hasil gabungan kedua dataset terdiri dari 6 kolom: Book Title, Book Author, Publisher, Publication Year, Genre, dan Summary dengan total 4.898 baris. Seluruh kolom bertipe string, dimana butuh dilakukan perubahan tipe data dari string ke integer untuk variabel Publication Year.

1. Pengecekan Nilai yang Hilang (missing values)
```python
# Cek missing value dengan fungsi isnull()
book_all.isnull().sum()
```
Tidak terdapat kolom yang memiliki missing values

2. Pengecekan Data Duplikat
```python
duplicates = book_all.duplicated(subset=['booktitle', 'bookauthor'])

print("Baris duplikat:")
print(book_all[duplicates])
```
Output:
```
Baris duplikat:
[2317 rows x 6 columns]
```
Diketahui bahwa terdapat 2.317 baris yang terduplikat. Hal ini akan ditangani lebih lanjut pada tahap Data Preparation.


3. Pengecekan Outlier
```python
def outlier_check(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers_iqr)}")

numeric_features = book_all[book_all.columns].select_dtypes(include=['number']).columns

print("Jumlah outlier yang ditemukan pada setiap kolom:")
for col in book_all[numeric_features]:
    outlier_check(book_all, col)
```
Output:
```
Jumlah outlier yang ditemukan pada setiap kolom:
publicationyear: 69
```
Dari hasil pengecekan outlier, terdapat 69 baris dari kolom `publicationyear` yang dikategorikan sebagai outlier. Namun, hal ini wajar berdasarkan visualisasi pada tahap EDA, karena memang banyak buku yang diterbitkan sejak awal tahun 1900.

### Exploratory Data Analysis



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
