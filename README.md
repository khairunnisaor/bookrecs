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
3. Atribut seperti apa yang krusial dalam menggambarkan kemiripan antar buku?

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
[2482 rows x 6 columns]
```
Diketahui bahwa terdapat 2.482 baris yang terduplikat. Hal ini akan ditangani lebih lanjut pada tahap Data Preparation.


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
Untuk mengetahui karakteristik dan distribusi data, dilakukan eksplorasi sebagai berikut:

1. Menampilkan jumlah jenis Buku, Penulis, dan Penerbit
```python
print('Banyak Buku: ', len(book_all.booktitle.unique()))
print('Banyak Penulis: ', len(book_all.bookauthor.unique()))
print('Banyak Penerbit: ', len(book_all.publisher.unique()))
```
Output:
```
Banyak Buku:  1656
Banyak Penulis:  1547
Banyak Penerbit:  716
```
Walaupun terdapat 4.898 baris pada dataset ini, hanya ada 1.656 judul buku, 1.547 penulis, dan 716 penerbit yang terlibat.

2. Visualisasi jumlah buku berdasarkan tahun penerbitan
<br>Membuat fungsi untuk menampilkan barchar berdasarkan suatu kolom pada dataset
```python
def viz_data(df_col, chart_title, x_label, y_label):
    # Hitung jumlah item per per kolom
    item_per_col = df_col.value_counts().sort_index()

    # Membuat plot
    plt.figure(figsize=(10, 4))
    sns.barplot(x=item_per_col.index, y=item_per_col.values, palette='viridis')
    plt.title(chart_title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotasi label tahun agar tidak tumpang tindih
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() # Tampilkan plot
```
Karena `publicationyear` bertipe data string, dilakukan perubahan tipe data menjadi integer. Kemudian dilakukan visualisasi dengan diagram batang untuk melihat distribusi data terhadap tahun penerbitan.
```python
book_all['publicationyear'] = book_all['publicationyear'].astype(int)
viz_data(book_all['publicationyear'], 'Book per Year', 'Publication Year', 'Number of Books')
```
Output:
![by_year](https://github.com/user-attachments/assets/2be928cc-3825-40f6-b6a3-15f2dda8cd83)

Dari visualisasi di atas, dapat dilihat bahwa kebanyakan buku pada dataset ini terbit setelah tahun 1980.

3. Visualisasi jumlah buku berdasarkan jenis Genre
```python
viz_data(book_all['genre'], 'Book per Genre', 'Genre', 'Number of Books')
```
Output:
![by_genre](https://github.com/user-attachments/assets/fc827b2a-45e4-4df2-88b3-771a24f7082d)

Sedangkan dari segi genre, Thriller dan Horor mendominasi, yang kemudian diikuti oleh Crime, Fantasy, History, dan Science Fiction.

4. Pengecekan data
```python
book_all.head()
```
Langkah terakhir pada step ini, ditampilkan 5 baris teratas dari dataset.
| Book Title        | Book Author | Publisher | Publication Year | Genre | Summary |
| ----------------- | ----------- | --------- | ---------------- | ----- | ------- |
| Jane Doe          |	R. J. Kaiser | Mira Books |	1999 |	thriller |	A double life with a single purpose: revenge... |
|	The Testament     |	John Grisham |	Dell |	1999 |	thriller |	Troy Phelan, an eccentric elderly billionaire... |
|	The Testament     |	John Grisham |	Dell |	1999 |	thriller |	The book is about Braverman Shaw, whose fathe... |
|	The Testament     |	John Grisham |	Dell |	1999 |	thriller |	In a plush Virginia office, a rich, angry old ... |
|	Seabiscuit: An American Legend	| LAURA HILLENBRAND |	Ballantine Books	| 2002	| sports |	There's an alternate cover edition here... |

Tabel di atas menunjukkan 5 baris teratas dataset. Namun, dapat dilihat bahwa terdapat salah satu baris yang memiliki booktitle, bookauthor, publisher, publication year, dan genre yang sama, namun summary yang berbeda. Berdasarkan hal ini, saya akan hanya menyimpan baris pertama dari data yang redundan seperti contoh di atas. Hal ini akan dilakukan pada tahap Data Preparation.



## Data Preparation
Setelah memahami data yang akan digunakan untuk melatih model content-based filtering, selanjutnya adalah data preparation. Pada tahap ini dilakukan penanganan nilai yang hilang (jika ada), penghapusan data yang terduplikat, dan pembersihan data. Setelah penanganan untuk menghasilkan data yang bersih dan siap digunakan ini selesai, dilanjutkan dengan tahap penggabungan fitur sesuai kebutuhan analisis. Beberapa tahapan yang dilakukan yaitu:

1. Penanganan Data Duplikat
<br>Penanganan data duplikat dilakukan berdasarkan nilai `booktitle` dan `bookauthor`. Kemunculan pertama akan disimpan dan kemunculan berikutnya dihapus.
```python
book_all = book_all.drop_duplicates(subset=['booktitle', 'bookauthor'], keep='first', inplace=False)
book_all.shape
```
Output:
```
(2416, 6)
```
Setelah penanganan, terdapat sisa 2.416 baris setelah data yang redundan dihapus.

2. Data Cleaning
<br>Untuk membersihkan fitur teks dari dataset, dilakukan perubahan teks menjadi huruf kecil agar lebih seragam dan penghapusan simbol dengan cara sebagai berikut
```python
def clean_text(df_column):
    df_column_clean = df_column.str.lower()\
                      .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)\
                      .str.replace(r'\s+', ' ', regex=True)\
                      .str.strip()
    return df_column_clean

def after_cleaning_cnt(col_name, df_before, df_after):
    print(f"before cleaning {col_name}: {len(df_before[col_name].unique())}")
    print(f"after cleaning {col_name}: {len(df_after[col_name].unique())}\n")
```
Data cleaning dilakukan pada variabel `genre`, `bookauthor`, `publisher`, dan `summary`.

Contoh data cleaning pada satu kolom:
```
cbf_data['bookauthor'] = clean_text(book_all['bookauthor'])
after_cleaning_cnt('bookauthor', book_all, cbf_data)
```
Output:
```
before cleaning genre: 10
after cleaning genre: 10

before cleaning bookauthor: 1547
after cleaning bookauthor: 1466

before cleaning publisher: 532
after cleaning publisher: 526

before cleaning summary: 1656
after cleaning summary: 1656
```
Dapat dilihat bahwa tidak ada yang berkurang dari variabel genre dan summary, tapi terdapat penurunan jumlah bookauthor dan publisher unik.


3. Combine Features
<br>Pada tahap ini, dilakukan pemilihan fitur yang akan digunakan sebagai descriptor yang menggambarkan masing-masing buku. Pada studi kasus ini, dipilih genre buku, nama penulis, dan ringkasan buku sebagai fitur. Ketiga fitur ini kemudian dikombinasikan ke dalam satu kolom.
```python
cbf_data['combined_features'] = cbf_data['genre'] + ' ' + cbf_data['bookauthor'] + ' ' + cbf_data['summary']
cbf_data
```
Contoh hasil penggabungan fitur:
| Book Title        | Book Author | Publisher | Publication Year | Genre | Summary | Combined Features |
| ----------------- | ----------- | --------- | ---------------- | ----- | ------- | ----------------- |
| Jane Doe          |	R. J. Kaiser | Mira Books |	1999 |	thriller |	A double life with a single purpose: revenge... | thriller r j kaiser a double life with a singl... |
|	The Testament     |	John Grisham |	Dell |	1999 |	thriller |	Troy Phelan, an eccentric elderly billionaire... | thriller john grisham troy phelan an eccentric... |
|	Seabiscuit: An American Legend |	Laura Hillenbrand |	Ballantine Books	| 2002	| sports |	The book is about Braverman Shaw, whose fathe... | sports laura hillenbrand theres an alternate c... |

Dapat dilihat bentuk data setelah penggabungan fitur ada pada kolom `combined_features`.



## Modeling: Content-Based Filtering
Content-Based Filtering merupakan metode sistem rekomendasi yang memberikan saran item kepada pengguna berdasarkan kesamaan atribut antara item dan preferensi pengguna di masa lalu. Sistem ini menganalisis profil "konten" dari item yang disukai pengguna (misalnya, genre, kata kunci, penulis dari buku yang pernah dibaca) dan kemudian menyarankan item baru yang memiliki karakteristik atau atribut serupa. Kelebihan utamanya adalah kemampuannya untuk merekomendasikan item yang belum pernah dilihat oleh banyak pengguna lain (masalah cold-start pada item), karena rekomendasi didasarkan pada analisis konten itu sendiri, bukan interaksi dari pengguna lain.

Hal yang dilakukan dalam pembangunan model adalah:
1. Ekstraksi fitur menggunakan TF-IDF
<br>Melakukan ekstraksi fitur `combined_features` yang terdiri dari `genre`, `bookauthor`, dan `summary` menggunakan TF-IDF. Metode ekstraksi fitur ini mengubah data text menjadi vektor sehingga kemiripan antar item dapat dihitung dengan kalkukasi matematika.
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()

# Melakukan perhitungan idf pada Combined Feature: Genre, Book Author, dan Summary
tf.fit(cbf_data['combined_features'])

# Mapping array dari fitur index integer ke fitur nama
tf.get_feature_names_out()

# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tf.fit_transform(cbf_data['combined_features'])
```

2. Perhitungan Cosine Similarity antar item
<br>Cosine Similarity dalam metode Content-Based Filtering adalah metrik yang digunakan untuk mengukur kemiripan antara dua vektor dalam ruang multidimensional. Dalam konteks rekomendasi buku, vektor ini merepresentasikan profil sebuah buku (misalnya, berdasarkan sinopsis, genre, atau penulis). Nilai kemiripan dihitung dari cosinus sudut antara kedua vektor, dengan perhitungan sebagai berikut:

<ul><ul>$$Similarity(A,B) = cos(θ) = A⋅B / ∣∣A∣∣⋅∣∣B∣∣$$</ul></ul>

<ul>Perhitungan ini memungkinkan sistem untuk secara efektif mengidentifikasi dan merekomendasikan buku yang kontennya paling sesuai dengan minat pengguna.</ul>

Dalam prakteknya, perhitungan tersebut dilakukan sebagai berikut:
```python
cosine_sim = cosine_similarity(tfidf_matrix)
```
Maka, telah didapatkan matrix dengan nilai *cosine similarity* untuk satu buku dengan buku lainnya berdasarkan genre, nama penulis, dan rangkumannya.


### Rekomendasi Top-N
Setelah berhasil membangun model dengan similariy antar item, selanjutnya dilakukan pengujian untuk model dapat memberikan informasi yang sesuai.
1. Menampilkan buku yang dijadikan referensi
<br>Pada tahap ini kita dapat mencari terlebih dulu apakah buku yang ingin kita jadikan acuan ada di dalam database.
```python
cbf_data[cbf_data.booktitle.eq('Quidditch Through the Ages')]
```
Output:
| Book Title	| Book Author	| Publisher	| Publication Year	| Genre |	Summary	| Combined Features |
| ---------- | ----------- | --------- | ---------------- | ----- | ------- | ----------------- |
| Quidditch Through the Ages	| j k rowling |	sagebrush education resources |	2001 |	fantasy | the most checkedout book in the hogwarts libra... |	fantasy j k rowling the most checkedout book i... |

2. Menampilkan Rekomendasi Top-5
<br>Rekomendasi yang diberikan model untuk buku Quidditch Through the Ages
```python
book_recommendations('Quidditch Through the Ages')
```
Output:
| Book Title |	Book Author |	Genre |
| ---------- | ----------- | ----- |
|	Fantastic Beasts and Where to Find Them	| J. K. Rowling |	fantasy |
|	Harry Potter and the Goblet of Fire	| J. K. Rowling |	fantasy |
|	Harry Potter and the Chamber of Secrets |	J. K. Rowling	| fantasy |
|	Harry Potter and the Philosopher's Stone	| J.K. Rowling	| fantasy |
|	Hyperion	| DAN SIMMONS |	science |

Secara intuitif, rekomendasi yang diberikan mayoritas relevan. 4 buku pertama merupakan sequel dari Harry Potter yang juga ditulis oleh J. K. Rowling dengan genre fantasy. Namun, pada hasil ke-5, model mengeluarkan Hyperion karya DAN Simmons. Jika dibandingkan dengan Quiddicth Through the Ages, buku ini tidak relevan, karena merupakan buku bergenre science fiction.


## Evaluation
Pada tahap ini dilakukan evaluasi model yang telah dibuat, untuk mengetahui seberapa relevan dan efektif model tersebut. Tahap ini bertujuan untuk memastikan sistem berhasil menyarankan item yang benar-benar cocok dengan minat atau kebutuhan pengguna.

### Metrik Evaluasi
Dalam mengevaluasi sistem rekomendasi content-based filtering, terutama yang merekomendasikan berdasarkan kesamaan fitur konten seperti genre, penulis, dan rangkuman, presisi adalah metrik kunci yang digunakan. Presisi mengukur seberapa banyak rekomendasi yang diberikan benar-benar relevan bagi pengguna.

Rumus untuk menghitung presisi sistem rekomendasi adalah:
<ul><ul>$$P = \text{Jumlah rekomendasi yang relevan} / \text{Jumlah total rekomendasi yang diberikan}$$</ul></ul>

Dari 5 buku yang direkomendasikan, hanya 4 buku yang relevan. Sehingga dalam hal ini, precission model ini adalah 80%.



### Kesimpulan
Dalam menghadapi masalah *information overload* untuk mencari buku yang mirip dengan selera pembaca, proyek ini telah berhasil membangun sebuah sistem rekomendasi buku yang efisien berdasarkan kemiripan buku satu dengan yang lainnya. Sistem rekomendasi yang dibuat menggunakan metode Content-Based Filtering dengan TF-IDF vectorizer berhasil memberikan 5 buku yang secara general paling mirip dengan buku yang dijadikan acuan. Karena perhitungan kemiripannya menggunakan TF-IDF dan cosine similarity, kemiripan antar teks sangat berpengaruh. Oleh karena itu, pemilihan genre, penulis, dan rangkuman sebagai fitur adalah yang paling sesuai, karena variabel lain seperti penerbit dan tahun terbit kurang relevan untuk menggambarkan kemiripan buku satu sama lain. 

Pembaca mungkin melewatkan buku-buku relevan yang sangat sesuai dengan selera mereka karena kurangnya sistem yang efektif untuk mempersonalisasi rekomendasi. Dengan adanya model yang memiliki informasi tentang berbagai buku dalam database, pengguna dapat mengetahui buku-buku baru yang sebelumnya tidak ada dalam radar mereka. Harapannya, sistem ini dapat mempersingkat proses pencarian buku secara manual oleh pengguna dan memberikan rekomendasi yang relevan dan personal. 
