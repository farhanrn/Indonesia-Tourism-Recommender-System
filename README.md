# Laporan Peoyek Akhir Machine Learning - Farhan Rahman
Parawisata - Rekomendasi Wisata Indonesia

## Domain Proyek
Industri pariwisata telah muncul sebagai penggerak utama pertumbuhan ekonomi di berbagai daerah. Organisasi Pariwisata Dunia (UNWTO) melaporkan bahwa kedatangan wisatawan internasional mencapai 1,5 miliar pada tahun 2019, dengan peningkatan 4% dibandingkan tahun sebelumnya [[1]](https://www.e-unwto.org/doi/abs/10.18111/wtobarometereng.2020.18.1.2). Tren pertumbuhan ini diperkirakan akan berlanjut, didorong oleh penemuan destinasi wisata baru dan dinamika yang berkembang dalam pasar pariwisata inbound dan outbound, terutama di negara-negara seperti Tiongkok dan Thailand [[2]](https://www.tandfonline.com/doi/full/10.1080/10941665.2020.1745855). Seiring dengan persiapan industri pariwisata untuk bangkit kembali, sangat penting untuk mengembangkan strategi dan kegiatan yang efektif guna memastikan pelayanan berkualitas bagi para wisatawan dan semua pemangku kepentingan yang terlibat. Persiapan ini harus melibatkan partisipasi semua pemangku kepentingan, dengan mengakui peran spesifik mereka [[3]](https://pwk.teknik.untan.ac.id/files/buku/fullbook-perencanaan-destinasi-pariwisata-compressed-compressed_1706694217.pdf). 

Indonesia adalah negara yang memiliki wilayah yang luas serta keberanekaragaman sumber daya alam dan kebudayaan serta adat istiadatnya yang beragam merupakan potensi yang dimiliki oleh Indonesia untuk dijadikan modal dalam pengembangan sektor pariwisatanya. Potensi-potensi tersebut memiliki kelayakan untuk dikelola dengan maksimal untuk membangun sektor pariwisata di Indonesia [[4]](https://jurnal.stie-aas.ac.id/index.php/jie/article/viewFile/13101/pdf). Banyaknya jumlah destinasi wisata membuat orang bingung dalam memilih destinasi wisata yang sesuai. Sistem rekomendasi adalah cara yang tepat untuk membantu masyarakat Indonesia memilih destinasi wisata yang sesuai dengan preferensi mereka [[5]](https://jurnal-itsi.org/index.php/jitsi/article/view/254/120).

## Business Understanding
### Problem Statement
- Bagaimana cara membuat sistem rekomendasi berdasarkan Kategori Wisata?
- Bagaimana cara membuat sistem rekomendasi wista berdasarkan rating yang diberi user sebelumnya?

### Goals
- Membangun sistem rekomendasi berdasarkan Kategori Wisata 
- Membangun cara membuat sistem rekomendasi wista berdasarkan rating yang diberi user sebelumnya

### Solution Statement
Proyek ini bertujuan untuk menyelesaikan masalah dengan mengembangkan model machine learning yang mampu merekomendasikan destinasi wisata kepada pengguna. Rekomendasi ini didasarkan pada penilaian atau rating pengguna terhadap tempat wisata tertentu. Dalam pembuatan model, digunakan dua pendekatan utama, yaitu content-based filtering recommendation dan collaborative filtering recommendation.

## Data Understanding
Dataset yang digunakan pada proyek ini adalah  `Indonesia Tourism Destination` yang diperoleh dari Kaggle. [Tautan pada Dataset dapat diakses pada tautan ini](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)

Struktur pada dataset adalah sebagai berikut

![Data](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/Data%20Explorer.png)

### Place Tourism
Berikut deskripsi dataset place tourism (`tourism_with_id.csv`)
| Nama Kolom | Tipe Data | Deskripsi |
|---|---|---|
| Place_Id | int64 | ID unik untuk setiap tempat |
| Place_Name | object | Nama tempat wisata |
| Description | object | Deskripsi tempat |
| Category | object | Kategori tempat (misalnya, Alam, Budaya) |
| City | object | Kota tempat wisata berada |
| Price | float64 | Harga atau biaya masuk |
| Rating | float64 | Rating rata-rata tempat |
| Time_Minutes | float64 | Estimasi waktu kunjungan |
| Coordinate | object | Koordinat latitude dan longitude |
| Lat | float64 | Koordinat latitude |
| Long | float64 | Koordinat longitude |

Dari hasil assessing data pada dataset place_tourism diperoleh bahwa :

```
- Tidak ada data duplikasi pada dataset
- Terdapat missing value pada kolom Time Minutes sebanyak 232 dan Unnamed :11 sebanyak 437. sehingga perlu dilakukan cleaning
```

### Rating

Berikut deskripsi dataset rating (`tourism_rating.csv`)

| **Kolom**       | **Deskripsi**                                                                 | **Tipe Data**  |
|------------------|------------------------------------------------------------------------------|---------------|
| `User_Id`        | ID unik yang mengidentifikasi setiap pengguna dalam sistem.                 | Integer       |
| `Place_Id`       | ID unik yang mengacu pada tempat wisata tertentu.                           | Integer       |
| `Place_Ratings`  | Penilaian pengguna terhadap tempat wisata, dinyatakan dalam skala tertentu. | Integer       |

```
Bentuk Data : (10000, 3)
==========
Informasi pada setiap Kolom pada dataset place_toursim :
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 3 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   User_Id        10000 non-null  int64
 1   Place_Id       10000 non-null  int64
 2   Place_Ratings  10000 non-null  int64
dtypes: int64(3)
memory usage: 234.5 KB
==========
Jumlah Duplikat Data : 0
Jumlah Null Data : 0
```

### User

Berikut deskripsi data `user.csv`


| **Kolom**    | **Deskripsi**                                                       | **Tipe Data** |
|--------------|---------------------------------------------------------------------|---------------|
| `User_Id`    | ID unik yang mengidentifikasi setiap pengguna dalam sistem.         | Integer       |
| `Location`   | Lokasi pengguna yang mencakup kota dan provinsi.                    | String        |
| `Age`        | Usia pengguna, dinyatakan dalam tahun.                              | Integer       |


## Data Preprocessing
### Handling Missing Value in Dataset Place Tourism

Dataset sebelum dilakukan handling

![](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/before_preprocess.png)

Dataset setelah dilakukan handling

![](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/Screenshot%202024-11-29%20014137.png)

### Menghilangkan Kolom yang Tidak Diinginkan

Karena kolom `unnamed:11` dan `Unnamed: 12` tidak diinginkan, sehingga dikeluarkan saja

``place_tourism.drop(['Unnamed: 11', 'Unnamed: 12'], axis=1, inplace=True)``


## Exploratory Data Analysis
### Univariate EDA on *place_tourism* dataset
- Place 

```
print("Total lokasi wisata :",place_tourism['Place_Name'].nunique())
output :
Total lokasi wisata : 437
```
Terdapat 437 Data Lokasi Wisata 

- Category

Ada berapa lokasi wisata dalam dataset?
![](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/EDA-CATEGORY.png)

Pada distribusi Data Kategori tempat Wisata, Tempat Hiburan menempati posisi teratassebanyak 135 kemudian disusul oleh Budaya sebesar 117. Adapun distribusi pada Kategori Pusat Perbelanjaan adalah sebanyak 15 yang menempati posisi terakhir

- City

Ada berapa lokasi kota wisata dalam dataset?

![](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/EDA-CITY.png)

Yogyakarta memiliki jumlah terbanyak sebesar 126, selisih 2 dengan Bandung yang sebesar 124. Kemudian Surabaya menjadi yang terakhir sebesar 46 saja

- Pesebaran harga

![](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/EDA-PRICE.png)

Sebagian besar tempat wisata memiliki harga yang rendah atau gratis. Hal ini terlihat dari tingginya frekuensi pada rentang harga 0 hingga sekitar 25000.

### Univariate EDA on *rating* dataset

![](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/EDA-RATING-KDE.png)

Berdasarkan grafik distribusi rating tempat wisata dengan KDE, terlihat bahwa sebagian besar tempat wisata mendapatkan rating di rentang 4.0 hingga 4.5. Hal ini menunjukkan bahwa mayoritas tempat wisata di dataset tersebut memiliki kualitas yang baik dan memuaskan bagi pengunjung. Puncak distribusi KDE berada di sekitar rating 4.2, yang mengindikasikan bahwa rating tersebut merupakan rating yang paling umum diberikan oleh pengguna. Selain itu, distribusi rating cenderung condong ke kanan (right-skewed), yang berarti terdapat beberapa tempat wisata dengan rating yang sangat tinggi, meskipun jumlahnya relatif sedikit. Secara keseluruhan, distribusi rating tempat wisata menunjukkan bahwa mayoritas tempat wisata memiliki kualitas yang baik dan memuaskan bagi pengunjung, dengan beberapa tempat wisata yang memiliki rating sangat tinggi.

### Univariate EDA on *user* dataset

- Location

![](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/EDA-USER-LOCATION.png)

Berdasarkan visualisasi data, terlihat bahwa lokasi wisata terbanyak berada di Jakarta, diikuti oleh Bogor dan Bandung. Jumlah lokasi wisata di Jakarta jauh lebih banyak dibandingkan dengan kota-kota lainnya dalam dataset. Sebaliknya, lokasi wisata paling sedikit berada di Yogyakarta, Surabaya, dan Malang. Hal ini mengindikasikan bahwa Jakarta merupakan pusat wisata yang populer, sementara kota-kota seperti Yogyakarta, Surabaya, dan Malang mungkin memiliki daya tarik wisata yang lebih sedikit atau kurang dikenal oleh pengguna dalam dataset.

- Age

![](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/EDA-USER-AGE.png)

Berdasarkan analisis distribusi usia pengunjung, diketahui bahwa mayoritas pengunjung berusia antara 20 hingga 30 tahun. Distribusi data usia cenderung terdistribusi normal dengan sedikit kemiringan ke kanan (right-skewed), mengindikasikan adanya beberapa pengunjung dengan usia yang lebih tua. Meskipun terdapat outlier pada data usia, namun secara umum dapat disimpulkan bahwa tempat wisata tersebut lebih banyak dikunjungi oleh kalangan muda. Hal ini dapat menjadi pertimbangan dalam strategi pemasaran dan pengembangan produk wisata yang disesuaikan dengan preferensi dan kebutuhan pengunjung dari rentang usia tersebut.

## Data Preprocessing : Content-Based Filtering
Pada tahap ini, dataset place_tourism dan rating akan digabungkan untuk pemodelan content-based filtering. sehingga hanya beberapa kolom saja yang akan digunakan dalam dataset

### Feature Selection for Place_tourism

```
place_tourism.drop(['Rating','Time_Minutes','Coordinate','Lat','Long'],axis=1,inplace=True)
```
Pada dataset ini hanya akan menggunakan Place_Id, Place_Name, Description, Category, City, dan Price 

### Menggabungkan Data Place_Tourism dengan Rating

```
merged_data = pd.merge(rating.groupby('Place_Id')['Place_Ratings'].mean(),
                       place_tourism,
                       on='Place_Id')
```
### Text Processing

Tujuan utama dari text processing adalah untuk mempersiapkan data teks agar dapat diproses oleh model machine learning. Text processing pada kode ini bertujuan untuk membersihkan dan mentransformasi data teks pada kolom 'Description' dan 'Category' agar lebih terstruktur dan informatif untuk proses content-based filtering.

```
stem = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocessing(data):
    data = data.lower()
    data = stem.stem(data)
    data = stopword.remove(data)
    return data
# Menyatukan Deskripsi dan Kategori ke dalam kolom Pattern
data_content_based['Pattern'] = data_content_based['Description'] + ' ' + data_content_based['Category']
# Menghilangkan kolom Price, Place Rating, Deskripsi, dan City
data_content_based.drop(['Price','Place_Ratings','Description','City'],axis=1,inplace=True)

# Menerapkan Function
data_content_based['Pattern'] = data_content_based['Pattern'].apply(preprocessing)

```

- Case Folding: data = data.lower(): Mengubah semua teks menjadi huruf kecil untuk penyeragaman.
- Stemming: data = stem.stem(data): Memotong imbuhan kata menjadi kata dasar menggunakan library Sastrawi. Contoh: "bermain" menjadi "main".
- Stop Word Removal: data = stopword.remove(data): Menghilangkan kata-kata umum (stop words) seperti "yang", "dan", "di", dll., yang dianggap tidak memiliki makna penting dalam analisis teks.
- Kode ini menerapkan fungsi preprocessing yang telah didefinisikan sebelumnya pada setiap baris data di kolom 'Pattern'.
- Hasilnya, data teks di kolom 'Pattern' telah dibersihkan dan diubah menjadi bentuk yang lebih sederhana dan terstruktur, siap untuk digunakan dalam pemodelan content-based filtering.


## Modeling Content-Based Filtering

Content-based filtering adalah pendekatan dalam sistem rekomendasi yang memberikan rekomendasi kepada pengguna berdasarkan karakteristik atau atribut dari item yang disukai atau digunakan oleh pengguna tersebut sebelumnya. Model ini berfokus pada analisis konten dari item dan mencari item serupa untuk direkomendasikan.

Beberapa langkah yang dilakukan dalam membangun sistem rekomendasi menggunakan pendekatan content-based filtering meliputi penggunaan TF-IDF Vectorizer, perhitungan kesamaan menggunakan cosine similarity, serta pengujian sistem rekomendasi.

### TF-IDF
TF-IDF adalah teknik dalam *Natural Language Processing (NLP)* yang digunakan untuk merepresentasikan teks dalam bentuk numerik. Teknik ini bertujuan untuk menilai seberapa penting suatu kata dalam sebuah dokumen relatif terhadap kumpulan dokumen (corpus). TF-IDF sering digunakan dalam sistem rekomendasi berbasis konten dan tugas seperti pencarian teks.

TF-IDF terdiri dari dua komponen utama:
1. **Term Frequency (TF)**: Mengukur seberapa sering sebuah kata muncul dalam dokumen tertentu.
2. **Inverse Document Frequency (IDF)**: Mengukur pentingnya kata tersebut di seluruh dokumen dalam corpus.

### Cosine Similarity

Cosine similarity digunakan untuk mengukur sejauh mana dua data tempat (place) memiliki kesamaan dengan menghitung sudut antara keduanya. Teknik ini menilai tingkat kesamaan berdasarkan sudut antara data tempat yang sedang dianalisis. Hasil perhitungan ini akan menghasilkan nilai yang mencerminkan tingkat kesamaan antara dua data tempat, di mana nilai yang mendekati 1 menunjukkan kesamaan yang tinggi, dan nilai yang mendekati 0 menunjukkan kesamaan yang rendah.

## Modeling 
