# Laporan Peoyek Akhir Machine Learning - Farhan Rahman
Parawisata - Rekomendasi Wisata Indonesia

## Domain Proyek
Industri pariwisata telah muncul sebagai penggerak utama pertumbuhan ekonomi di berbagai daerah. Organisasi Pariwisata Dunia (UNWTO) melaporkan bahwa kedatangan wisatawan internasional mencapai 1,5 miliar pada tahun 2019, dengan peningkatan 4% dibandingkan tahun sebelumnya [[1]](https://www.e-unwto.org/doi/abs/10.18111/wtobarometereng.2020.18.1.2). Tren pertumbuhan ini diperkirakan akan berlanjut, didorong oleh penemuan destinasi wisata baru dan dinamika yang berkembang dalam pasar pariwisata inbound dan outbound, terutama di negara-negara seperti Tiongkok dan Thailand [[2]](https://www.tandfonline.com/doi/full/10.1080/10941665.2020.1745855). Seiring dengan persiapan industri pariwisata untuk bangkit kembali, sangat penting untuk mengembangkan strategi dan kegiatan yang efektif guna memastikan pelayanan berkualitas bagi para wisatawan dan semua pemangku kepentingan yang terlibat. Persiapan ini harus melibatkan partisipasi semua pemangku kepentingan, dengan mengakui peran spesifik mereka [[3]](https://pwk.teknik.untan.ac.id/files/buku/fullbook-perencanaan-destinasi-pariwisata-compressed-compressed_1706694217.pdf). 

Indonesia adalah negara yang memiliki wilayah yang luas serta keberanekaragaman sumber daya alam dan kebudayaan serta adat istiadatnya yang beragam merupakan potensi yang dimiliki oleh Indonesia untuk dijadikan modal dalam pengembangan sektor pariwisatanya. Potensi-potensi tersebut memiliki kelayakan untuk dikelola dengan maksimal untuk membangun sektor pariwisata di Indonesia [[4]](https://jurnal.stie-aas.ac.id/index.php/jie/article/viewFile/13101/pdf). Banyaknya jumlah destinasi wisata membuat orang bingung dalam memilih destinasi wisata yang sesuai. Sistem rekomendasi adalah cara yang tepat untuk membantu masyarakat Indonesia memilih destinasi wisata yang sesuai dengan preferensi mereka [[5]](https://jurnal-itsi.org/index.php/jitsi/article/view/254/120).

## Business Understanding
### Problem Statement
- Bagaimana membuat sistem rekomendasi berdasarkan Kategori Wisata?
- Bagaimana membuat sistem rekomendasi wista berdasarkan rating yang diberi user sebelumnya?

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

**TF-IDF Vectorizer**
- Sebelum membangun sistem rekomendasi berbasis content-based filtering, persiapkan data dan simpan dalam variabel baru bernama *data*. 
- Selanjutnya, buat sistem rekomendasi berdasarkan tempat wisata yang pernah dikunjungi sebelumnya dengan memanfaatkan **TF-IDF Vectorizer** dari pustaka *scikit-learn*. Langkah ini meliputi inisialisasi **TfidfVectorizer**, perhitungan nilai *idf* pada kolom *place_name*, serta pemetaan indeks fitur ke nama fitur.
- Lakukan proses *fitting* dan transformasi pada fitur *place_name* untuk menghasilkan matriks representasi data.
- Ubah vektor hasil transformasi *TF-IDF* menjadi matriks menggunakan fungsi `todense()`.

### Cosine Similarity

Cosine similarity digunakan untuk mengukur sejauh mana dua data tempat (place) memiliki kesamaan dengan menghitung sudut antara keduanya. Teknik ini menilai tingkat kesamaan berdasarkan sudut antara data tempat yang sedang dianalisis. Hasil perhitungan ini akan menghasilkan nilai yang mencerminkan tingkat kesamaan antara dua data tempat, di mana nilai yang mendekati 1 menunjukkan kesamaan yang tinggi, dan nilai yang mendekati 0 menunjukkan kesamaan yang rendah.

- Selanjutnya, buat sebuah dataframe yang menampilkan matriks *TF-IDF* dengan kolom berisi *Place_Name* dan baris *Category*. Dataframe ini digunakan untuk menganalisis hubungan antara *place_name* dan kategorinya.  
- Setelah itu, hitung tingkat kesamaan (*similarity degree*) antar *place_name* dengan memanfaatkan fungsi **cosine_similarity** dari pustaka *scikit-learn*.

### Develop Content-Based Filtering Recommender System and Get Recommendation

Selanjutnya, dibuatkan function untuk mendapatkan rekomendasi berdasarkan kode berikut

```
def recommend_by_content_based_filtering(nama_tempat, similarity_data=cosine_sim_df, items_data=data_content_based[['Place_Name','Pattern','Category']], k=10):
    # Find the index of the given item
    filtered_items = items_data[items_data['Place_Name'] == nama_tempat]

    # Check if the item exists in the dataset
    if filtered_items.empty:
        print(f"Item '{nama_tempat}' not found in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame if item not found

    nama_tempat_index = filtered_items.index[0]

    # Get the similarity scores for the given item
    similarity_scores = similarity_data.iloc[nama_tempat_index]

    # Sort the items by similarity score in descending order
    nama_tempat_list = similarity_scores.sort_values(ascending=False).index[1:k + 1]

    # Create a DataFrame containing the recommendations
    recommended_items = items_data[items_data['Place_Name'].isin(nama_tempat_list)][['Place_Name', 'Category']]
    recommended_items['similarity_score'] = similarity_scores[nama_tempat_list].values

    return recommended_items
```
Fungsi ini dirancang untuk memberikan rekomendasi tempat wisata menggunakan *content-based filtering* dengan menghitung tingkat kesamaan (*similarity*) antara tempat yang diminta dengan tempat lainnya.

Parameter Fungsi

1. **`nama_tempat`**  
   Nama tempat wisata yang akan dicari rekomendasinya (dalam bentuk *string*).  
   
2. **`similarity_data`**  
   DataFrame berisi matriks kesamaan (*similarity matrix*) antar tempat wisata (default: `cosine_sim_df`).  

3. **`items_data`**  
   DataFrame yang memuat informasi tempat wisata, termasuk kolom *Place_Name*, *Pattern*, dan *Category*.  

4. **`k`**  
   Jumlah rekomendasi yang ingin dihasilkan (default: 10).  

## Kelebihan dan Kekurangan Content-Based Filtering
Content-based filtering adalah salah satu pendekatan dalam sistem rekomendasi yang menggunakan informasi atau atribut dari item dan preferensi pengguna untuk memberikan rekomendasi. Berikut adalah kelebihan dan kekurangan dari metode ini:

**Kelebihan:**
- Personalisasi Tinggi, Sistem merekomendasikan item berdasarkan preferensi unik pengguna, sehingga hasil rekomendasi lebih relevan bagi masing-masing individu.
- Tidak Membutuhkan Data Pengguna Lain (Cold Start User) Tidak memerlukan data dari pengguna lain karena hanya fokus pada data pengguna tersebut, cocok untuk situasi di mana data komunitas terbatas.
Kemampuan Mempelajari Preferensi yang Spesifik
- Sistem dapat menangkap preferensi pengguna dengan baik jika atribut item memiliki deskripsi yang lengkap dan relevan.
Kesesuaian untuk Item Baru (Cold Start Item)
- Item baru dengan deskripsi yang lengkap dapat dengan mudah direkomendasikan tanpa menunggu ulasan atau peringkat dari pengguna lain.
Independen terhadap Komunitas
- Karena fokus pada pengguna individu, tidak ada risiko efek "popularity bias" seperti pada collaborative filtering.

**Kekurangan :**

- Keterbatasan pada Data Atribut (Feature Engineering), Kualitas rekomendasi sangat bergantung pada kelengkapan dan akurasi atribut item. Jika atribut kurang representatif, hasilnya bisa kurang optimal.
Tidak Menangkap Keanekaragaman
- Sistem cenderung merekomendasikan item yang mirip dengan item yang disukai sebelumnya, sehingga rentan terhadap masalah overspecialization (tidak memberikan rekomendasi yang beragam).
Kesulitan pada Preferensi Baru

## Modeling Collaborative Filtering

Collaborative Filtering adalah pendekatan dalam sistem rekomendasi yang membuat prediksi atau rekomendasi item berdasarkan perilaku, preferensi, atau interaksi pengguna dengan item, tanpa memerlukan atribut spesifik dari item tersebut. Teknik ini memanfaatkan data dari komunitas pengguna untuk menemukan pola kesamaan antara pengguna atau item.

### Data Understanding
Data yang digunakan adalah data **`user`**

## Data Preparation
**Mengubah userID menjadi list tanpa nilai yang sama dan Melakukan proses encoding angka ke ke userID**

Pada tahapan mengubah User_Id menjadi list unik, kode mengambil kolom User_Id dari data data_collaborative_filtering, lalu mengekstrak nilai unik dari kolom tersebut menggunakan .unique(). Hasilnya diubah menjadi list menggunakan .tolist() yang bertujuan untuk menghilangkan duplikasi nilai User_Id, sehingga hanya nilai unik yang digunakan. Hal ini penting untuk memastikan setiap pengguna di-encode hanya sekali.

Selanjutnya, Membuat dictionary (user_to_user_encoded) yang memetakan setiap User_Id unik ke indeks angka menggunakan fungsi enumerate dengan tujuan untuk mengubah User_Id menjadi representasi numerik, yang lebih efisien untuk diproses oleh algoritma machine learning, terutama dalam operasi matriks seperti collaborative filtering.

Terakhir, Membuat dictionary (user_encoded_to_user) yang merupakan kebalikan dari user_to_user_encoded. Dictionary ini memetakan indeks angka kembali ke nilai User_Id. Hal ini memungkinkan decoding hasil prediksi (yang dalam format numerik) kembali ke format asli User_Id agar lebih mudah dipahami atau
ditampilkan.

Sehingga hasil pada tahapan ini adalah sebagai berikut

| userID | Encoded userID | Decoded userID |
|--------|----------------|----------------|
| 1      | 0              | 1              |
| 2      | 1              | 2              |
| 3      | 2              | 3              |
| 4      | 3              | 4              |
| 5      | 4              | 5              |
| 6      | 5              | 6              |
| 7      | 6              | 7              |
| 8      | 7              | 8              |
| 9      | 8              | 9              |
| 10     | 9              | 10             |
| ...    | ...            | ...            |
| 300    | 299            | 300            |

## Penyesuaian Data

```

# Mengubah placeID menjadi list tanpa nilai yang sama
place_ids = data_collaborative_filtering['Place_Id'].unique().tolist()

# Melakukan proses encoding placeID
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}

# Melakukan proses encoding angka ke placeID
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

# Mapping userID ke dataframe user
data_collaborative_filtering['user'] = data_collaborative_filtering['User_Id'].map(user_to_user_encoded)

# Mapping placeID ke dataframe
data_collaborative_filtering['place_encoded'] = data_collaborative_filtering['Place_Id'].map(place_to_place_encoded)

# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)

# Mendapatkan jumlah resto
num_place = len(place_to_place_encoded)
print(num_place)

# Mengubah rating menjadi nilai float
data_collaborative_filtering['Place_Ratings'] = data_collaborative_filtering['Place_Ratings'].values.astype(np.float32)

# Nilai minimum rating
min_rating = min(data_collaborative_filtering['Place_Ratings'])

# Nilai maksimal rating
max_rating = max(data_collaborative_filtering['Place_Ratings'])

print('Number of User: {}, Number of Places: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_place, min_rating, max_rating
))

```
Kode di atas merupakan bagian dari proses pemrosesan data untuk sistem rekomendasi berbasis kolaboratif filtering. Pertama, kode ini mengambil nilai unik dari kolom Place_Id dalam dataframe data_collaborative_filtering dan mengubahnya menjadi sebuah list tanpa nilai yang sama. Selanjutnya, dilakukan encoding terhadap Place_Id dengan membuat dua dictionary: place_to_place_encoded, yang memetakan setiap Place_Id ke indeks numerik yang unik, dan place_encoded_to_place, yang melakukan pemetaan terbalik dari indeks ke Place_Id.

Kemudian, kode ini memetakan User_Id ke dataframe dengan menggunakan dictionary user_to_user_encoded, sehingga setiap pengguna mendapatkan representasi numerik. Selanjutnya, Place_Id juga dipetakan ke dataframe menggunakan dictionary place_to_place_encoded, menghasilkan kolom baru bernama place_encoded.

Setelah proses pemetaan, kode menghitung jumlah pengguna dan jumlah tempat yang ada dalam data dengan menghitung panjang dari dictionary yang telah dibuat. Selain itu, rating tempat diubah menjadi tipe data float untuk memudahkan analisis lebih lanjut. Kode ini juga menentukan nilai minimum dan maksimum dari rating yang diberikan. Terakhir, informasi mengenai jumlah pengguna, jumlah tempat, serta nilai minimum dan maksimum rating dicetak ke layar dalam format yang jelas. Kode ini secara keseluruhan bertujuan untuk mempersiapkan data agar siap digunakan dalam algoritma rekomendasi.
Relate

Output :

```
300
437
Number of User: 300, Number of Places: 437, Min Rating: 1.0, Max Rating: 5.0
```


## Spliting
Dua kolom dari dataframe data_collaborative_filtering diambil, yaitu user dan place_encoded, dan mengubahnya menjadi array numpy yang disimpan dalam variabel x. Array ini akan digunakan sebagai fitur input untuk model, di mana setiap baris mewakili pasangan pengguna dan tempat.

Selanjutnya, kode membuat variabel y, yang berisi rating tempat yang telah dinormalisasi. Proses normalisasi dilakukan dengan menggunakan rumus min-max scaling, di mana setiap rating dikurangi dengan nilai minimum (min_rating) dan dibagi dengan rentang rating (selisih antara nilai maksimum dan minimum). Hasilnya adalah rating yang berada dalam rentang 0 hingga 1, yang lebih mudah digunakan dalam model pembelajaran mesin.

Setelah itu, **data dibagi menjadi dua set: 80% untuk data pelatihan (train) dan 20% untuk data validasi (validation)**. Indeks pemisahan ditentukan dengan menghitung 80% dari jumlah total baris dalam dataframe. Kemudian, x_train dan y_train berisi data pelatihan, sedangkan x_val dan y_val berisi data validasi.

Terakhir, kode mencetak nilai dari x dan y, memberikan gambaran tentang data yang telah dipersiapkan untuk digunakan dalam pelatihan model rekomendasi. Proses ini penting untuk memastikan bahwa model dapat belajar dari data yang representatif dan juga dapat diuji pada data yang tidak terlihat sebelumnya untuk mengevaluasi performanya.

## Model Development and Training of Colaborative Filtering
Model pada Colaborative filtering dibuat sebanyak 2, yaitu **`RecommenderNet`** dan **`RecommenderNetv2`**

### RecommenderNet
Kode model pada RecommenderNet adalah sebagai berikut

```
class RecommenderNet(tf.keras.Model):

  # Insialisasi fungsi
  def __init__(self, num_users, num_place, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_place = num_place
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.place_embedding = layers.Embedding( # layer embeddings resto
        num_place,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.place_bias = layers.Embedding(num_place, 1) # layer embedding resto bias

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    place_vector = self.place_embedding(inputs[:, 1]) # memanggil layer embedding 3
    place_bias = self.place_bias(inputs[:, 1]) # memanggil layer embedding 4

    dot_user_place = tf.tensordot(user_vector, place_vector, 2)

    x = dot_user_place + user_bias + place_bias

    return tf.nn.sigmoid(x) # activation sigmoid
```
Model RecommenderNet yang didefinisikan dalam kode di atas adalah sebuah jaringan saraf yang dirancang untuk sistem rekomendasi berbasis pembelajaran mendalam. Model ini menggunakan teknik embedding untuk merepresentasikan pengguna dan tempat (restoran) dalam ruang vektor yang lebih rendah, memungkinkan model untuk belajar dari interaksi antara pengguna dan tempat dengan lebih efektif. Berikut adalah penjelasan detail mengenai arsitektur model ini:

**1. Inisialisasi Model**

**Konstruktor (__init__)**

Parameter:
- num_users: Jumlah pengguna dalam dataset.
- num_place: Jumlah tempat (restoran) dalam dataset.
- embedding_size: Ukuran vektor embedding untuk pengguna dan tempat.

Layer Embedding:
- user_embedding: Layer ini digunakan untuk mengubah ID pengguna menjadi vektor embedding berukuran embedding_size. Vektor ini diinisialisasi menggunakan distribusi normal He (he_normal) dan dilengkapi dengan regularisasi L2 untuk mencegah overfitting.
- user_bias: Layer ini menyimpan bias untuk setiap pengguna, yang merupakan nilai tambahan yang ditambahkan ke prediksi akhir untuk memperbaiki akurasi model.
- place_embedding: Serupa dengan user_embedding, layer ini mengubah ID tempat menjadi vektor embedding berukuran embedding_size, juga dengan inisialisasi dan regularisasi yang sama.
- place_bias: Layer ini menyimpan bias untuk setiap tempat, memberikan penyesuaian tambahan pada prediksi berdasarkan karakteristik spesifik dari tempat tersebut.

**2. Metode call**
Metode call adalah inti dari model, di mana proses komputasi dilakukan saat model dipanggil dengan input.
Input:
- inputs: Sebuah tensor dua dimensi yang berisi pasangan ID pengguna dan ID tempat. Setiap baris mewakili satu interaksi antara pengguna dan tempat.
- Proses:
- Mendapatkan Vektor Embedding:
- user_vector: Mengambil embedding pengguna berdasarkan ID pengguna dari kolom pertama input.
- user_bias: Mengambil bias pengguna berdasarkan ID pengguna.
- place_vector: Mengambil embedding tempat berdasarkan ID tempat dari kolom kedua input.
- place_bias: Mengambil bias tempat berdasarkan ID tempat.
- Menghitung Prediksi:
- dot_user_place: Menghitung produk titik (dot product) antara vektor embedding pengguna dan vektor embedding tempat. Ini memberikan ukuran seberapa cocok pengguna tersebut dengan tempat tertentu.
- x: Menjumlahkan hasil produk titik dengan bias pengguna dan bias tempat. Ini menghasilkan nilai prediksi yang belum dinormalisasi.

**Aktivasi:**
Model menggunakan fungsi aktivasi sigmoid (tf.nn.sigmoid(x)) pada output akhir. Fungsi sigmoid mengubah nilai prediksi menjadi rentang antara 0 dan 1, yang cocok untuk masalah regresi biner atau probabilitas rating.

## RecommenderNetv2

Model `RecommenderNetV2` adalah jaringan saraf yang dirancang untuk sistem rekomendasi menggunakan teknik embedding dan lapisan dense tambahan. Berikut adalah rincian dari arsitektur model ini.

**1. Inisialisasi Model**

**Konstruktor (`__init__`)**

- **Parameter**:
  - `num_users`: Jumlah pengguna dalam dataset.
  - `num_place`: Jumlah tempat (restoran) dalam dataset.
  - `embedding_size`: Ukuran vektor embedding untuk pengguna dan tempat.

- **Layer Embedding**:
  - **`user_embedding`**: Layer ini mengubah ID pengguna menjadi vektor embedding berukuran `embedding_size`. Vektor diinisialisasi menggunakan distribusi normal He (`he_normal`) dan dilengkapi dengan regularisasi L2 untuk mencegah overfitting.
  
  - **`user_bias`**: Layer ini menyimpan bias untuk setiap pengguna, memberikan penyesuaian tambahan pada prediksi akhir.

  - **`place_embedding`**: Layer ini mengubah ID tempat menjadi vektor embedding berukuran `embedding_size`, juga dengan inisialisasi dan regularisasi yang sama.

  - **`place_bias`**: Layer ini menyimpan bias untuk setiap tempat, memberikan penyesuaian tambahan pada prediksi berdasarkan karakteristik spesifik tempat tersebut.

**2. Metode `call`**

Metode `call` adalah bagian utama dari model, di mana proses komputasi dilakukan saat model dipanggil dengan input.

- **Input**: 
  - `inputs`: Sebuah tensor dua dimensi yang berisi pasangan ID pengguna dan ID tempat. Setiap baris mewakili satu interaksi antara pengguna dan tempat.

- **Proses**:
  - **Mendapatkan Vektor Embedding**:
    - `user_vector`: Mengambil embedding pengguna berdasarkan ID pengguna dari kolom pertama input.
    - `user_bias`: Mengambil bias pengguna berdasarkan ID pengguna.
    - `place_vector`: Mengambil embedding tempat berdasarkan ID tempat dari kolom kedua input.
    - `place_bias`: Mengambil bias tempat berdasarkan ID tempat.
  
  - **Menghitung Prediksi**:
    - `dot_user_place`: Menghitung produk titik (dot product) antara vektor embedding pengguna dan vektor embedding tempat, memberikan ukuran seberapa cocok pengguna tersebut dengan tempat tertentu.
    
    - `x`: Menjumlahkan hasil produk titik dengan bias pengguna dan bias tempat. Ini menghasilkan nilai prediksi yang belum dinormalisasi.

- **Aktivasi**:
  - Model menggunakan fungsi aktivasi sigmoid (`tf.nn.sigmoid(x)`) pada output akhir. Fungsi sigmoid mengubah nilai prediksi menjadi rentang antara 0 dan 1, yang cocok untuk masalah regresi biner atau probabilitas rating.

**3. Penambahan Lapisan Dense**

Setelah menghitung nilai prediksi dasar, model menambahkan beberapa lapisan dense untuk meningkatkan kompleksitas:

- **Lapisan Dense Pertama**: 
  - Menggunakan 64 neuron dengan fungsi aktivasi ReLU (`relu`). Ini membantu model belajar representasi non-linear dari data.

- **Dropout Layer**: 
  - Dropout sebesar 20% ditambahkan setelah lapisan dense pertama untuk mencegah overfitting dengan secara acak menonaktifkan beberapa neuron selama pelatihan.

- **Lapisan Dense Kedua**: 
  - Menggunakan 32 neuron dengan fungsi aktivasi ReLU. Ini memberikan lapisan tambahan untuk memperdalam representasi fitur sebelum menghasilkan output akhir.

# Evaluation
## Evaluation of Content-Based Filtering Model Result
Pada hasil oleh model Content-Based Filtering, Dibuat sebanyak 3 skenario untuk melakukan rekomendasi. Dalam hal ini akan diuji N sebanyak 10 rekomendasi tempat wisata dengan kategorinya. 

Pada bagian ini, Metrik evaluasi yang digunakan adalah recommender system precision. Disini precision merupakan jumlah item yang direkomendasikan yang relevan.

![precision formula](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/precision.png)

### Hasil pada Skenario 1
Skenario 1 adalah mencari rekomendasi tempat pada **Candi Prambanan**
```
# Skenario 1 : Candi Prambanan
recommend_by_content_based_filtering('Candi Prambanan')
```
hasilnya adalah 
| Place_Name   | Category       | similarity_score |
|:-------------|:---------------:|------------------|
| Keraton Yogyakarta     | Budaya         | 0.634871          |
| Tebing Breksi           | Budaya         | 0.553593          |
| Candi Donotirto         | Budaya         | 0.488287          |
| Puncak Pinus Becici      | Taman Hiburan   | 0.407251          |
| Candi Borobudur         | Budaya         | 0.305599          |
| Candi Ijo               | Budaya         | 0.222767          |
| Candi Ratu Boko         | Budaya         | 0.207963          |
| Candi Sewu              | Budaya         | 0.170799          |
| Candi Gedong Songo      | Budaya         | 0.096699          |
| Pura Giri Natha         | Budaya         | 0.096278          |


Presisi budaya dapat dihitung dengan menggunakan rumus berikut:
```
Presisi Budaya = (9/10)*100 = 90%
```

Di mana:
- **9** adalah jumlah prediksi positif yang benar pada kategori budaya.
- **10** adalah total jumlah prediksi positif.


### Hasil pada Skenario 2
Skenario 2 adalah mencari rekomendasi tempat pada **Museum Basoeki Abdullah**
```
# Case 2 : Museum Basoeki Abdullah
recommend_by_content_based_filtering('Museum Basoeki Abdullah')
```

Hasil dari kode :
| Place_Name                          | Category | similarity_score |
|-------------------------------------|----------|------------------|
| Museum Taman Prasasti              | Budaya   | 0.249897         |
| Museum Wayang                       | Budaya   | 0.215493         |
| Museum Nasional                     | Budaya   | 0.212838         |
| Museum Bahari Jakarta               | Budaya   | 0.209531         |
| Museum Seni Rupa dan Kramik        | Budaya   | 0.207353         |
| Museum Tengah Kebun                 | Budaya   | 0.203953         |
| De Mata Museum Jogja                | Budaya   | 0.197842         |
| Museum Sonobudoyo Unit I            | Budaya   | 0.190391         |
| Museum Barli                        | Budaya   | 0.185569         |
| Museum Pendidikan Nasional           | Budaya   | 0.174056         |

```
Presisi Budaya = (10/10)*100 = 100%
```

Di mana:
- **10** adalah jumlah prediksi positif yang benar pada kategori budaya.
- **10** adalah total jumlah prediksi positif.

### Hasil pada Skenario 3
Skenario 3 adalah mencari rekomendasi tempat pada **Dunia Fantasi**

`recommend_by_content_based_filtering('Dunia Fantasi')`

menghasilkan output 
| Place_Name                                | Category      | similarity_score |
|-------------------------------------------|---------------|------------------|
| Taman Mini Indonesia Indah (TMII)        | Taman Hiburan | 0.469224         |
| Taman Impian Jaya Ancol                  | Taman Hiburan | 0.189286         |
| Pelabuhan Marina                          | Bahari       | 0.140093         |
| Kidzania                                  | Taman Hiburan | 0.129180         |
| Sea World                                 | Taman Hiburan | 0.119722         |
| Jakarta Aquarium dan Safari               | Taman Hiburan | 0.119421         |
| Taman Situ Lembang                        | Taman Hiburan | 0.116117         |
| Pantai Ancol                              | Bahari       | 0.114812         |
| Taman Spathodea                          | Taman Hiburan | 0.113678         |
| Taman Balai Kota Bandung                  | Taman Hiburan | 0.112755         |

```
Presisi Taman Hiburan = (8/10)*100 = 80%
```

Di mana:
- **8** adalah jumlah prediksi positif yang benar pada kategori Taman Hiburan.
- **10** adalah total jumlah prediksi positif.


Secara umum, model **Content-Based Filtering** telah berhasil melakukan rekomendasi dengan sangat baik. **Hal ini telah menjawab Problem Statement 1 : Bagaimana membuat sistem rekomendasi berdasarkan Kategori Wisata?**

## Evaluation of Collaborative Filtering Model Result
### Hasil pada **`RecommenderNet`** Model

![](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/RecommendedNet1.png)


1. Nilai Loss dan RMSE
Loss (0.1156): Nilai loss yang rendah menunjukkan bahwa model memiliki kesalahan yang kecil dalam memprediksi output pada data pelatihan. 
RMSE (0.3398): RMSE memberikan gambaran tentang seberapa besar rata-rata deviasi prediksi model dari nilai sebenarnya. Nilai RMSE yang rendah menunjukkan bahwa prediksi model cukup akurat. Namun, interpretasi nilai ini juga bergantung pada skala dan konteks data yang digunakan. Misalnya, jika output yang diprediksi berada dalam rentang 0 hingga 1, maka RMSE sebesar 0.3398 mungkin dianggap cukup baik.

2. Validation Metrics
Val Loss (0.1256): Nilai loss pada data validasi sedikit lebih tinggi dibandingkan dengan loss pada data pelatihan, yang bisa mengindikasikan adanya overfitting. Ini berarti model mungkin telah belajar pola dari data pelatihan tetapi tidak dapat menggeneralisasi dengan baik ke data baru.
Val RMSE (0.3542): RMSE pada data validasi juga lebih tinggi dibandingkan dengan RMSE pada data pelatihan. Ini menunjukkan bahwa prediksi model pada data validasi kurang akurat dibandingkan dengan prediksi pada data pelatihan.

### Hasil pada **`RecommenderNetv2`** Model

![](https://raw.githubusercontent.com/farhanrn/Indonesia-Tourism-Recommender-System/refs/heads/main/src/RecommendedNetv2.png)

1. Loss dan RMSE
Loss (6.1803e-08): Nilai loss yang sangat rendah ini menunjukkan bahwa model memiliki kesalahan yang sangat kecil dalam memprediksi output pada data pelatihan. Notasi 6.1803e-08 berarti 0.000000061803, yang menunjukkan performa model yang sangat baik.
Root Mean Squared Error (RMSE) (0.3475): RMSE ini juga menunjukkan bahwa rata-rata deviasi prediksi model dari nilai sebenarnya cukup kecil, meskipun nilainya sedikit lebih tinggi dibandingkan dengan loss.

2. Validation Metrics
Val Loss (6.0744e-08): Nilai loss pada data validasi juga sangat rendah, hanya sedikit lebih rendah dari loss pada data pelatihan, yang menunjukkan bahwa model tidak hanya bekerja baik pada data pelatihan tetapi juga dapat generalisasi dengan baik ke data baru.
Val RMSE (0.3472): RMSE pada data validasi hampir sama dengan RMSE pada data pelatihan, menunjukkan bahwa model mampu mempertahankan akurasi prediksi yang baik di kedua set data.

# References
[1] "World Tourism Barometer," UNWTO, vol. 18, no. 1, 2020. [Online]. Available: https://www.e-unwto.org/doi/abs/10.18111/wtobarometereng.2020.18.1.2. [Accessed: Nov. 29, 2024].
[2] 

[2] Fakfare P, Taawanich S, Wattanacharoensilb W, "A scale development and validation on domestic tourists’ motivation: the case of second-tier tourism destinations" *Asia Pacific Journal of Tourism Research*, vol. 29, no. 4, pp. 123-145, 2020. [Online]. Available: https://www.tandfonline.com/doi/full/10.1080/10941665.2020.1745855. [Accessed: Nov. 29, 2024].

[3] Nasrullah et al., *Perencanaan Destinasi Pariwisata*, Yayasan Kita Menulis, 2023. [Online]. Available: https://pwk.teknik.untan.ac.id/files/buku/fullbook-perencanaan-destinasi-pariwisata-compressed-compressed_1706694217.pdf. [Accessed: Nov. 29, 2024].

[4] Wijianto, "Strategi Pengembangan Wisata Alami dalam Era Digitalisasi," *Edunomika*, vol. 8, no. 2, 2024. [Online]. Available: https://jurnal.stie-aas.ac.id/index.php/jie/article/viewFile/13101/pdf. [Accessed: Nov. 29, 2024].

[5] D. M. Saputra, N. Angelia, and N. Yusliani, “Recommender System for Tourist Destinations in Indonesia Using Matrix Factorization Method”, jitsi, vol. 5, no. 3, pp. 122 - 127, Oct. 2024.
