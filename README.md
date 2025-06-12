# Laporan Proyek Machine Learning - Yuliana

## Project Overview

Membaca merupakan aktivitas penting yang mendukung pengembangan pengetahuan, imajinasi, serta keterampilan berpikir kritis. Namun, di tengah banyaknya pilihan buku yang tersedia, pembaca sering kali mengalami kesulitan dalam menemukan buku yang sesuai dengan preferensi mereka. Oleh karena itu, diperlukan sistem rekomendasi yang mampu menyarankan buku-buku yang relevan secara personal.

Proyek ini bertujuan untuk membangun sistem rekomendasi buku berbasis machine learning dengan memanfaatkan data rating yang diberikan oleh pengguna terhadap buku tertentu. Sistem ini diharapkan mampu memberikan daftar top-N rekomendasi buku untuk setiap pengguna berdasarkan pola preferensi mereka.

Pentingnya proyek ini terletak pada kontribusinya dalam meningkatkan pengalaman pengguna, menghemat waktu dalam memilih buku, dan juga membantu penulis/penerbit dalam menjangkau pembaca yang lebih tepat sasaran.

## Business Understanding
### Problem Statements
Di era digital saat ini, jumlah buku yang tersedia secara online sangat besar, sehingga pengguna sering mengalami kesulitan dalam menemukan buku yang sesuai dengan preferensi mereka. Tanpa sistem rekomendasi yang tepat, pengguna bisa kesulitan menyaring informasi yang relevan dari ratusan ribu pilihan.

### Goals

Tujuan dari proyek ini adalah membangun sistem rekomendasi buku yang mampu:
- Memberikan rekomendasi buku yang relevan untuk setiap pengguna.
- Meningkatkan pengalaman pengguna dalam menemukan buku baru yang sesuai dengan minat mereka.
- Mengidentifikasi pola preferensi pengguna berdasarkan data historis.

### Solution statements
Dalam menyelesaikan masalah ini, dua pendekatan sistem rekomendasi dipertimbangkan:
1. **Content-Based Filtering**
Sistem ini memberikan rekomendasi berdasarkan kesamaan atribut buku (genre, penulis, dll.) dengan buku yang sebelumnya disukai user. Pendekatan ini tidak dipilih sebagai fokus karena keterbatasan atribut konten pada data.

2. **Collaborative Filtering**
Pendekatan ini memberikan rekomendasi berdasarkan pola interaksi antar pengguna. Jika dua pengguna memberi rating yang mirip pada beberapa buku, sistem akan merekomendasikan buku dari satu pengguna ke pengguna lainnya.

Dalam proyek ini digunakan pendekatan Neural Collaborative Filtering (NCF), yang mengandalkan embedding layer dan dense layers untuk mempelajari hubungan non-linier antara user dan item.

### **Alasan Pemilihan Collaborative Filtering**
- Cocok untuk dataset besar dengan interaksi user-item.
- Berpotensi menghasilkan rekomendasi yang lebih personal dan fleksibel.

## Data Understanding
Dataset yang digunakan berasal dari Kaggle: [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

Dataset ini berasal dari komunitas Book-Crossing, dikumpulkan oleh Cai-Nicolas Ziegler dalam crawling selama 4 minggu (Agustus–September 2004). Dataset ini sangat cocok digunakan untuk membangun sistem rekomendasi berbasis buku karena memiliki informasi eksplisit dan implisit dari pengguna terhadap berbagai buku.
1. Dataset terdiri dari tiga file utama:

- Users.csv

  Berisi data pengguna, termasuk ID, lokasi, dan usia. → Total: 278.858 pengguna

- Books.csv
  
  Berisi metadata buku seperti ISBN, judul, penulis, tahun terbit, penerbit, dan URL sampul buku. → Total: 271.360 buku

- Ratings.csv
  
  Berisi data rating buku dari pengguna. Rating eksplisit berupa angka 1–10, sedangkan 0 dianggap sebagai implicit feedback (misalnya hanya melihat atau klik). → Total: 1.149.780 rating

2. Deskripsi Fitur
   
   Books.csv
  - ISBN : ID unik untuk setiap buku
  - Book-Title : Judul buku
  - Book-Author : Nama penulis (jika lebih dari satu, hanya penulis pertama)
  - Year-Of-Publication : Tahun buku diterbitkan
  - Publisher : Nama penerbit
  - Image-URL-S/M/L : URL gambar sampul kecil/sedang/besar (dari Amazon)

  Users.csv
  - User-ID : ID unik pengguna (sudah dianonimkan)
  - Location : Lokasi pengguna (negara/kota)
  - Age : Umur pengguna (beberapa kosong atau tidak tersedia)

  Ratings.csv
  - User-ID : ID pengguna
  - ISBN : ID buku
  - Book-Rating : Nilai rating, skala 0–10
    - 0 = rating implisit (misal: hanya melihat/klik, bukan review)
    - 1–10 = rating eksplisit
   
3.Kondisi Awal Data
  1. Dataset books
  - Memiliki missing value:
      - Book-Author:            2
      - Publisher:              2
      - Image-URL-L:            3
  - Tidak memiliki duplikat
       
  2. Dataset users
  - Memiliki missing value:
      - Age:         110762
  - Tidak memiliki duplikat
       
  3. Dataset ratings
  - Tidak memiliki missing value
  - Tidak memiliki duplikat
  
## Data Preparation
### Data Prepocessing
Sebelum membangun model sistem rekomendasi, dilakukan beberapa tahap data preprocessing untuk memastikan kualitas data yang baik dan relevan. Tahapan preprocessing yang dilakukan adalah sebagai berikut:
1. Filter Rating
   Dataset awal mengandung nilai rating 0 yang menunjukkan bahwa pengguna tidak benar-benar memberikan penilaian terhadap buku. Oleh karena itu, data difilter agar hanya menyertakan rating dengan nilai lebih dari 0.
   
       ratings = ratings[ratings['Book-Rating'] > 0]

2. Filter Pengguna dan Buku yang Aktif
   Untuk mengurangi sparsity dan meningkatkan kualitas interaksi dalam sistem rekomendasi, hanya pengguna dan buku yang aktif yang disertakan dalam analisis. Kriteria yang digunakan adalah sebagai berikut:
- Pengguna yang telah memberikan setidaknya 5 penilaian.
- Buku yang telah menerima setidaknya 5 penilaian dari pengguna berbeda.
   Hasilnya, diperoleh subset data dengan pengguna dan buku yang lebih aktif, yang dapat membantu model dalam mempelajari pola preferensi dengan lebih efektif.

  Sebagai ilustrasi, berikut adalah 10 pengguna dengan jumlah rating terbanyak:
    - ID 11676:     8524
    - ID 98391:     5802
    - ID 153662:    1969
    - ID 189835:    1906
    - ID 23902:     1395
    - ID 76499:     1036
    - ID 171118:    1035
    - ID 235105:    1023
    - ID 16795:      968
    - ID 248718:     948

3. Penggabungan Data Buku
   Untuk memperkaya informasi pada dataset, data rating digabungkan dengan metadata buku berdasarkan kolom ISBN.
   
4. Variabel dalam Dataset Gabungan ratings
- User-ID:	ID unik untuk setiap pengguna
- ISBN:	Kode unik untuk buku
- Book-Rating:	Nilai rating yang diberikan user (range: 0–10)
- Book-Title:	Judul buku
- Book-Author:	Nama penulis
- Year-Of-Publication:	Tahun publikasi
- Publisher:	Nama penerbit
- Image-URL-S/M/L:	Link gambar sampul buku dalam berbagai ukuran
  
5. Kondisi Awal Data Gabungan
- Sebagian besar rating = 0 (implisit) → Hanya rating eksplisit (>0) yang dipertahankan untuk modeling.
- Distribusi rating tidak merata → Sebagian besar pengguna hanya memberi sedikit rating → Filtering pengguna aktif yang memberi ≥5 rating.
- Filtering buku populer → Hanya menyertakan buku yang mendapat ≥5 rating untuk menjamin relevansi model.
- Merge antar file, file Ratings.csv digabung dengan Books.csv mendapatkan metadata buku → diperoleh data gabungan yang terdiri dari 137.214 interaksi user-book-rating.

### Tahapan Data Preparation
Tahap data preparation dilakukan untuk memastikan bahwa data yang digunakan sesuai dan optimal bagi model sistem rekomendasi berbasis Collaborative Filtering. Proses ini penting untuk mengubah data mentah menjadi format numerik yang dapat diproses oleh algoritma pembelajaran mesin. Berikut adalah langkah-langkah yang diterapkan secara berurutan:

1. Cek missing value dan duplikat data gabungan (tidak ada).
2. Salin Dataframe Asli
   Tujuan: Menghindari perubahan langsung pada data asli `ratings`, agar tetap tersedia jika diperlukan untuk keperluan lain (seperti visualisasi atau validasi).
3. Konversi Tipe Data Rating
  Tujuan: Rating dikonversi ke tipe float32 agar lebih efisien dalam pemrosesan numerik dan kompatibel dengan algoritma pembelajaran mendalam.
4. Hitung Nilai Minimum dan Maksimum Rating
   Tujuan: Mengetahui rentang nilai rating yang akan membantu dalam proses normalisasi atau interpretasi model.
5. Encoding User dan Book
   Tujuan: Mengubah ID pengguna (User-ID) dan ID buku (ISBN) dari format string menjadi nilai numerik integer. Ini penting karena algoritma machine learning hanya dapat bekerja dengan data numerik.
6. Ubah Nama Kolom Rating
    Tujuan: Menyederhanakan nama kolom untuk memudahkan integrasi dengan model rekomendasi, terutama jika menggunakan framework deep learning.
7. Simpan Jumlah Unik User dan Book
    Tujuan: Informasi ini dibutuhkan dalam pembuatan layer embedding pada model pembelajaran mendalam, agar dapat menentukan dimensi input yang tepat.
   
8. Shuffle Data
   Agar urutan sampel tidak memengaruhi pelatihan, seluruh baris data diacak secara acak (shuffle).

9. Siapkan Fitur dan Label
- Fitur (x): Pasangan (user, book) yang sudah diencode ke integer.
- Label (y): Nilai rating asli, dinormalisasi ke rentang [0, 1] agar sesuai dengan fungsi aktivasi sigmoid di model.
  
        # Fitur: kolom encoded user dan book
        x = df[['user', 'book']].values

        # Label: rating dinormalisasi
        y = df['rating'] \
                  .apply(lambda r: (r - min_rating) / (max_rating - min_rating)) \
                  .values
    
10. Membagi Data untuk Training dan Validasi

  Sebelum melakukan pelatihan model, data dibagi menjadi dua bagian:
  - 80% data training: digunakan untuk melatih model.
  - 20% data validasi: digunakan untuk mengevaluasi performa model pada data yang belum pernah dilihat.

  *Alasan Normalisasi Rating:*

  Rating dinormalisasi ke rentang 0–1 untuk menyesuaikan dengan fungsi aktivasi sigmoid pada model dan mempercepat proses konvergensi
    
## Modeling
1. Membangun Arsitektur Model-based Collaborative Filtering
   
Pada tahap ini, pendekatan yang digunakan adalah Collaborative Filtering berbasis Neural Network Embedding, di mana baik pengguna maupun buku direpresentasikan dalam bentuk vektor embedding berdimensi tertentu. Model dirancang dengan dua input embedding (untuk user dan book), yang kemudian digabungkan dan diteruskan ke beberapa lapisan dense untuk menghasilkan skor prediksi rating.

Model dilatih menggunakan binary crossentropy sebagai loss function, dengan Root Mean Squared Error (RMSE) sebagai metrik evaluasi. Dataset dibagi menjadi 80% data latih dan 20% data validasi.

Pelatihan dilakukan selama 100 epoch. Hasil pelatihan menunjukkan bahwa RMSE pada data latih menurun secara signifikan hingga mencapai nilai mendekati 0.15, sedangkan pada data validasi cenderung stabil pada kisaran 0.18 setelah epoch ke-15.

  Model yang digunakan merupakan neural network sederhana yang terdiri dari:
  - Embedding Layer: untuk memetakan user dan book ke dalam vektor berdimensi tetap (50 dimensi).
  - Dot Product: menghitung kesamaan antara vektor user dan buku.
  - Bias Term: menangkap kecenderungan umum user atau buku terhadap rating tinggi/rendah.
  - Sigmoid Activation: menghasilkan skor prediksi dalam rentang 0–1 (karena rating telah dinormalisasi).
    
2. Proses Training
  Model dilatih dengan parameter:
  - Loss function: BinaryCrossentropy karena target berupa skor dalam rentang 0–1.
  - Optimizer: Adam dengan learning_rate=0.001.
  - Metric evaluasi: Root Mean Squared Error (RMSE).

      model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
      )
    
  - Model dilatih menggunakan teknik early stopping untuk menghindari overfitting:
      early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
      )
  - Training dijalankan dengan:
    
      history = model.fit(
        x = x_train,
        y = y_train,
        batch_size = 8,
        epochs = 100,
        validation_data = (x_val, y_val)
      )
  3. Top-N Recommendation (Output)
    Setelah model dilatih, sistem dapat memberikan Top-N Recommendation kepada user dengan cara:
    - Ambil ID user tertentu (misalnya user_id = 5).
    - Prediksi skor untuk semua buku yang belum pernah diberi rating oleh user tersebut.
    - Urutkan skor tertinggi.
    - Tampilkan N buku dengan skor prediksi tertinggi sebagai rekomendasi.

    Berikut ini adalah contoh preferensi genre atau pola rating user ID 139467:
    - Buku yang disukai user (dengan rating tinggi):
      - Charming Billy - Alice McDermott
      - The Professor and the Madman - Simon Winchester
      - Staggerford - Jon Hassler
      - A Widow for One Year - John Irving
      - The Trumpet of the Swan - E. B. White

    - Rekomendasi 10 Buku Teratas:
      - Postmarked Yesteryear - Pamela E. Apkarian-Russell
      - Uncle John's Supremely Satisfying Bathroom Reader
      - The Hobbit - J. R. R. Tolkien
      - Betsy and Tacy Go Downtown - Maud Hart Lovelace
      - Our Bodies Ourselves For The New Century
      - Goodnight Moon Board Book - Margaret Wise Brown
      - Dreams of Childhood
      - Dilbert: A Book of Postcards
      - Natural California
      - Postcards from Live and Learn and Pass It On

Sistem rekomendasi yang dibangun memanfaatkan pendekatan Collaborative Filtering berbasis Neural Network dengan embedding user dan buku. Model telah dilatih menggunakan data rating yang dinormalisasi dan dievaluasi menggunakan RMSE. Sistem dapat menghasilkan Top-N Recommendation yang dipersonalisasi untuk setiap pengguna berdasarkan pola preferensi kolektif.

## Evaluation
Grafik evaluasi model menunjukkan perbedaan yang cukup jelas antara RMSE pada data latih dan data validasi. RMSE pada data latih terus menurun, sedangkan RMSE pada data validasi mengalami kenaikan perlahan setelah sekitar epoch ke-15. Hal ini mengindikasikan terjadinya overfitting, di mana model terlalu menyesuaikan terhadap data pelatihan dan kehilangan kemampuan generalisasi pada data validasi.

Meskipun demikian, nilai RMSE pada data validasi masih berada pada kisaran yang dapat diterima (sekitar 0.18), sehingga model tetap dapat digunakan untuk merekomendasikan buku dengan tingkat kesalahan prediksi yang relatif rendah.

Sebagai tindak lanjut, diperlukan pendekatan seperti early stopping, dropout, atau regularisasi L2 untuk mengurangi overfitting dan meningkatkan kemampuan generalisasi model.

**Kesimpulan:**
Proyek ini berhasil membangun sebuah sistem rekomendasi buku menggunakan pendekatan Collaborative Filtering berbasis Neural Network Embedding. Model dilatih menggunakan data rating antara pengguna dan buku, dengan pendekatan pemetaan vektor embedding untuk merepresentasikan hubungan antara keduanya.

Hasil pelatihan menunjukkan bahwa model mampu mencapai RMSE sekitar 0.15 pada data pelatihan dan sekitar 0.18 pada data validasi, yang menunjukkan performa prediksi yang cukup baik. Namun, pola perbedaan metrik antara data latih dan validasi menunjukkan adanya overfitting, sehingga perbaikan lebih lanjut seperti penggunaan early stopping atau regularisasi dapat dipertimbangkan.

Secara keseluruhan, sistem ini mampu memberikan rekomendasi buku yang cukup relevan berdasarkan pola interaksi pengguna sebelumnya, dan dapat dikembangkan lebih lanjut ke arah hybrid filtering atau integrasi konten untuk meningkatkan akurasi dan personalisasi.

