# Laporan Proyek Machine Learning - Haekal Hilmi Zain

## Domain Proyek
Berkembangnya teknologi berperan besar dalam kehidupan sehari hari, termasuk contohnya penggunaan Machine Learning untuk membantu manusia dalam menyelesaikan permasalahan yang mempunyai komputasi rumit. Dalam penggunaan Machine Learning kali ini kita akan melakukan prediksi harga rumah di kota Amsterdam.

Rumah merupakan kebutuhan yang diperlukan bagi manusia sebagai tempat tinggal. Dalam kebutuhan membeli rumah, beberapa aspek dapat menjadi pertimbangan untuk menentukan harga jual rumah. Dengan menggunakan teknologi untuk memprediksi harga rumah, orang dapat menghitung korelasi berbagai aspek rumah, yang dapat memberikan informasi tentang harga rumah berdasarkan keadaan.

Berdasarkan dataset, data latih Model Machine Learning yang mampu prediksi harga rumah di kota Amsterdam. Penulisa akan menyelesaikan permasalahan prediksi harga rumah dengan 3 model yaitu KNN, Random Forest dan Boost yang nantinya akan menghasiljkan prediksi harga rumah dengan akurasi tinggi berdasrkan data yang sudah di bagi menjadi data latih dan data uji.
## Business Understanding

### Problem Statements
- Membuat model untuk prediksi harga rumah di kota Amsterdam dengan akurasi tertinggi

### Goals
- Mengetehui model yang mempunyai akurasi tinggi untuk prediksi harga rumah di kota Amsterdam

    ### Solution statements
    - Melakukan proses *Exploratory Data Analysis* untuk melihat data yang berkolerasi dan memiliki pengaruh terhadap harga rumah.
    - Menggunakan model Machine Learning regresi. Untuk menemukan hasil prediksi harga rumah yang terbaik. Berikut model-model yang akan digunakan:
    - *Random Forest Regressor*
    - *K-Neighbors Regressor*
    - *AdaBoost Regressor*

## Data Understanding
Dataset yang digunakan untuk prediksi harga rumah diambil dari platfrom kaggle.com yang di publikasikan oleh THOMASNIBB. Dataset ini terdiri dari 1 file csv dan berisi data harga rumah yang berada di kota Amsterdam. Berikut akses datasetnya https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction. Dataset ini memiliki 924 data dan 8 kolum, dengan penejelasan kolumnya sebagai berikut:
- Rows : nomor data.
- Address : alamat rumah.
- Zip : kode pos.
- price : harga rumah.
- Area : area rumah dalam meter.
- Room : jumlah ruangan.
- Lon : koordinat bujur.
- Lat : koordinat lintang.

Tahap selanjutmnya melakukan *Exploratory Data Analysis* (EDA) yang bertujuan untuk menghilangkan outliers, serta menampilkan korelasi antar data baik data kategorikal maupun data numerik.

Berikut merupakan visualisasi boxplot dari data numerik dari Price, Area, Lon dan Lat.

![Visualisasi BoxPlot Price (JT)](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/1.png)

![Visualisasi BoxPlot Area](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/2.png)

![Visualisasi BoxPlot Lon](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/3.png)

![Visualisasi BoxPlot Lat](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/4.png)

Bisa kita lihat dari ke empat gambar bahwa semua semua fitur memiliki outliers. Oleh karena itu digunakan lah metode *Interquartile Range* (IQR) untuk mengatasi outliers. Yang hasilnya data tersebut akan direduksi dan dieleminasi untuk mengatasi outliers.

Proses selanjutnya melakukan *univariate analysis* untuk data kategorikal dan data numerik. 

![Visualisasi Data Kategorikal Room](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/5.png)

Dari gasdmbar diatas dapat disimpulkan bahwa pada fitur jumlah ruangan, data terbanyak yaitu total 3 ruangan dalam satum rumah dan paling sedikit 8 ruangan dalam satu rumah.

![Visualisasi Data Kategorikal district](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/6.png)

Dari gambar diatas dapat disimpulkan bahwa pada fitur district, data terbanyak yaitu pada district 6 yang berarti jumlah rumah pada daerah tersebut banyak lalu yang paling sedikit yaitu district 2.

Visualisasi data numerik dilakukan dengan menggunakan plot histogram.

![Visualisasi Data Numerik](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/7.png)

Dari gambar diatas, dapat ditarik kesimpulan, yaitu:
- Pada data "Price", data harga rumah kebanyakan terdapat direntang 200.000$ hingga 400.000$
- Distribusi data miring ke kanan (right skewed) yang dimana akan berdampak pada hasil prediksi model.

Tahap selanjutnya proses *multivariate analysis* untuk data kategorikal dan data numerik.

![Visualisasi Data Kategorikal](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/8.png)

Dari data diatas, dapat disimpulkan:
- Data pada Room (Jumlah Ruangan), jumlah Ruangan 10 memiliki nilai yang tinggi, sehingga dapat disimpulkan fitur room memiliki pengaruh dampak yang tinggi terhadap rata-rata harga.
- Data pada District (Komplek/Perumahan), dari 10 district/perumahan yang berbeda beda. Memiliki nilai dibawah jumlah ruangan (Room). Sehingga jumlah ruangan pada rumah memiliki dampak paling tinggi terhadap rata-rata harga rumah di amsterdam.

Pada data numerik, digunakan pairplot untuk melihat hubungan antara data fitur dan data target.

![Visualisasi Data Multivariate](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/9.png)

Dapat disimpulkan berdasarkan gambar diatas, bahwa fitur "Lon", "Lat" dan "Area" memiliki hubungan data yang positif dengan data "Price".

Serta terdapat juga heatmap yang bertujuan untuk memvisualisasikan korelasi antara fitur "Lon", "Lat" dan "Area" dengan data "Price" agar lebih mudah untuk dilihat dan dipahami.

![Visualisasi Data Matrix](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/10.png)

## Data Preparation
- Mengatasi outliers dengan menggunakan metode *Interquartile Range* (IQR) yang akan berdampak pada pengurangan data pada dataset.
- Melakukan one hot encoding pada data-data kategorikal dengan menggunakan fungsi *get_dummies* pada library Pandas, dimana data diubah menjadi bilangan biner.
- Train test split aja proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 924 dibagi menjadi 633 untuk data latih dan 159 untuk data uji.
- Melakukan standarisasi untuk fitur numerik agar menghasilkan nilai standar deviasi sama dengan 1 dan mean sama dengan 0. Standarisasi dilakukan agar memudahkan algoritma dalam melakukan komputasi perhitungan.

## Modeling

- Berikut penjelasan beberapa algoritma yang membantu dalam pembuatan model Machine Learning, dimana algoritma yang diambil merupakan algoritma bertipe regresi.
    - **Random Forest**, merupakan salah satu algoritma populer yang digunakan karena kesederhanaannya dan memiliki stabilitas yang baik. 
    - **AdaBoost**, merupakan singkatan dari Adaptive Boosting. Algoritma ini bertujuan untuk memberikan bobot lebih pada observasi yang tidak tepat atau disebut weak classification. 
    - **K-Neighbors Regressor**, K-Nearest Neighbour bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat. Proyek ini menggunakan [sklearn.neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_neighbors` = Jumlah k tetangga tedekat.. 

- Berikut merupakan tahapan pembuatan model dengan beberapa algoritma yang berbeda.
    1. Sebelum membuat model, dilakukan dulu pembuatan DataFrame yang akan diisi dengan hasil MSE data train dan data test pada setiap algoritma. 
    2. Selanjutnya, dilakukan pembuatan model Random Forest dengan melakukan import library pada sklearn.ensemble yang mengambil fungsi RandomForestRegressor. Setelah itu membuat model dengan diisikan beberapa parameter seperti n_estimators=150, max_depth=16, dan random_state=100.
    3. Pada algoritma Boosting, melakukan import library sklearn.ensemble yang mengambil fungsi AdaBoostRegressor. Digunakan beberapa parameter seperti n_estimators=50, learning_rate=0.001, dan random_state=100.
    4. Pada tahapan ini, dilakukan import library sklearn.neighbors yang mengambil fungsi KNeighborsRegressor. Pada algoritma K-Neighbors Regressor, digunakan parameter n_neighbors=13.
    *Catatan: pada nilai yang terdapat pada tiap parameter diisi dengan angka acak dimana dilakukan *trial dan error* beberapa kali hingga mendapatkan nilai MSE yang terkecil dari hasil tersebut.

## Evaluation
Metrik evaluasi yang digunakan pada proyek ini adalah akurasi dan mean squared error (MSE). Akurasi menentukan tingkat kemiripan antara hasil prediksi dengan nilai yang sebenarnya (y_test). Mean squared error (MSE) mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. Berikut formulan MSE

![MSE Formula](https://www.gstatic.com/education/formulas2/472522532/en/mean_squared_error.svg)

MSE	=	mean squared error
n	=	jumlah dataset
Yi	=	nilai sebenarnya
Ŷi	=	nilai prediksi

Berikut merupakan hasil dari MSE yang dilakukan oleh ketiga model Machine Learning.

![Visualisasi Data mse](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/11.png)

Dan berikut merupakan hasil akurasi dari tiga model Machine Learning.
![Visualisasi Data akurasi](https://raw.githubusercontent.com/haekalhlm/ML-TERAPAN/main/Gambar/13.png)


Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma Random Forest memiliki akurasi lebih tinggi tinggi dan tingkat error lebih kecil dibandingkan algoritma lainnya dalam proyek ini.



## Kesimpulan
Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma Random Forest memiliki akurasi lebih tinggi tinggi dan tingkat error lebih kecil dibandingkan algoritma lainnya dalam proyek ini.

