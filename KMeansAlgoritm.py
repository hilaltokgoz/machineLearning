#K-Means Algoritm
#kümeleme işlemi gerçekleştirir.
#K-Means, Denetimsiz bir makine öğrenme yöntemidir.
#Gama ışın oranı: GR, Yığın yoğunluğu: RHOB, Nötron gözenekleri: NPHI,Fotoelektrik faktör:PEF, Akustik Sıkıştırma Yavaşlığı: DTC


import pandas as pd
import matplotlib.pyplot as plt #sonuçları görüntülemek için kullanılır
from sklearn.preprocessing import StandardScaler # Verileri ölçeklendirmek için kullanılır
from sklearn.cluster import KMeans #

df = pd.read_csv('force.csv',index_col= 'DEPTH_MD') #derinlik boyunca tamamla
print(df.head(3))

#NPHI sütununda veriler yok->NAND, o sütun kaldırılsın
df.dropna(inplace=True) #dropna NAN olan değerleri siler.
print(df.head(3))

#verileri dönüştürme, verileri standartlaştırma
print( df.describe()) #veri setinin istatiksel özelliklerini özetler (count ,mean,standart sapma vs.)

#model oluşturma 
scaler = StandardScaler()
                                  
df[["RHOB_T","NPHI_T","GR_T","PEF_T","DTC_T"]] = scaler.fit_transform(df[["RHOB","NPHI","GR","PEF","DTC"]]) #sütun normalizasyonu
print(df.head(3))

#k-means uygulama
def optimise_k_means(data, max_k):
    means = [] #küme sayısını tutan liste
    inertlias = [] #kümeye karşılık gelen inertiaları tutar.

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k) #model tanımlama ve k sayısı belirleme
        kmeans.fit(data) #eğit

        means.append(k) #küme sayısını listeye ekle
        inertlias.append(kmeans.inertia_) #hesaplanan inertia listeye ekle

#grafik ayarları
    fig = plt.subplots(figsize=(10,5)) #figür boyutu
    plt.plot(means,inertlias,'o-') #çizgi grafiği
    plt.xlabel('Küme Sayısı')
    plt.ylabel('Inertia') #kümeye olan uzaklık
    plt.grid(True) #grafiğe grid ekle
   # plt.show()


optimise_k_means(df[['RHOB_T', 'NPHI_T']], 10) #optimize küme sayısı belirlenmeye çalışıyor.

kmeans = KMeans(n_clusters=3) # 3 küme
kmeans.fit(df[['NPHI_T','RHOB_T']]) #kümeleme gerçekleştiriliyor
#KMeans(n_clusters=3)
df['kmeans_3'] = kmeans.labels_ #veriye her bir gözlem için küme sütunu ekle
print(df.head(3))

plt.scatter(x=df['NPHI'], y=df['RHOB'],c=df['kmeans_3'], cmap='viridis') #küme etiketlerine göre renklendir.
plt.xlim(-0.1,1) #sınırlar
plt.ylim(3,1.5)
plt.show()
