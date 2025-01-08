
#Hata Metrikleri
# MSE (Mean Squared Error), RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error), R2 (R Squared)

# MSE (Mean Squared Error) = 1/n * Σ(yi - y^i)^2
# RMSE (Root Mean Squared Error) = sqrt(1/n * Σ(yi - y^i)^2)
# MAE (Mean Absolute Error) = 1/n * Σ|yi - y^i|
# MAPE (Mean Absolute Percentage Error) = 1/n * Σ|yi - y^i| / yi

#yaşadığı bölgeye, cinsiyete bağlı olarak ne kadarlık harcama yapacağını tahmin etmeye çalışan bir modelin oluşturulacak
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error #hata metrikleri

df = pd.read_csv("insurance.csv")
print(df.head(3))

df = pd.get_dummies(df, columns = ['sex','smoker','region'],drop_first=True)
df.info()

y = df['charges'] #harcama, bağımlı değişken
x = df.drop('charges', axis=1) #harcama hariç herşey, bağımsız değişkenler
lm = LinearRegression() #model oluşturuldu
model = lm.fit(x,y) #model eğitildi.
model.score(x,y) #modelin doğruluk oranı

m = model.predict([[19,26,0,1,1,0,0,1]]) #tahmin yapıldı.
print("Tahmin edilen harcama değeri:",m) # [24517.93110714]

df_hata = pd.DataFrame() #hata metriklerini hesaplamak için dataframe oluşturuldu.
df_hata['gercek_y'] = y #gerçek y değerleri
print(df_hata.head(3))

df_hata['tahmin_y'] = model.predict(x) #tahmin edilen y değerleri
print(df_hata.head(3))


df_hata['fark'] = df_hata['gercek_y'] - df_hata['tahmin_y'] #hata hesaplandı, noktaların doğruya olan uzaklığı
print(df_hata.head(3))

#MSE (Mean Squared Error) = hataların karesinin ortalaması
df_hata['hata_kare'] = df_hata['fark']**2 #hataların karesi
print(df_hata.head(3))

#MAE (Mean Absolute Error) = hataların mutlak değerlerinin ortalaması
df_hata['hata_mutlak'] = np.abs(df_hata['fark']) #hataların mutlak değerleri
print(df_hata.head(3))

#MAPE (Mean Absolute Percentage Error) = hataların mutlak değerlerinin y değerlerine oranının ortalaması
df_hata['hata_oran'] = np.abs(df_hata['fark']) / df_hata['gercek_y'] #hataların oranları
print(df_hata.head(3))

print(df_hata.mean()) #hataların ortalamaları

###########################################################
########### KISA YOL ######################################
#from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error #hata metrikleri import ile.

mse = mean_squared_error(df_hata['gercek_y'], df_hata['tahmin_y']) #MSE
print('MSE:',mse) #36501893.00741544

mae= mean_absolute_error(df_hata['gercek_y'], df_hata['tahmin_y']) #MAE
print('MAE:',mae) #4170.8868941635865

mape = mean_absolute_percentage_error(df_hata['gercek_y'], df_hata['tahmin_y'])
print('MAPE:',mape) #0.42035268473727033
