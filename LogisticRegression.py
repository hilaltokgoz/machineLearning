#Logistic Regression
#cinsiyet, kredi kartı limiti, eğitim, son ödeme yapmış mı gibi verilere göre 
#bir sonraki ay kredi kartını ödeyecek mi ödemeyecek mi tahmin etmek için 
#logistic regression kullanacağız.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #grafik çizdirme
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #veriyi train ve test olarak ayırma

df = pd.read_csv("card.csv")
print(df.head(3))

#id sütunu gereksiz, çıkartılacak.
df = df.drop('ID', axis=1)

y = df['default.payment.next.month'] #tahmin edilmesini istediğimiz sütun
x = df.drop('default.payment.next.month', axis=1) #bağıumsız değişkenler

print(df.shape) #(30000, 24), 30000 satır 24 sütun
#verilerin bir kısmını ezberlememesi için vermiyoruz. Bu yüzden verileri train ve test olarak ayırıyoruz.

#veriyi train ve test olarak ayırma
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.77, random_state=6) #random_state: verilerin karıştırılması için

log = LogisticRegression() #model oluşturma
model = log.fit(x_train, y_train) #modeli eğitme
score = model.score(x_test, y_test) #doğruluk oranı
print("score:",score) #0.7739130434782608, ½77 oranında bildi.

denemex = np.array(x.iloc[1903]) #1903. satırı tahmin et
print(model.predict([denemex]) ) #[0], ödemeyecek.

print(y.iloc[1903]) #0, ödemeyecek.-> Doğru tahmin etti

