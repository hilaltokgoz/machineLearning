
#Underfitting, az öğrenme
#Overfitting aşırı öğrenme
#BalancedFitting, dengeli öğrenme

import seaborn as sns #veri görselleştirme kutuphanesi, hazır veri setleri ile çalışır.
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #veriyi train ve test olarak ayırır.

df = sns.load_dataset("diamonds") #diamonds veri setini yükler
print(df) 

df = pd.get_dummies(df, columns = ['cut','color','clarity'],drop_first=True) #kategorik verileri sayısal verilere dönüştürür.
print(df.head(3))

y = df['price'] #bağımlı değişken
x = df.drop('price', axis=1) #bağımsız değişkenler

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.76, random_state=13) #bilginin ne kadarı verilecek ne kadarı test edilecek. #random_state=13 veri her çalıştırıldığında aynı şekilde bölünür.
lm = LinearRegression() #model oluşturuldu
model = lm.fit(x_train,y_train) #model eğitildi. fit öğrenmeye yarar o yüzden train kullanıldı
test_score = model.score(x_test,y_test) #modelin doğruluk oranı
print(test_score) #0.9165060308161193

train_score = model.score(x_train,y_train) #modelin doğruluk oranı, öğrenmeye verdiklerininde doğruluk oranı hesaplanır.
print(train_score) #0.9207668162774053

#train_score ve test_score arasındaki farkın az olması gerekir. 
#Eğer train_score yüksek test_score düşükse overfitting aşırı öğrenme
#Eğer train_score düşük test_score yüksekse underfitting az öğrenme




