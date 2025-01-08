#Ridge Regresyon (L2)
#Aşırı öğrenme(Overfitting) durumları için kullanılır.
#ridge regresyonda cezalar karesi ile orantılıdır.


#y = a0 + a1*x1 + a2*x2 + a3*x3 + ... + an*xn + b + alfa*(b1^2 + b2^2 + b3^2 + ... + bn^2)

#y=50, 50=40+10+0 # b: sabit
#y=50, 50=30+10+10 # b: sabit, alfa tarafını artırmak için katsayıalr (a) kısmını küçültür.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

df = pd.read_csv("student_scores.csv")
print(df.head(3))

#saate bağlı olarak skor tahmini yapılacak.
y = df['Scores']
x = df[['Hours']]

#grafik çizdirme
plt.style.use('fivethirtyeight') #grafik stili
plt.figure(figsize=(8,8)) #grafik boyutu
plt.scatter(x,y, color='red') #scatter plot
plt.show() #grafik gösterilir.

#model oluşturma
lr = LinearRegression()
model = lr.fit(x,y)
score = model.score(x,y)
print("score:",score) #0.9529481969048356, doğruluk oranı

#ridge regresyon
alfalar = [1,10,20,100,200]

for a in alfalar:
    r = Ridge(alpha=a) #alpha: ceza katsayısı
    model_ridge = r.fit(x,y)
    score_ridge = model.score(x,y)
    print("score_ridge:",score_ridge) #0.9529481969048356, doğruluk oranı
    print("katsayı:",model_ridge.coef_) #katsayılar

#alfa arttıkça katsayılar küçülür.
