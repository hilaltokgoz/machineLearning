
from sklearn.linear_model import LinearRegression # Import the Linear Regression model
import pandas as pd # Import the pandas library

df= pd.read_csv('student.csv') # Read the data from the CSV file

print(df.head(3)) # Print the first 3 rows of the data
print(df) # Print the entire data

#kaç puan alacak o tahmin edilecek : y (dependent variable)
#kaç saat çalışmış, kaçıncı sınıf : x (independent variable)
#y = a1x1 + a2x2 + b

y= df['Marks'] # Dependent variable
x= df[['number_courses','time_study']] # Independent variable

df.info() #data hakkında bilgi verir

linearModel= LinearRegression() # Linear Regression modeli oluşturuldu
model = linearModel.fit(x,y) # Model eğitildi, x ve y verileri ile a ve b katsayıları bulundu.
m=model.predict([[3,5]]) # 3.sınıf ve 5 saat çalışma süresi için tahmin yapıldı.
print("Tahmin edilen Marks değeri:",m) # Tahmin sonucu yazdırıldı. //Tahmin edilen Marks değeri: [25.13169993]

max = df['Marks'].max() # En yüksek Marks değeri
print("En yüksek Marks değeri:",max) # En yüksek Marks değeri: 55.299

model.score(x,y) # Modelin doğruluk oranı
print("Modelin doğruluk oranı:",model.score(x,y)) # Modelin doğruluk oranı: 0.9403656320238896 //%94

df.columns = ['Sınıf', 'Çalışma Saati', 'Puan'] # Sütun isimlerini değiştirme
print(df) # Verilerin son hali yazdırıldı.

model.coef_ # Katsayıları verir.
print("Katsayılar:",model.coef_) # Katsayılar: Katsayılar: [1.86405074 5.39917879]

model.intercept_ # Sabiti verir.
print("Sabit:",model.intercept_) # Sabit(b): -7.456346231178355

s1 = model.predict([[3,4.508]])  
print(s1) #[22.47530397]alması bekleniyor tabloda 19.202, sapma var. 

(19.202-s1)/19.202 # Sapma oranı, %17
