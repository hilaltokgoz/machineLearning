
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #veriyi train ve test olarak ayırır.

#Preprocessing (Ön Hazırlık)

df = pd.read_csv('audi.csv')

df.drop(columns=['index','href', 'MileageRank','PriceRank','PPYRank','Score','PPY'], inplace=True) # sütunları siler
print(df.head(3)) # ilk 3 satırı yazdırır

df.columns = ['yil', 'kasa', 'mil', 'motor', 'ps', 'vites', 'yakit', 'sahip', 'fiyat'] # sütun isimlerini değiştirir.

df.info() #kasa,motor,vites ve yakıt : object onları sayısal verilere dönüştürmemiz gerekiyor.veri tiplerini gösterir.
df['motor'] = df['motor'].str.replace("L","") #L harfini siler. hala kategorik veri
df['motor'] = pd.to_numeric(df['motor']) # motor sütununu sayısal veriye dönüştürür.

df= pd.get_dummies(df,columns=['kasa','vites','yakit'],drop_first=True) # drop_first=True ile ilk sütun silinir.

#Herşey nümerik veriye dönüştü. Artık model oluşturabiliriz.

y=df['fiyat'] #bağımlı değişken
x=df.drop( "fiyat", axis=1) #bağımsız değişken. Fiyat hariç her şey.

#train ve test verilerini ayırma
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=13) #test_size=0.3 %70 train, bilginin ne kadarı verilecek ne kadarı test edilecek. #random_state=13 veri her çalıştırıldığında aynı şekilde bölünür.
#yani  her çalıştığında o 13 te  sonuç aynı olur.

lm = LinearRegression() #model oluşturuldu
model = lm.fit(x_train,y_train) #model eğitildi. fit öğrenmeye yarar o yüzden train kullanıldı

score = model.score(x_test,y_test) #modelin doğruluk oranı, score test ile yapılır
print("doğruluk değeri:",score) # 0.9023627111035176, testten önceki değere göre biraz düşük olabilir.

m= model.predict([[2016,30000,1.0,90,5,0,1]]) #tahmin yapıldı.
print("Tahmin edilen fiyat değeri:",m) #13892.11369605] sterlin cinsinden