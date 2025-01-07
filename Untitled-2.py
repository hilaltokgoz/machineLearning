
import pandas as pd
from sklearn.linear_model import LinearRegression

#Preprocessing (Ön Hazırlık)

df = pd.read_csv('audi.csv')
print(df) # ilk 3 satırı yazdırır
# bir aracın fiyatını tahmin etmek için model oluşturacağız.
# y = fiyat independent variable
# x = model yaratmak için kullanacağımız veriler(year, type, fuel vs.) dependent variable
# x için bazeı değerler kaldırılmalıdır. (index, PPYRank, score, PriceRank, MileageRank, )

df.drop(columns=['index','href', 'MileageRank','PriceRank','PPYRank','Score'], inplace=True) # sütunları siler
print(df.head(3)) # ilk 3 satırı yazdırır

#arabanın tipi, yakıt türü, vitesi gibi veriler matematiksel işlem yapamaz. Bunlar katagorisel verilerdir.
#PPY,Price gibi veriler ise sayısal(numeric) verilerdir.
# kategorik verileri sayısal verilere dönüştürmek için One Hot Encoding kullanılır.

df.info() #veri tiplerini gösterir.

df.columns = ['yil', 'kasa', 'mil', 'motor', 'ps', 'vites', 'yakit', 'sahip', 'fiyat', 'ppy'] # sütun isimlerini değiştirir.
print(df.head(3)) # ilk 3 satırı yazdırır

df.info() #kasa,motor,vites ve yakıt : object onları sayısal verilere dönüştürmemiz gerekiyor.
df['motor'] = df['motor'].str.replace("L","") #L harfini siler. hala kategorik veri
df['motor'] = pd.to_numeric(df['motor']) # motor sütununu sayısal veriye dönüştürür.

#df= pd.get_dummies(df,columns=['kasa','vites','yakit']) # kasa,vites,yakit sütununu sayısal veriye dönüştürür. boolean veriye dönüştürür.
#Tabloda 0 ve 1 değerleri de gösterilir. 1 olan değerler o kategoride olduğunu gösterir.drop_first ile secenepğin ilki kaldırılır.
df= pd.get_dummies(df,columns=['kasa','vites','yakit'],drop_first=True) 
df.info()
#Herşey nümerik veriye dönüştü. Artık model oluşturabiliriz.

y=df['fiyat'] #bağımlı değişken
x=df.drop( "fiyat", axis=1) #bağımsız değişken. Fiyat hariç her şey.

lm = LinearRegression() #model oluşturuldu
model =lm.fit(x,y) #model eğitildi.
m= model.predict([[2017,30000,1.6,110,1,2600,0,1]]) #tahmin yapıldı.
print("Tahmin edilen fiyat değeri:",m) #[13145.9080419] sterlin cinsinden

score = model.score(x,y) #modelin doğruluk oranı
print("doğruluk değeri:",score) #0.9595295772839718, %96

