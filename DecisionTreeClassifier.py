#Decision Tree Classifier
#Overfitting olmaya meyilli bir modeldir.

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_graphviz #export_graphviz: görselleştirme için
import graphviz #görüntüleme için
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split



# Veri yükleme
df = pd.read_csv("heart.csv")

# Tüm sütunları göster
pd.set_option('display.max_columns', None)
print(df.head(3))

# Kategorik değişkenleri seç
categorical_columns = ['Gender', 'Physical_Activity_Level', 'Stress_Level', 
                       'Chest_Pain_Type', 'Thalassemia', 'ECG_Results']

# Kategorik sütunları one-hot encode yap
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Heart_Attack_Risk hedef değişkenini label encoding ile dönüştür
label_encoder = LabelEncoder()
df['Heart_Attack_Risk'] = label_encoder.fit_transform(df['Heart_Attack_Risk'])

# Bağımlı ve bağımsız değişkenleri ayır
y = df['Heart_Attack_Risk']  # Hedef değişken
x = df.drop('Heart_Attack_Risk', axis=1)  # Bağımsız değişkenler

# Modeli oluştur
tree = DecisionTreeClassifier(random_state=42)
model = tree.fit(x, y)
score = model.score(x, y)
print("Model score:", score) #1.0, %100 doğru tahmin etti.


#Train ve test olarak ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70, random_state=16)
model = tree.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("Train Model score:", score) #0.3802, %30 doğru tahmin etti.

#Train öncesi %100, sonrası %30 doğru tahmin etti. Overfitting(verileri ezberlemiş) var.

