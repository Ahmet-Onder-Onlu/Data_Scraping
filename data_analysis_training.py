import pymysql
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# MySQL veritabanı bağlantı bilgileri
host = 'localhost'  # MySQL sunucusunun adresi
user = 'root'  # MySQL kullanıcı adı
password = 'Ahmet.5ond'  # MySQL parolası
database = 'Training'  # Veritabanı adı

# MySQL bağlantısı oluşturma
connection = pymysql.connect(
    host=host,
    user=user,
    password=password,
    database=database
)

with open("petlebi_fetching.sql", "r") as sql_file:
    sql_query = sql_file.read()



# Sorguyu çalıştırıp sonuçları bir DataFrame'e yükleme
df = pd.read_sql(sql_query, connection)

# Bağlantıyı kapatma
connection.close()

#print(df)

"""print(df.info())

print(df.describe())

sns.pairplot(df)
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x='marka', y='fiyat', data=df)
plt.title('Marka ve Fiyat İlişkisi')
plt.xlabel('Marka')
plt.ylabel('Fiyat')
plt.show()"""

"""X = df[['marka', 'model', 'donanim', 'motor', 'yakit', 'vites']]  # Bağımsız değişkenler
y = df['fiyat']  # Bağımlı değişken

# (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Eğitim ve test veri setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapalım
y_pred = model.predict(X_test)

# Hata ölçümü
print('Hata ölçümü:', mean_squared_error(y_test, y_pred))
# Eğitilen modelin katsayılarını ve intercept'ini inceleyelim
print('Model coefficients:', model.coef_)
print('Intercept:', model.intercept_)

r2 = r2_score(y_test, y_pred)

print(f'R^2 Score: {r2}')"""

# İyi bir sonuç alamadım

# Kategorik değişkenleri sayısal hale getirelim (One-Hot Encoding)
X = pd.get_dummies(df[['marka', 'model', 'vites', 'donanim', 'motor', 'yakit']], drop_first=True)
y = df['fiyat']

# Eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Model parametrelerini tanımlayalım
params = {
    'objective': 'reg:squarederror',  # Regresyon için hata kareler toplamını kullanalım
    'eval_metric': 'rmse'  # Hata ölçüsü olarak RMSE (kök ortalama kare hatası) kullanalım
}

# Modeli eğitelim
num_round = 100  # 100 turda eğitelim
model = xgb.train(params, dtrain, num_round)

# Test seti üzerinde tahmin yapalım
y_pred = model.predict(dtest)

# Model performansını değerlendirelim
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Eğitilen modelin katsayılarını inceleyelim (XGBoost'da katsayılar bulunmaz, ağırlıklar vardır)
feature_importances = model.get_score(importance_type='weight')
print('Feature Importances:', feature_importances)

# xgboost hata oranını azalttı. Farklı parametrelerle devam edilcek.
