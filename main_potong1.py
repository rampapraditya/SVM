import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
import joblib

# Memuat dataset dari file CSV
dataset = pd.read_csv('MAXMINY.csv')


# Memisahkan fitur (X) dan label (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Melatih model SVM dengan kernel RBF menggunakan one-vs-one
svm_model = OneVsOneClassifier(SVC(kernel='rbf', gamma='scale', random_state=42))
svm_model.fit(X_train, y_train)

# Simpan model dan scaler
joblib.dump(svm_model, 'svm_model_motor.pkl')
joblib.dump(scaler, 'scaler_motor.pkl')

print("Proses simpan selesai")