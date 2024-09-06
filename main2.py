import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Memuat dataset Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Membagi dataset menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Membuat model SVM dengan kernel RBF
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # C dan gamma bisa disesuaikan

# Melatih model
svm_model.fit(X_train, y_train)

# Membuat prediksi
print("======== X_test ========")
print(X_test)
print(type(X_test))


y_pred = svm_model.predict(X_test)

# Evaluasi model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Menentukan x_max dan x_min dari fitur-fitur di set pelatihan
x_max = X_train.max(axis=0)
x_min = X_train.min(axis=0)

# Standarisasi x_max dan x_min menggunakan scaler yang sudah dipasang
x_max_scaled = scaler.transform([x_max])
x_min_scaled = scaler.transform([x_min])

# Prediksi menggunakan model SVM
prediction_max = svm_model.predict(x_max_scaled)
prediction_min = svm_model.predict(x_min_scaled)

# Menampilkan hasil prediksi
print("\nPrediksi untuk x_max:", prediction_max)
print("Prediksi untuk x_min:", prediction_min)
