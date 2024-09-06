import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from matplotlib.colors import ListedColormap

# Memuat dataset dari file CSV
# dataset = pd.read_csv('vibration_data.csv')
dataset = pd.read_csv('MAXMINY.csv')


# Memisahkan fitur (X) dan label (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print()

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_test)
print(y_test)


# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Melatih model SVM dengan kernel RBF menggunakan one-vs-one
svm_classifier = OneVsOneClassifier(SVC(kernel='rbf', gamma='scale', random_state=42))
svm_classifier.fit(X_train, y_train)

# Memprediksi label untuk data latih dan data uji
y_pred_train = svm_classifier.predict(X_train)
y_pred_test = svm_classifier.predict(X_test)

# Evaluasi performa model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Menampilkan laporan klasifikasi dan matriks kebingungan
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

# Visualisasi Training Set
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, svm_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('lightcoral', 'moccasin', 'lightcyan')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('maroon', 'darkgoldenrod', 'darkslategray'))(i), label=j)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
L = plt.legend()
L.get_texts()[0].set_text('Normal Pump')
L.get_texts()[1].set_text('Impeller Problem')
L.get_texts()[2].set_text('Bearing Defect')
plt.show()

# Visualisasi Test Set
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, svm_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('lightcoral', 'moccasin', 'lightcyan')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('maroon', 'darkgoldenrod', 'darkslategray'))(i), label=j)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
L = plt.legend()
L.get_texts()[0].set_text('Normal Pump')
L.get_texts()[1].set_text('Impeller Problem')
L.get_texts()[2].set_text('Bearing Defect')
plt.show()