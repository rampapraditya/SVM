import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
import joblib

# Memuat dataset dari file CSV
dataset = pd.read_csv('MAXMINZEDIT.csv')

# Memisahkan fitur (X) dan label (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Melatih model SVM dengan kernel RBF menggunakan one-vs-one
svm_classifier = OneVsOneClassifier(SVC(kernel='rbf', gamma='scale', random_state=42))
svm_classifier.fit(X_train, y_train)

print(X_test)
print(X_test.shape)
print()
print(y_test)
print(y_test.shape)

# Simpan model dan scaler
# joblib.dump(svm_classifier, 'svm_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')

# Function to get input from the keyboard and predict
def predict_from_input():
    try:

        # Input should be a space-separated string of numbers
        user_input = input("Enter four features separated by spaces: ")
        features = np.array([float(x) for x in user_input.split()]).reshape(1, -1)

        # Predict using the trained model
        prediction = svm_classifier.predict(features)

        # Output the prediction
        print(f"Predicted class: {prediction[0]}")
    except ValueError:
        print("Invalid input. Please enter four numerical values separated by spaces.")


# Example usage
if __name__ == "__main__":
    while True:
        predict_from_input()
        cont = input("Do you want to predict another sample? (yes/no): ").strip().lower()
        if cont != 'yes':
            break