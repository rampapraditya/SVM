from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "<h1>Layanan SVM</h1>";

@app.route('/latih', methods=['GET'])
def latih():
    dataset = pd.read_csv('MAXMINZEDIT.csv')

    # Memisahkan fitur (X) dan label (y)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalisasi fitur
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Melatih model SVM dengan kernel RBF menggunakan one-vs-one
    svm_classifier = OneVsOneClassifier(SVC(kernel='rbf', gamma='scale', random_state=42))
    svm_classifier.fit(X_train, y_train)

    print(X_test)
    print(X_test.shape)
    print()
    print(y_test)
    print(y_test.shape)

    # Simpan model dan scaler
    joblib.dump(svm_classifier, 'svm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return jsonify({'status' : 'sukses' })

@app.route('/echo', methods=['POST'])
def echo():
    data = request.get_json()
    max = data['max']
    min = data['min']

    return jsonify({'max': max + 1, 'min': (min + 1)})

@app.route('/predict', methods=['POST'])
def predict():
    # Muat model dan scaler
    model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')

    data = request.get_json()
    features = data['features']

    # Standarisasi fitur
    # features = scaler.transform([features])

    features1 = np.array([float(x) for x in features.split()]).reshape(1, -1)

    # Prediksi
    prediction = model.predict(features1)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)