import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load a sample dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM with RBF kernel
clf = SVC(kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)

print(X_test.shape)
print(y_test.shape)

# Function to get input from the keyboard and predict
def predict_from_input():
    try:
        # Input should be a space-separated string of numbers
        user_input = input("Enter four features separated by spaces: ")
        features = np.array([float(x) for x in user_input.split()]).reshape(1, -1)

        # Predict using the trained model
        prediction = clf.predict(features)

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