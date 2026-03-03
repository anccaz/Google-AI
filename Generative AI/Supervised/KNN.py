# K-Nearest Neighbors example (classification task)
# predicts a discrete category (flower species)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data (Iris dataset is built-in)
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42)

# 3. Create and Train the Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)  # The model learns from the labeled data

# 4. Make Predictions
predictions = knn.predict(X_test)

# 5. Evaluate the model
print('Accuracy Score:', accuracy_score(y_test, predictions))
