# Supervised learning: labeled data to train a model

# Linear Regression example: linear correlation b/w data-points
# predicts a continuous numerical outcome (house price)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# import numpy as np

# 1. Load Data
# In a real scenario, you'd load a CSV file (e.g., house_data.csv)
# For this example, we'll use a hypothetical small dataset
data = {'SquareFootage': [1500, 1600, 1700, 1800, 1900, 2000],
        'Price': [300000, 320000, 340000, 360000, 380000, 400000]}
df = pd.DataFrame(data)

# 2. Define Features (X) and Target (y)
X = df[['SquareFootage']]  # Features (input)
y = df['Price']           # Target (output/label)

# 3. Split data into training and testing sets
# We train the model on the training data and evaluate it on the test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# 4. Create and Train the Model (Fit)
model = LinearRegression()
model.fit(X_train, y_train)  # The model learns relationship between X and y

# 5. Make Predictions
predictions = model.predict(X_test)

# 6. Evaluate the model (optional but recommended)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))

