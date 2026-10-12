
from sklearn.linear_model import LinearRegression
import numpy as np

# Input data: Hours studied
X = np.array([[1], [2], [3], [4], [5]])

# Output data: Marks scored
y = np.array([20, 30, 40, 50, 60])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks for 6 hours of study
predicted_marks = model.predict([[6]])

print("Simple Linear Regression")
print("------------------------")
print("Hours Studied: 6")
print("Predicted Marks:", predicted_marks[0])
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
