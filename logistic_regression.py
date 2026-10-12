
from sklearn.linear_model import LogisticRegression
import numpy as np

# Input data: Hours studied
X = np.array([[1], [2], [3], [4], [5], [6]])

# Output data: Pass (1) or Fail (0)
y = np.array([0, 0, 0, 1, 1, 1])

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Predict result for a student who studied 4 hours
prediction = model.predict([[4]])

print("Logistic Regression")
print("-------------------")
print("Hours Studied: 4")

if prediction[0] == 1:
    print("Predicted Result: Pass")
else:
    print("Predicted Result: Fail")

print("Prediction Probability:", model.predict_proba([[4]])[0])
