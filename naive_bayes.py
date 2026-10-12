
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Input data: Hours studied
X = np.array([[1], [2], [3], [4], [5], [6]])

# Output: Fail (0) or Pass (1)
y = np.array([0, 0, 0, 1, 1, 1])

# Create and train the model
model = GaussianNB()
model.fit(X, y)

# Predict for a student who studied 5 hours
prediction = model.predict([[5]])

print("Naive Bayes Classification")
print("--------------------------")
print("Hours Studied: 5")

if prediction[0] == 1:
    print("Predicted Result: Pass")
else:
    print("Predicted Result: Fail")
