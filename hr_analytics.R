
# HR Analytics: Employee Demand Prediction

# Sample historical employee data
month <- c(1, 2, 3, 4, 5, 6)
employees_needed <- c(20, 22, 21, 25, 27, 30)

# Create data frame
hr_data <- data.frame(month, employees_needed)

# Build Linear Regression Model
model <- lm(employees_needed ~ month, data = hr_data)

# Predict employees required for next month
next_month <- data.frame(month = 7)
prediction <- predict(model, newdata = next_month)

# Display output
cat("HR Analytics - Employee Demand Prediction\n")
cat("-----------------------------------------\n")
cat("Current Month: 6\n")
cat("Next Month: 7\n")
cat("Predicted Employees Required:",
    round(prediction), "\n")
