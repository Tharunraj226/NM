import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sys  # Added to allow safe exit

# Load the employee dataset
try:
    data = pd.read_csv('employee_data.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: employee_data.csv not found. Please check the file location.")
    sys.exit(1)  # Exit the script if file not found

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Calculate productivity = tasks_completed / work_hours
data['productivity'] = data['tasks_completed'] / data['work_hours']

# -----------------------------
# Visualization: Productivity Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['productivity'], kde=True, bins=20)
plt.title('Distribution of Employee Productivity')
plt.xlabel('Productivity (Tasks per Hour)')
plt.ylabel('Frequency')
plt.show()

# -----------------------------
# Visualization: Monthly Productivity Trend
monthly_productivity = data.groupby(data['date'].dt.to_period('M')).agg({'productivity': 'mean'}).reset_index()

plt.figure(figsize=(12, 6))
plt.plot(monthly_productivity['date'].dt.to_timestamp(), monthly_productivity['productivity'], marker='o')
plt.title('Monthly Average Productivity Trends')
plt.xlabel('Month')
plt.ylabel('Average Productivity')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# -----------------------------
# Machine Learning: Predict Productivity
X = data[['work_hours', 'tasks_completed']]
y = data['productivity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Model Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Productivity')
plt.xlabel('Actual Productivity')
plt.ylabel('Predicted Productivity')
plt.grid(True)
plt.show()
