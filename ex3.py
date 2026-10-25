import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("student_data.csv")
print("First five records:")
print(data.head())

# Separate input features and target
X = data[["Study_Hours", "Attendance",
          "Assignment_Score", "Previous_Score"]]
y = data["Result"]

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Bagging using Random Forest
bagging_model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

bagging_model.fit(X_train, y_train)
bagging_prediction = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_prediction)

# Boosting using AdaBoost
boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)

boosting_model.fit(X_train, y_train)
boosting_prediction = boosting_model.predict(X_test)
boosting_accuracy = accuracy_score(y_test, boosting_prediction)

# Display accuracy
print(f"Bagging Accuracy: {bagging_accuracy * 100:.2f}%")
print(f"Boosting Accuracy: {boosting_accuracy * 100:.2f}%")

# Display the comparison graph
plt.bar(["Bagging", "Boosting"],
        [bagging_accuracy * 100, boosting_accuracy * 100])

plt.xlabel("Ensemble Method")
plt.ylabel("Accuracy (%)")
plt.title("Bagging vs Boosting")
plt.ylim(0, 100)
plt.show()