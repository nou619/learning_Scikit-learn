import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Step 2: Explore data ( just to peek)
print(train_data.info())        # check column types and missing values
print(train_data.describe())    # summary stats for numeric columns
print(train_data.isnull().sum()) # count missing values per column

# Step 3: Data Cleaning
# 3a: Fill missing 'Age' and 'fare' with median
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# 3b: Fill missing 'Embarked' with most common value (mode)
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])

# 3c: Drop columns we won't use for simplicity
train_data = train_data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
test_passenger_ids = test_data['PassengerId']  # save for submission
test_data = test_data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)

# Step 4: Encode categorical columns
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

train_data['Embarked'] = train_data['Embarked'].map({'C':0, 'Q':1, 'S':2})
test_data['Embarked'] = test_data['Embarked'].map({'C':0, 'Q':1, 'S':2})

# Step 5: Split features and target
X = train_data.drop('Survived', axis=1)  # features
y = train_data['Survived']               # target

# Step 6: Split internal train/test (20% for testing)
X_train, X_internal_test, y_train, y_internal_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 8: Evaluate on internal test set
y_pred = model.predict(X_internal_test)
print("Accuracy on internal test set:", accuracy_score(y_internal_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_internal_test, y_pred))

# Step 9: Predict on Kaggle test set
kaggle_predictions = model.predict(test_data)

# Step 10: Create submission file
submission = pd.DataFrame({
    "PassengerId": test_passenger_ids,
    "Survived": kaggle_predictions
})
submission.to_csv("submission3.csv", index=False)
print("Submission file created! 🚀")
