from sklearn.tree import DecisionTreeClassifier

# Features: [size, color] → size in cm, color coded 0=green, 1=yellow, 2=red
X = [
    [3, 0],  # small green
    [4, 1],  # medium yellow
    [5, 2],  # big red
    [2, 0],  # tiny green
    [4, 2]   # medium red
]

# Labels: 1 = sweet, 0 = not sweet
y = [0, 1, 1, 0, 1]

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X, y)

# Predict new fruits
new_fruits = [[3, 1], [5, 0], [4, 2]]  # medium yellow, big green, medium red
predictions = model.predict(new_fruits)

for i, fruit in enumerate(new_fruits):
    result = "Sweet" if predictions[i] == 1 else "Not Sweet"
    print(f"Fruit with size {fruit[0]} and color {fruit[1]}: {result}")
