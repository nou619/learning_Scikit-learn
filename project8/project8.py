from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
emails = [
    "Win a free iPhone now",
    "Limited offer, click here",
    "Meeting at 10 am tomorrow",
    "Lunch with friend today",
    "Congratulations, you won cash",
    "Project deadline is Monday"
]
labels = [1, 1, 0, 0, 1, 0]  # 1=spam, 0=not spam
vectorizer=CountVectorizer()
X = vectorizer.fit_transform(emails)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.9, random_state=42)
nb = MultinomialNB()
nb.fit(X, labels)
y_pred = nb.predict(X_test)
print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
