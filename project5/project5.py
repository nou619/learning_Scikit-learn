import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data=pd.read_csv("mushrooms.csv")
print(data.head())
x=pd.get_dummies(data.drop("class",axis=1))
y=data["class"].map({'e':0,'p':1})
x_train ,x_test, y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=19)
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print('accuracy is : ',accuracy_score(y_predict, y_test))