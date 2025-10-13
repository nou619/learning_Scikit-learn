import pandas as pd #to read the dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv("Iris.csv")
#learning about head and tail
print('first 5 rows')
print(data.head())
print('first  row')
print(data.head(1))
print('last 5 rows')
print(data.tail())
print('last  2 rows')
print(data.tail(2))
#axis (1 =column, 0=row)
#we droped species cuz theyre the label
x=data.drop("Species", axis=1).values
y=data["Species"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
# random state :Python, shuffle my flowers but do it according to recipe #5
model=DecisionTreeClassifier(random_state=5)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print("accuracy",accuracy_score(y_predict,y_test))

