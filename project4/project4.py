import pandas as pd #to read the dataset
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