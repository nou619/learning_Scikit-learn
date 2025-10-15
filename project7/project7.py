import pandas as pd
data=pd.read_csv("dataset.csv")
 #print(data.isnull().sum())
#TotalCharges → 11 missing values 
#Everything else → 0 missing
data['TotalCharges'] = data['TotalCharges'].fillna(0)
print(data.isnull().sum())
