import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
df=pd.read_csv('spam.csv',encoding='latin-1')
df=df[['v1','v2']]
df.columns=['label','text']
vectoriser=CountVectorizer()
x=vectoriser.fit_transform(df['text'])
y=df['label'].map({'ham':0,'spam':1})
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=MultinomialNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#print('accuracy',accuracy_score(y_test,y_pred))
msg="Subject: Meeting Tomorrow at 10 AM | Body: Hi team, just a reminder we have a scheduled meeting tomorrow at 10 AM in the conference room. Please bring your updates."
vec_msg=vectoriser.transform([msg])
pred=model.predict(vec_msg)[0]
print("Result:", "spam" if pred==1 else "ham")