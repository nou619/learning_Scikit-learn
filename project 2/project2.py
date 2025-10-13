from sklearn.linear_model import LogisticRegression
x=[[0.5],[1],[4],[3.5],[5],[6]]#how many hours studied
y=[0,0,0,0,1,1]#pass or fail
model=LogisticRegression()#predicts categories not numbers
model.fit(x,y)
newstudents=[[3],[0.5],[7],[5],[6],[5.30]]
prediction=model.predict(newstudents)
result=""
for i,hours in enumerate(newstudents):
    result='pass' if prediction[i]==1 else result=='fail'
    print(f"Student studying {hours[0]} hours: {result}")
