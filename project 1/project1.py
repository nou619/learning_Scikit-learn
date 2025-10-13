from sklearn.linear_model import LinearRegression
# (feature = size of house in 100s m², label = price in $1000s)
x = [[1], [2], [3], [4], [5]]   # sizes, its a feature that's why it's 2D
y = [150, 200, 250, 300, 350]   # prices , answers(labels) so it can be 1D
model=LinearRegression()
model.fit(x,y)#teach the model
new_house=[[6]]
predicted_price=model.predict(new_house)
print(f"predicted price:{ predicted_price[0]}k$")