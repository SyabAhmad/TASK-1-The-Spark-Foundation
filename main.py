# task #1
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv("http://bit.ly/w-data")

regressor = LinearRegression()

X = data.iloc[: ,:-1]
Y = data.iloc[:, 1]

model = regressor.fit(X,Y)

hours = [[9.25]]
score = regressor.predict(hours)
print(score)

# to test the prediction
xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.2, random_state=0)
yPred = regressor.predict(xTrain)
print(yPred)







