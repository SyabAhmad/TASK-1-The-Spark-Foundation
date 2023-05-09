# Task #1

### First we will import all the necessary imports 
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```
### And will load the data also
```
data = pd.read_csv("http://bit.ly/w-data")
```


### To create a linear regression model
```
regressor = LinearRegression()
```
### We will assign rows and columns to each X and Y

```
X = data.iloc[: ,:-1]
Y = data.iloc[:, 1]
```

### Then we will fit the X and Y in the Regression model

```
model = regressor.fit(X,Y)
```
### Then here we predict the score as hours are given

```
hours = [[9.25]]
score = regressor.predict(hours)
print(score)
```
### Here we have done a simple prediction on our test data, waht we want to do here is we check that is our prediction working properly or not
# To test the prediction
```
xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.2, random_state=0)
yPred = regressor.predict(xTrain)
print(yPred)
```

### So prediction working properly

### Now this code juts show how to do visualisation of the Given Data using matplotlib.pyplot
```python
data.plot()
plt.title("Hours and Score Prediction || The Spark Foundation || Task #1")
plt.xlabel("Score")
plt.ylabel("Hours")
plt.show()
```



## Task#1 Completed Successfully






