# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thaanesh.V
RegisterNumber:  212223230228
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)

print(df.head())

print(df.tail())

x = df.iloc[:,:-1].values
print(x)

y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

print(y_pred)

print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:

## Dataset:

![Screenshot (400)](https://github.com/user-attachments/assets/c03e1a66-1b3f-41b4-9bbc-6ac434c40726)

## Head Values:

![Screenshot (402)](https://github.com/user-attachments/assets/5fa61611-0579-4c2f-9d97-337a99c02b9f)

## Tail Values:

![Screenshot (402)](https://github.com/user-attachments/assets/b0add2e6-a4ee-49ae-94b2-1caebeaed20c)

## X and Y Values:

![Screenshot (403)](https://github.com/user-attachments/assets/750cdee8-5569-4e72-b7c6-30deae9a378c)

## Predication Value of X and Y:

![Screenshot (404)](https://github.com/user-attachments/assets/c552d5ac-3d3f-4a6a-ace4-f0b579436b35)

## MSE,MAE and RMSE:

![Screenshot (405)](https://github.com/user-attachments/assets/027b1ac2-3852-477d-a0b4-f7f6c9e80021)

## Training Set:

![Screenshot (406)](https://github.com/user-attachments/assets/026fff1b-82f9-41d5-a7ee-9da822bec803)

## Testing Set:

![Screenshot (407)](https://github.com/user-attachments/assets/41080251-80e2-4111-9a5c-6f4b7eba31d2)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
