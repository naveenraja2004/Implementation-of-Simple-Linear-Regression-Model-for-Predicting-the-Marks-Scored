# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: NAVEEN RAJA N R

RegisterNumber: 212222230093
 
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
print(df)
print()
df.head(0)
df.tail(0)
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
print("RMSE= ",rmse)
*/
```

## Output:
### Dataset
![WhatsApp Image 2024-08-28 at 08 59 19_d18ec941](https://github.com/user-attachments/assets/fd64a46e-2557-4d61-82ba-3a05c610545a)

### Head Values
![WhatsApp Image 2024-08-28 at 08 59 23_499e79f8](https://github.com/user-attachments/assets/a1581322-7c6e-4c4b-b7eb-6ab39210f7a7)

### Tail Values
![WhatsApp Image 2024-08-28 at 08 59 17_51bae5e7](https://github.com/user-attachments/assets/271688dd-ec76-49c4-af04-5fb81e76a990)

### X and Y values
![WhatsApp Image 2024-08-28 at 08 59 18_4cf4e98c](https://github.com/user-attachments/assets/f7443fcf-1e02-464d-a0f0-c155945e48af)


### Predication values of X and Y
![WhatsApp Image 2024-08-28 at 08 59 18_c3f2daa3](https://github.com/user-attachments/assets/2dc8d606-c5d5-4970-92ad-114bb797e96a)


### MSE,MAE and RMSE
![WhatsApp Image 2024-08-28 at 08 59 23_d38b4849](https://github.com/user-attachments/assets/64d1140d-b76f-42dc-abd6-8637eaefa5e0)


### Training Set
![WhatsApp Image 2024-08-28 at 08 59 14_ea7f21de](https://github.com/user-attachments/assets/07391813-a8c3-44bd-8818-ad7ee3d7abaa)


### Testing Set
![WhatsApp Image 2024-08-28 at 08 59 21_4e0ace07](https://github.com/user-attachments/assets/798975e1-6f0e-4abc-bfa2-a59473315099)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

