import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"D:\NIT\DECEMBER\12 DEC(multiple LR)\12th\MLR\Investment.csv")

X = dataset.iloc[:, :-1]

y = dataset.iloc[:, 4]


X=pd.get_dummies(X,dtype=int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split( X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,1]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
