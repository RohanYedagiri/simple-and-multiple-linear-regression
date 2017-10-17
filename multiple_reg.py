# -*- coding: utf-8 -*-
# multiple regression
import pandas as pd
#import numpy as np

# load dataset
dataset = pd.read_csv('50_Startups.csv')

# separate x & y columns
X = dataset.drop('Profit',axis=1)
y = dataset.Profit

# dummies for x dataset
X = pd.get_dummies(X,columns=['State'],drop_first=True)

# split the data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# fit the model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

# predict
y_pred = reg.predict(X_test)

# backward elimination
import statsmodels.formula.api as sm

# add column of ones to x dataset
X['ones'] = 1
X_opt = X   # contains all independant vars
reg_ols = sm.OLS(endog=y,exog=X_opt).fit()
reg_ols.summary() # State_New York P value is 0.990 > 0.05 , so remove it from X_opt


X_opt = X.drop('State_New York',axis=1)   # contains all independant vars
reg_ols = sm.OLS(endog=y,exog=X_opt).fit()
reg_ols.summary() #  State_Florida P value is 0.940 > 0.05 , so remove it from X_opt


X_opt = X.drop(['State_Florida','State_New York'],axis=1)   # contains all independant vars
reg_ols = sm.OLS(endog=y,exog=X_opt).fit()
reg_ols.summary() # Administration P value is 0.602 > 0.05 , so remove it from X_opt


X_opt = X.drop(['State_Florida','State_New York','Administration'],axis=1)   # contains all independant vars
reg_ols = sm.OLS(endog=y,exog=X_opt).fit()
reg_ols.summary() # Marketing Spend P value is 0.060 > 0.05 , so remove it from X_opt


X_opt = X.drop(['State_Florida','State_New York','Administration','Marketing Spend'],axis=1)   # contains all independant vars
reg_ols = sm.OLS(endog=y,exog=X_opt).fit()
reg_ols.summary()

'''
To conclude independent variable 'R&D Spend' is (in this case) a powerful predictor 
'''














