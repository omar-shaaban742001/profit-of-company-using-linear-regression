#!/usr/bin/env python
# coding: utf-8

# In[227]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os 


# In[228]:


#importing data 
companies = pd.read_csv("E:\python_for_ai\python projects\linear regression\profit of company using linear regression/1000_Companies.csv")
X  = companies.iloc[: , :-1].values
Y = companies.iloc[:, 4].values

companies.head()


# In[229]:


X


# In[230]:


type(X)


# In[231]:


companies.dtypes


# In[232]:


#data visualisation
sns.heatmap(companies.corr() , annot= True)


# In[233]:


from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X = np.array(columnTransformer.fit_transform(X))
X = X[:, 1:]
X = pd.DataFrame(X)


# In[250]:


#Spliting data
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split (X, Y , test_size = 0.3, random_state=40) 


# In[251]:


#traning model 
from sklearn.linear_model import LinearRegression
model_fit = LinearRegression()
model_fit.fit(X_train , y_train)
X_train.shape


# In[252]:


X_train.head()


# In[253]:


#predicting the data set 
y_pred = model_fit.predict(X_test)
y_pred


# In[254]:


#calculating the coefs
model_fit.coef_


# In[255]:


#calculating intercept
model_fit.intercept_


# In[256]:


#calculating the error
from sklearn.metrics import r2_score
r2_score(y_pred,y_test)


# In[257]:


#plot final resutl 
sns.regplot(x=y_pred , y = y_test)
plt.xlabel("predicted price")
plt.ylabel("actual price")
plt.title("Actual vs predicted price")
plt.show()

