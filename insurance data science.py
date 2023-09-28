#step 1- import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#step 2 import datasets
data=pd.read_csv('insurance.csv')
#step 3- prepocess the data
data.shape
data.head()
data.head(10)
data.tail()
data.tail(7)
data.columns
data.info()
data.describe()
data.isna().sum()
#############################################
#plotting
sns.heatmap(data.corr(),annot=True)
#age versus charges
sns.scatterplot(x=data['age'],y=data['charges'])
sns.scatterplot(x=data['bmi'],y=data['charges'])
#gender versus charges
sns.boxplot(x=data['sex'],y=data['charges'])
#children versus charges
sns.boxplot(x=data['children'],y=data['charges'])
#smoker versus charges
sns.boxplot(x=data['smoker'],y=data['charges'])
#region versus charges
sns.boxplot(x=data['region'],y=data['charges'])

columns=['sex','smoker','region']
###############################################
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
for column in columns:
    data[column]=encoder.fit_transform(data[column])
     
    
###########################################
#step4-seggregate 
x=data.drop(['charges'],axis=1)
y=data['charges']
x_train=x_train.loc[:,['age','sex','bmi','children','smoker','region']]

x_test=x_test.loc[:,['age','sex','bmi','children','smoker','region']]

#############################################
#step5-split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

###################################################
#step6-create a model
from sklearn.linear_model import LinearRegression
regressor1=LinearRegression()
regressor1.fit(x_train,y_train)

#step7-get model prdicted value
regressor1.coef_
regressor1.intercept_
y_pred1=regressor1.predict(x_test)

######################################
#step8-evaluate the modal performance by using appropriate metrics regression
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test, y_pred1))

metrics.mean_absolute_error(y_test, y_pred1)
metrics.r2_score(y_test,y_pred1)






