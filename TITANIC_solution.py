#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


df_initial=pd.read_csv(r'C:/Users/Jayant/Desktop/DATA-SCIENCE_JOURNEY/titanic/train.csv')
test_set=pd.read_csv(r'C:/Users/Jayant/Desktop/DATA-SCIENCE_JOURNEY/titanic/test.csv')
#print(df_initial.columns)
#df_initial.describe(include='all')

df=df_initial[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age','Cabin', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Embarked']]

df_initial.describe(include='all')
sns.swarmplot(y=df.Age,x=df.Survived)

# fare,age,sex,pclass,sibsp,Embarked we will consider after checking their plots 


# In[3]:


df


# In[4]:


test_set


# In[ ]:





# In[5]:


df=df_initial[['PassengerId','Age','Survived','Sex','Pclass','SibSp','Embarked','Fare']]
df_test=test_set[['PassengerId','Age','Sex','Pclass','SibSp','Embarked','Fare']]
#print(df.describe(include='all'))

#Since the embarked has lesser columns we have to impute or drop it 
#also convert categorical values
#first deal with categorical values
cat_lis=[]
for col in df:
    if(df[col].dtype=='object'):
        cat_lis.append(col)
    
print(cat_lis)
# since only 2 rows will be dropped
df=df.dropna()
#df.describe(include='all') after this we have 889 rows 





# In[6]:


df['Sex']=df["Sex"].astype("category")
df_test['Sex']=df_test["Sex"].astype("category")
df['Embarked']=df["Embarked"].astype("category")
df_test['Embarked']=df_test["Embarked"].astype("category")

df['Sex']=df['Sex'].cat.codes
df['Embarked']=df['Embarked'].cat.codes
df_test['Sex']=df_test['Sex'].cat.codes
df_test['Embarked']=df_test['Embarked'].cat.codes


# In[7]:


df


# In[8]:


df_test


# In[9]:


#OR ANOTHER WAY OF ONE HOT ENCODING ###THIS CELL ISN'T NECESSARY.
pd.get_dummies(df[['Sex','Embarked']])


# In[10]:


#FROM HERE WE START BUILDINFG VARIOUS MODELS OF MACHINELEARNING
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as mae
from sklearn.tree import DecisionTreeRegressor as dtr

cols_considered=['Age','Sex','Pclass','SibSp','Embarked','Fare']
x=df[cols_considered]
df_test=df_test[cols_considered]
y=df['Survived']

train_x, val_x,train_y,val_y= tts(x,y,random_state=1)
# decision tree model
#first check for larger range like [5,10,50,100,500,1000] then wheere you find the least mae check for numbers closer to that
#in this case i found it to be 500 and then finally sette with 130
nodes=[50,100,120,130,150]

for i in nodes:
    model= dtr(max_leaf_nodes=i,random_state=3)
    model.fit(train_x,train_y)
    pred=model.predict(val_x)
    print("the mae for {} turns out to be : {}".format(i,mae(val_y,pred)))


# In[11]:


model= dtr(max_leaf_nodes=130,random_state=1)
model.fit(train_x,train_y)
pred=model.predict(val_x)
print("the mae : {}".format(mae(val_y,pred)))


# In[15]:


df_test.describe()


# In[39]:



mean_age=df_test['Age'].mean()
df_test['Age']=df_test['Age'].fillna(mean_age)
mean_fare=df_test['Fare'].mean()
df_test['Fare']=df_test['Fare'].fillna(mean_fare)


# In[40]:


df_test.describe()


# In[45]:


#NOW WE TRAIN THE MODE WITH THE ENTIRE SET
final_model= dtr(max_leaf_nodes=130,random_state=1)
final_model.fit(x,y)
final_pred=final_model.predict(df_test)

for i in range(len(final_pred)):
    if final_pred[i]>0.5:
        final_pred[i]=1
    else:
        final_pred[i]=0
        
        
        
for i in range(len(final_pred)):
    print(int(final_pred[i]))


# In[ ]:


#so now we know how many nodes our tree should have to perfor best



# In[ ]:




