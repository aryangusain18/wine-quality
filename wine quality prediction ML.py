#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset = pd.read_csv(R'C:\Users\Hp\Desktop\Aryan  folder\winequalityN.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.groupby(['type']).count()


# In[5]:


dataset=pd.get_dummies(dataset)


# In[6]:


dataset.head()


# In[7]:


dataset.info()


# In[8]:


dataset = dataset.dropna()


# In[9]:


import matplotlib.pyplot as plt
dataset['quality'].hist(bins=20,figsize=(10,5))
plt.show()


# In[10]:


import seaborn as sns;
sns.set_theme(color_codes=True)
for x in dataset:
 ax = sns.regplot(y="quality", x=x, data=dataset)
 plt.show()


# # Splitting independent and dependent variable

# In[11]:


x = dataset.drop(["quality"],axis=1,inplace=False)


# In[12]:


from sklearn.preprocessing import LabelEncoder
bins = (2, 6, 9)
group_names = ['bad', 'good']
dataset['quality'] = pd.cut(dataset['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
dataset['quality'] = label_quality.fit_transform(dataset['quality'])
dataset['quality'].value_counts()


# In[13]:


y = dataset['quality']
y.head()


# # Splitting into two sets so as to check accuracy later

# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
print(len(x))
print(len(x_train))
print(len(x_test))


# In[15]:


from sklearn.preprocessing import MinMaxScaler
# creating normalization object 
norm = MinMaxScaler()
# fit data
norm_fit = norm.fit(x_train)
x_train=norm_fit.transform(x_train)
x_test=norm_fit.transform(x_test)
# display values
print(x_train)


# In[16]:


y_train.value_counts()


# In[17]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(x_train, y_train)
knpreds = kn.predict(x_test)
print(f"KNN Classifier\n\n {classification_report(y_test,knpreds)}")
cm=confusion_matrix(y_test,knpreds)
print("confusion matrix")
print(cm)
print("accuracy=",accuracy_score(y_test,knpreds))
print("f1 score=",f1_score(y_test,knpreds))


# In[18]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(x_train, y_train)
RFpreds = RF.predict(x_test)
print(f"Random Forest\n\n {classification_report(y_test,RFpreds)}")
cm=confusion_matrix(y_test,RFpreds)
print(cm)
print(accuracy_score(y_test,RFpreds))
print(f1_score(y_test,RFpreds))


# In[19]:


y_train.hist()


# In[20]:


from imblearn.over_sampling import SMOTE
oversample=SMOTE(random_state=42)
x_train, y_train = oversample.fit_resample(x_train, y_train)


# In[21]:


y_train.hist()


# In[22]:


kn = KNeighborsClassifier()
kn.fit(x_train, y_train)
knpreds = kn.predict(x_test)
print(f"KNN Classifier\n\n {classification_report(y_test,knpreds)}")
cm=confusion_matrix(y_test,knpreds)
print(cm)
print("accuracy=",accuracy_score(y_test,knpreds))
print("f1 score=",f1_score(y_test,knpreds))


# In[23]:


RF = RandomForestClassifier()
RF.fit(x_train, y_train)
RFpreds = RF.predict(x_test)
print(f"Random Forest\n\n {classification_report(y_test,RFpreds)}")
cm=confusion_matrix(y_test,RFpreds)
print(cm)
print("accuracy=",accuracy_score(y_test,RFpreds))
print("f1 score=",f1_score(y_test,RFpreds))


# In[24]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
DTpreds = DT.predict(x_test)
print(f"Decision Tree\n\n {classification_report(y_test,DTpreds)}")
cm=confusion_matrix(y_test,DTpreds)
print(cm)
print(accuracy_score(y_test,DTpreds))
print(f1_score(y_test,DTpreds))


# In[26]:


x_predict = list(RF.predict(x_test))
df = {'predicted':x_predict,'original':y_test}
pd.DataFrame(df).head(10)


# # THANK YOU
