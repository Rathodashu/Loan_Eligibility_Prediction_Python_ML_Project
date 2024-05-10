#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dataset = pd.read_csv("loan_eligibility.csv")


# In[6]:


dataset.head()


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# In[9]:


pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'], margins = True)


# In[10]:


dataset.boxplot(column='ApplicantIncome')


# In[11]:


dataset['ApplicantIncome'].hist(bins=20)


# In[12]:


dataset['CoapplicantIncome'].hist(bins=20)


# In[13]:


dataset.boxplot(column='ApplicantIncome', by = 'Education')


# In[14]:


dataset.boxplot(column='LoanAmount')


# In[15]:


dataset['LoanAmount'].hist(bins=20)


# In[16]:


dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[17]:


dataset.isnull().sum()


# In[30]:


dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)


# In[31]:


dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)


# In[32]:


dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)


# In[33]:


dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)


# In[34]:


dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())


# In[35]:


dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)


# In[36]:


dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)


# In[37]:


dataset.isnull().sum()


# In[38]:


dataset['TotalIncome']=dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome_log']= np.log(dataset['TotalIncome'])


# In[39]:


dataset['TotalIncome_log'].hist(bins=20)


# In[40]:


dataset.head()


# In[44]:


x = dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y = dataset.iloc[:,12].values


# In[45]:


x


# In[46]:


y


# In[47]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[48]:


print(x_train)


# In[50]:


from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()


# In[55]:


for i in range(0, 5):
    x_train[:,i]= labelencoder_x.fit_transform(x_train[:,i])


# In[56]:


x_train[:,7]= labelencoder_x.fit_transform(x_train[:,7])


# In[57]:


x_train


# In[59]:


labelencoder_y=LabelEncoder()
y_train= labelencoder_y.fit_transform(y_train)


# In[60]:


y_train


# In[61]:


for i in range(0, 5):
        x_train[:,i]= labelencoder_x.fit_transform(x_train[:,i])
    


# In[62]:


x_test[:,7]= labelencoder_x.fit_transform(x_test[:,7])


# In[63]:


labelencoder_y=LabelEncoder()
y_test= labelencoder_y.fit_transform(y_test)


# In[66]:


y_test


# In[68]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# In[69]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier= DecisionTreeClassifier(criterion= 'entropy', random_state=0)
DTClassifier.fit(x_train,y_train)


# In[71]:


y_pred= DTClassifier.predict(x_test)
y_pred


# In[72]:


from sklearn import metrics
print('The accuracy of decision tree is:', metrics.accuracy_score(y_pred,y_test))


# In[73]:


from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(x_train,y_train)


# In[74]:


y_pred= NBClassifier.predict(x_test)
y_pred


# In[75]:


print("The accuracy of Naive Bayes is:", metrics.accuracy_score(y_pred,y_test))


# In[76]:


testdata = pd.read_csv("loan_eligibility.csv")


# In[77]:


testdata.head()


# In[79]:


testdata.info()


# In[80]:


testdata.isnull().sum()


# In[82]:


testdata['Gender'].fillna(testdata['Gender'].mode()[0],inplace=True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0],inplace=True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0],inplace=True)
testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0],inplace=True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0],inplace=True)


# In[83]:


testdata.isnull().sum()


# In[84]:


testdata.boxplot(column='LoanAmount')


# In[85]:


testdata.boxplot(column='ApplicantIncome')


# In[86]:


testdata.LoanAmount= testdata.LoanAmount.fillna(testdata.LoanAmount.mean())


# In[87]:


testdata['LoanAmount_log']=np.log(testdata['LoanAmount'])


# In[88]:


testdata.isnull().sum()


# In[89]:


testdata['TotalIncome']= testdata['ApplicantIncome']+testdata['CoapplicantIncome']
testdata['TotalIncome_log']= np.log(testdata['TotalIncome'])


# In[90]:


testdata.head()


# In[91]:


test= testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[93]:


for i in range(0,5):
    test[:,i]=labelencoder_x.fit_transform(test[:,i])


# In[94]:


test[:,7]= labelencoder_x.fit_transform(test[:,7])


# In[95]:


test


# In[96]:


test= ss.fit_transform(test)


# In[97]:


pred= NBClassifier.predict(test)


# In[98]:


pred


# In[ ]:




