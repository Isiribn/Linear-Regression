#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import pandas_profiling as pp
import sweetviz as sv
data=pd.read_csv("delivery_time.csv")
data.head()


# In[3]:


data


# In[2]:


data.dtypes


# In[4]:


data.shape


# In[5]:


type(data)


# In[6]:


data[data.duplicated()].shape


# In[7]:


data[data.duplicated()]


# In[8]:


data['Sorting Time']=data['Sorting Time'].astype('float64')


# In[9]:


data=data.rename({'Delivery Time':'DT'},axis=1)


# In[10]:


data=data.rename({'Sorting Time':'ST'},axis=1)


# In[11]:


data


# In[13]:


import statsmodels.formula.api as smf 
model1 = smf.ols('DT~ST',data=data).fit()
model1.summary()


# # Predicting...

# In[ ]:


newdata=pd.Series([21.25,9.0])


# In[15]:


data_pred=pd.DataFrame(newdata,columns=['ST'])


# In[16]:


model1.predict(data_pred)


# In[2]:


data.corr()


# In[3]:


data.describe()


# In[4]:


data.info()


# In[6]:


import seaborn as sn
sn.distplot(data['Delivery Time'])


# In[7]:


import seaborn as sn
sn.distplot(data['Sorting Time'])


# # Using sm.OLS

# In[19]:


#from models
import statsmodels.api as sm
import numpy as np
y=data['Delivery Time']
x=data['Sorting Time']
#y
#x
x=sm.add_constant(x)
model=sm.OLS(y,x)
res=model.fit()
res.params


# In[20]:


print(res.t_test([1,0]))


# In[21]:


print(res.f_test(np.identity(2)))


# In[23]:


import seaborn as sn
sn.regplot(x="Sorting Time", y="Delivery Time", data=data)


# In[24]:


data.plot(kind="box")


# In[25]:


print(res.tvalues,'\n',res.pvalues)


# In[26]:


print(res.rsquared,'\n',res.rsquared_adj)


# In[37]:


newdata=pd.Series([19.2,8.1])


# In[38]:


data_pred=pd.DataFrame(newdata,columns=['x'])


# In[39]:


res.predict(data_pred)


# # Using Sklearn

# In[60]:


from sklearn.linear_model import LinearRegression
import numpy as np
model=LinearRegression().fit(x,y)


# In[62]:


r_sq=model.score(x,y)
print('coefficient',r_sq)


# In[63]:


print('intercept:',model.intercept_)
print('slope:',model.coef_)


# In[64]:


y_pred=model.predict(x)
print('Predicted response:',y_pred,sep='\n')


# In[68]:


import seaborn as sns
sns.regplot(x='Sorting Time',y='Delivery Time',data=data)


# In[70]:


y_pred=model.predict(x)
print('Predicted response:',y_pred,sep='\n')


# In[80]:


x_new=[9,11,12,13]
x_new=np.array(x_new).reshape(-1,2)
print(x_new)
y_new=model.predict(x_new)
print('Predicted response:',y_new,sep='\n')

