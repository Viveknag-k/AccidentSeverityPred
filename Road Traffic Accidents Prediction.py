#!/usr/bin/env python
# coding: utf-8

# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             IMPORTING REQUIRED LIBRARIES
# </p>
# </div>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             LOADING DATA
# </p>
# </div>

# In[2]:


data = pd.read_csv("C:\\Users\\Vivek Nag Kanuri\\Downloads\\Roadapp\\RTA Dataset.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#            EXPLORATORY ANALYSIS
# </p>
# </div>

# In[5]:


data.describe().T


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#            SKEWNESS OF THE DATA
# </p>
# </div>

# In[6]:


Skew = data.skew()
print(Skew)


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             CORRELATION
# </p>
# </div>

# In[7]:


correlation = data.corr()
sns.heatmap(correlation,annot = True, cmap = 'Blues')


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             PREPROCESSING 
# </p>
# </div>

# In[8]:


df=data.copy(deep=True)
df.head()


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#            TREATING WITH NULL VALUES
# </p>
# </div>

# In[9]:


df.isnull().sum()


# In[10]:


df.shape


# In[11]:


df.size


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             REMOVING UNWANTED FEATURES, keeping required columns.. 
# </p>
# </div>

# In[12]:


df.drop(['Time','Driving_experience','Type_of_vehicle','Educational_level'],axis=1,inplace=True)


# In[13]:


df.drop(['Vehicle_driver_relation','Lanes_or_Medians','Owner_of_vehicle','Area_accident_occured','Road_allignment',
         'Types_of_Junction','Light_conditions','Weather_conditions','Vehicle_movement','Fitness_of_casuality',
        'Vehicle_movement','Age_band_of_driver','Sex_of_driver'],axis=1,inplace=True)


# In[14]:


df.drop(['Pedestrian_movement','Cause_of_accident','Work_of_casuality','Road_surface_conditions'],axis=1,inplace=True)


# In[15]:


df.drop(['Service_year_of_vehicle','Defect_of_vehicle'],axis=1,inplace=True)


# In[16]:


df.dropna(inplace=True)


# In[17]:


for i in df.columns:
    if df[i].dtypes== object:
        print(i)
        print(df[i].unique())
        print(df[i].nunique())
        print()


# In[18]:


df.head()


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             LABEL ENCODING
# </p>
# </div>

# In[19]:


from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
for col in df.columns:
    if df[col].dtype == object:
        df[col] = l.fit_transform(df[col])


# In[20]:


df.head()


# <div style="color:black;
#            display:fill;
#            border-radius:25px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:0.5px">
# 
# <p style="padding: 10px;
#               color:black;">
#             MODEL BUILDING 
# </p>
# </div>

# In[21]:


x=df.drop('Accident_severity',axis=1)
y=df['Accident_severity']


# In[22]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.30)


# In[23]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler(feature_range=(0,1))
xtrain=mms.fit_transform(xtrain)
xtest=mms.fit_transform(xtest)
xtrain=pd.DataFrame(xtrain)
xtest=pd.DataFrame(xtest)


# In[24]:


R= {'Model':[],'Accuracy':[],'F1':[]}


# In[25]:


Results=pd.DataFrame(R)
Results.head()


# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier()
rf.fit(xtrain,ytrain)
    
ypred=rf.predict(xtest)

print('-----------------------------------------------------------------------------------------------------------------------')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score

print('confusion matrix :',confusion_matrix(ytest,ypred))
print('classification report:',classification_report(ytest,ypred))
print('accuracy :',round(accuracy_score(ytest,ypred),2))
print('precision :',round(precision_score(ytest,ypred,average='weighted'),2))
print('recall :',round(recall_score(ytest,ypred,average='weighted'),2))
print('f1 :',round(f1_score(ytest,ypred,average='weighted'),2))
print()


# In[27]:


import pickle
pickle_out = open("rf.pkl","wb")
pickle.dump(rf,pickle_out)
pickle_out.close()


# In[28]:


pickle_out = open('lbe.pkl','wb')
pickle.dump(l,pickle_out)
pickle_out.close()

