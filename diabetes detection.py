#!/usr/bin/env python
# coding: utf-8

# ### Diabetes Dataset
# The objective of the dataset is to diagnose, whether a patient has diabetes, based on certain diagnostic measurements included in the dataset.

# In[ ]:


import pandas as pd # Data processing like pd.read_csv
import numpy as np # Linear Algebra
import matplotlib.pyplot as plt # for python visualization
import seaborn as sns # for Statistics visualization
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data= pd.read_csv(r"D:\DS projects\Diabetes\Diabetes\Dataset\diabetes.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe().T


# In[6]:


data.info()


# In[7]:


data.nunique() # to check duplicate values


# In[8]:


missing_num=data.isnull().sum()
missing_num


# In[9]:


# pip install missingno ; helps you to give a quick overview of your dataset completeness by visualizing it
import missingno as msno
msno.matrix(data)


# In[10]:


# created Numercial columns in num_col as varaible
num_col= ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
num_col


# In[11]:


# Created all Columns in Variable, as 'Outcome' is number but categorical column
variable= ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age', 'Outcome']
variable


# ### Outliers detection and treatment

# In[12]:


plt.figure(figsize=(17,8))
for i,col in enumerate(num_col):
    plt.subplot(2,4,i+1)
    sns.histplot(data[col])


# In[13]:


plt.figure(figsize=(16,8))
for i,col in enumerate(num_col):
    plt.subplot(2,4,i+1)
    sns.boxplot(data[col])


# In[14]:


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# In[15]:


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# In[16]:


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# In[17]:


for x in num_col:
  print(x,check_outlier(data,x))


# In[18]:


for x in num_col:
  replace_with_thresholds(data,x)


# In[19]:


plt.figure(figsize = (16,8))
for i,x in enumerate(num_col):
  plt.subplot(2,4,i+1)
  sns.histplot(data[x])


# In[20]:


plt.figure(figsize = (16,8))
for i,x in enumerate(num_col):
  plt.subplot(2,4,i+1)
  sns.boxplot(data[x])


# In[21]:


sns.countplot('Outcome', data=data)


# In[22]:


plt.pie(data['Outcome'].value_counts(),
                   explode      = [0.0, 0.25], 
                   startangle   = 30, 
                   shadow       = True, 
                   colors       = ['#004d99', '#ac7339'], 
                   textprops    = {'fontsize': 8, 'fontweight': 'bold', 'color': 'white'},
                   pctdistance  = 0.50, autopct = '%1.2f%%'
                  );


# In[23]:


sns.jointplot('Age','Glucose', data=data, kind="reg")


# In[24]:


sns.pairplot(data=data, kind='scatter')


# In[25]:


sns.pairplot(data, hue='Outcome')


# In[26]:


plt.figure(figsize=(8,5))
sns.heatmap(data.corr(), cmap='coolwarm', annot= True, linewidths=0.5, linecolor="g")


# In[27]:


from sklearn.preprocessing import StandardScaler # scaling makes varaibles in one scale level between -1 to +1


# In[28]:


X= data.drop(columns=['Outcome'])
y= data['Outcome']
X.shape, y.shape


# In[29]:


sc= StandardScaler()


# In[30]:


sc.fit_transform(X)


# In[31]:


### Import the libraries and model creation


# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[33]:


models = []
models.append(('LR', LogisticRegression(random_state = 0)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state = 0)))
models.append(('RF', RandomForestClassifier(random_state = 0)))
models.append(('SVM', SVC(gamma='auto', random_state = 0)))
models.append(('XGB', GradientBoostingClassifier(random_state = 0)))
models.append(("LightGBM", LGBMClassifier(random_state = 0)))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    
        kfold = KFold(n_splits = 10, random_state = None)
        
        cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


# In[34]:


### hyper parameter tuning
lgbm = LGBMClassifier(random_state = 0)


# In[35]:


lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
              "n_estimators": [500, 1000, 1500],
              "max_depth":[3,5,8]}


# In[36]:


gs_cv = GridSearchCV(lgbm, 
                     lgbm_params, 
                     cv = 10, 
                     n_jobs = -1, 
                     verbose = 2).fit(X, y)


# In[37]:


gs_cv.best_params_


# In[64]:


gs_cv.fit(X_train,y_train)


# In[65]:


y_pred_lgbm=gs_cv.predict(X_test)


# In[66]:


ac_lgbm=accuracy_score(y_pred_lgbm,y_test)
ac_lgbm


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25, random_state=0)


# In[42]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[71]:


pickle.dump(sc, open('scaler.pkl','wb'))


# In[43]:


### Hyper parameter tuning for Logistic Regression with training the model
lr= LogisticRegression()


# In[44]:


lr_params={
    'penalty':['l1','l2','elasticnet'],
    'C'      : [-2,-1,1,2,3],
    'solver' :['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
}


# In[45]:


clf = GridSearchCV(lr,
                         lr_params,
                         n_jobs = -1,
                         cv = 10,
                         verbose = 2)


# In[46]:


clf.fit(X_train,y_train)


# In[47]:


clf.best_params_


# In[48]:


clf.best_score_


# In[49]:


y_pred= clf.predict(X_test)


# In[52]:


ac= accuracy_score(y_pred,y_test)
ac


# In[53]:


rfc= RandomForestClassifier()


# In[54]:


rfc_params={'n_estimators': [100,200,300,500,1000],
           'criterion': ['gini','entropy','log_loss'],
           'max_depth':[1,3,5,7,10],
           'max_features':['sqrt','log2']}


# In[55]:


clf_rfc= GridSearchCV(rfc,
                      rfc_params,
                      n_jobs = -1,
                      cv = 10,
                      verbose = 2)


# In[56]:


clf_rfc.fit(X_train,y_train)


# In[57]:


clf_rfc.best_params_


# In[58]:


clf_rfc.best_score_


# In[59]:


y_pred_rfc=clf_rfc.predict(X_test)


# In[60]:


ac=accuracy_score(y_pred_rfc,y_test)
ac

## how Hyperparameter tuning important in enhancing the accuracy of the model

Logistic Regression
Accuracy
Before  =  76.95
After   =  79.16

Random Forest Classifier
Accuracy
Before  =  75.91
After   =  77.60

# In[70]:


import pickle
pickle.dump(lr, open('diabetes.pkl','wb'))


# In[ ]:




