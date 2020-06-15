
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
import pickle


train_df= pd.read_excel("Data_Train.xlsx")
test_df = pd.read_excel("Test_set.xlsx")

big_df = train_df.append(test_df,sort=False)




big_df.tail()
big_df.dtypes




big_df['Date'] = big_df['Date_of_Journey'].str.split('/').str[0]
big_df['Month'] = big_df['Date_of_Journey'].str.split('/').str[1]
big_df['Year'] = big_df['Date_of_Journey'].str.split('/').str[2]




big_df.head()




big_df.dtypes


big_df['Date'] = big_df['Date'].astype(int)
big_df['Month'] = big_df['Month'].astype(int)
big_df['Year'] = big_df['Year'].astype(int)



big_df.dtypes

big_df=big_df.drop(['Date_of_Journey'],axis=1)

big_df[big_df['Total_Stops'].isnull()]

big_df['Total_Stops'] = big_df['Total_Stops'].fillna("1 stop")


big_df[big_df['Total_Stops'].isnull()]

big_df['Total_Stops'] = big_df['Total_Stops'].replace('non-stop','0 stop')




big_df['Stop'] = big_df['Total_Stops'].str.split(' ').str[0]




big_df.head()




big_df['Stop'] = big_df['Stop'].astype(int)
big_df = big_df.drop('Total_Stops',axis=1)


# In[97]:


big_df.head()


# In[98]:


big_df['Arrival_Time']=big_df['Arrival_Time'].str.split(' ').str[0]
big_df['Arrival_Hour'] = big_df['Arrival_Time'] .str.split(':').str[0]
big_df['Arrival_Minute'] = big_df['Arrival_Time'] .str.split(':').str[1]


# In[99]:


big_df['Arrival_Hour']=big_df['Arrival_Hour'].astype(int)
big_df['Arrival_Minute']=big_df['Arrival_Minute'].astype(int)
big_df=big_df.drop(['Arrival_Time'],axis=1)


# In[100]:


big_df.head()


# In[101]:


big_df['Departure_Hour'] = big_df['Dep_Time'] .str.split(':').str[0]
big_df['Departure_Minute'] = big_df['Dep_Time'] .str.split(':').str[1]


# In[102]:


big_df['Departure_Hour']=big_df['Departure_Hour'].astype(int)
big_df['Departure_Minute']=big_df['Departure_Minute'].astype(int)
big_df=big_df.drop(['Dep_Time'],axis=1)


# In[103]:


big_df.head()


# In[104]:


big_df['Route_1']=big_df['Route'].str.split('→ ').str[0]
big_df['Route_2']=big_df['Route'].str.split('→ ').str[1]
big_df['Route_3']=big_df['Route'].str.split('→ ').str[2]
big_df['Route_4']=big_df['Route'].str.split('→ ').str[3]
big_df['Route_5']=big_df['Route'].str.split('→ ').str[4]


# In[105]:


big_df['Price'].fillna((big_df['Price'].mean()),inplace=True)
big_df['Route_1'].fillna("None",inplace=True)
big_df['Route_2'].fillna("None",inplace=True)
big_df['Route_3'].fillna("None",inplace=True)
big_df['Route_4'].fillna("None",inplace=True)
big_df['Route_5'].fillna("None",inplace=True)


# In[106]:


big_df=big_df.drop(['Route'],axis=1)


# In[107]:


big_df.head()


# In[108]:


big_df.isnull().sum()


# In[109]:


big_df=big_df.drop(['Duration'],axis=1)


# 

# In[110]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
big_df["Airline"]=encoder.fit_transform(big_df['Airline'])
big_df["Source"]=encoder.fit_transform(big_df['Source'])
big_df["Destination"]=encoder.fit_transform(big_df['Destination'])
big_df["Additional_Info"]=encoder.fit_transform(big_df['Additional_Info'])
big_df["Route_1"]=encoder.fit_transform(big_df['Route_1'])
big_df["Route_2"]=encoder.fit_transform(big_df['Route_2'])
big_df["Route_3"]=encoder.fit_transform(big_df['Route_3'])
big_df["Route_4"]=encoder.fit_transform(big_df['Route_4'])
big_df["Route_5"]=encoder.fit_transform(big_df['Route_5'])


# ### Feature Selection

# In[111]:


big_df.head()


# In[112]:


df_train = big_df[0:10683]
df_test  = big_df[10683:]


# In[113]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[114]:


X = df_train.drop('Price',axis = 1)
Y = df_train.Price


# In[115]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state =0)


# In[116]:


print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)
print("Y_train shape:",Y_train.shape)
print("Y_test shape:",Y_test.shape)



# In[117]:


model = SelectFromModel(Lasso(alpha=0.005,random_state=0))


# In[118]:


model.fit(X_train, Y_train)


# In[119]:


model.get_support()


# In[120]:


selected_features = X_train.columns[model.get_support()]
selected_features


# 

# In[121]:


X_train=X_train.drop('Year',axis=1)
X_test=X_test.drop('Year',axis=1)


# ### RandomForestRegressor

# In[122]:


from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[123]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[124]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[125]:


# Random search of parameters, using 3 fold cross validation, 
# search across 50 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 50, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[126]:


rf_random.fit(X_train,Y_train)


# In[127]:


y_pred=rf_random.predict(X_test)


# In[ ]:



# Saving model to disk
pickle.dump(rf_random, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

