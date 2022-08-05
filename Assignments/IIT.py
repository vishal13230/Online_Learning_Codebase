# Introduction    

# This challenge is the capstone project of the Summer Analytics, a primer course on Data Science, conducted by 
# Consulting and Analytics Club of IIT Guwahati in the summers.

# The dataset is provided by DeltaX is the pioneering cross-channel digital advertising platform. 
# The cloud-based platform leverages big data, user behavior, and machine learning algorithms to improve 
# performance across the business funnel of advertisers.

# Problem Statement
# Let's take a case where an advertiser on the platform (DeltaX) would like to estimate the performance of their 
# campaign in the future.

# Imagine it is the first day of March and you are given the past performance data of ads between 1st August to 
# 28th Feb. You are now tasked to predict an ad's future performance (revenue) between March 1st and March 15th. 
# Well, it is now time for you to put on your problem-solving hats and start playing with the data provided under 
# the "data" section.

# Loading Libraries - 
# Importing Basic libraries to play with data and to construct EDA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importing Libraries to make a Model for the data
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# For plotting the regression tree
from IPython.display import Image
import pydotplus
from graphviz import Digraph


# To remove warnings
import warnings
warnings.filterwarnings('ignore')

# Reading Data
data_train = pd.read_csv('Train_Data.csv')
data_test = pd.read_csv('Test_Data.csv')

# Data Description

# date: the date on which the ad was made live
# campaign: campaign number
# adgroup: adgroup number
# ad: ad number
# impressions - Number of time the ad was shown
# clicks - Number of time the ad clicked shown
# cost - Amount spent to show ad
# conversions - Number of transactions received
# revenue: revenue generated from the ad

# Our goal is to predict the revenue from Test data with a low RMSE score.

# File structure and content
data_train.head()
data_test.head()

print("Row: ", data_train.shape[0])
print("Col: ", data_train.shape[1])

print("Row: ", data_test.shape[0])
print("Col: ", data_test.shape[1])

data_train.info()
data_test.info()

# Summary:

# The data_train data have 4571 rows and 9 columns.
# The data_test data have 318 rows and 8 columns.
# There are no missing data in both data_train and data_test.

# Exploratory Data Analysis(EDA)
data_train.describe()

sns.countplot("adgroup", data = data_train)
# There are 4 adgroups in adgroup column which will be converted into numerical variable by 
# creating dummy variable.

data_train.campaign.value_counts()
sns.countplot("campaign", data = data_train)
# We can see that there is only 1 campaign. So taking campaign column into consideration is meaningless.

data_train["ad"].value_counts().sort_values()
# We can see that there are 70 types of ads in ad column in data_train.

sns.scatterplot(x = data_train["revenue"], y = data_train["cost"])
sns.scatterplot(x = "conversions", y = "revenue", data = data_train)
sns.scatterplot(x = "adgroup", y = "revenue", data = data_train)

plt.figure(figsize=(10, 5))
plt.title("Correlation Analysis")
sns.heatmap(data_train.corr(), annot = True)

# Note:
# The variable conversion is highly positively correlated to our target variable revenue.
# The cost is least positively correlated with revenue.
# There is only positive relationship between the variables in our dataset.

# Adding Feature
def Add(x):
    x["CTR"] = x["clicks"] / x["impressions"]
    x["CPC"] = x["cost"] / x["clicks"]
    x["CPA"] = x["cost"] / x["conversions"]
    return x

Add(data_train)
Add(data_test)
data_train.head()

data_train = data_train.replace(np.inf, np.nan)
data_train = data_train.fillna(0)
data_test = data_test.replace(np.inf, np.nan)
data_test = data_test.fillna(0)

data_train.isnull().sum()
data_train.describe()

# Treating Outliers
data_train.cost.sort_values(ascending = False)
data_train.impressions.sort_values(ascending = False)
data_train.clicks.sort_values(ascending = False)
data_train.CPC.sort_values(ascending = False)
data_train.CPA.sort_values(ascending = False)

# By using this, we found that there are outliers in Cost, impressions, clicks, CPC and CPA. 
# So, we will treat it in the next line.

data_train.cost[(data_train.cost > 300)] = 300
data_train.impressions[(data_train.impressions > 2200)] = 2200
data_train.clicks[(data_train.clicks > 1200)] = 1200
data_train.CPC[(data_train.CPC > 2.5)] = 2.5
data_train.CPA[(data_train.CPA > 27)] = 27

data_train.describe()

# For Training data
X_train = data_train.drop(["date", "revenue", "campaign", "ad"], axis = 1)
X_train = pd.get_dummies(X_train)
X_train.head()

y_train = data_train["revenue"]
y_train.head()

# For Test data
X_test = data_test.drop(["date", "campaign", "ad"], axis = 1)
X_test = pd.get_dummies(X_test)
X_test.head()

# Regression Tree
# Got the best value for the parameters by using Grid Search CV
regtree = tree.DecisionTreeRegressor(criterion = "mse", 
                                     max_depth = 6, 
                                     max_features = "auto", 
                                     min_samples_split = 2, 
                                     splitter = "best", 
                                     random_state = 0)

regtree.fit(X_train, y_train)

y_train_pred = regtree.predict(X_train)
y_test_pred = regtree.predict(X_test)

print("RMSE score for Training data: ", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("R2 score for Training data: ", r2_score(y_train, y_train_pred))

# creating a csv file for the prediction data
res = pd.DataFrame(y_test_pred)
res.index = X_test.index
res.columns = ["revenue"]
res.to_csv("prediction_results.csv", index = False) 

res.describe()
res.head()

# The RMSE score for our Testing data is around 137.89 which is very near to our training model RMSE score 138.58, 
# which shows that the model is good for predicting revenue.

# Creating Image for Regression Tree
dot_data = tree.export_graphviz(regtree, out_file = None, 
                                feature_names = X_test.columns, filled = True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())