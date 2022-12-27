#This code is modified from krishnaik has developed, 
#I implemented this in vs code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
df=pd.read_csv('car data.csv')
#print(df.head())
print(df.shape)
#check categorical features
#df.column.unique()
#check missing values
print(df.isnull().sum())
#the feature year says alot since an old model 
#depreciating in value, hence ordinal value
print(df.columns)
final_df=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
current_year=2020
final_df['Year_diff']=current_year-final_df['Year']
print(final_df.head(5))
final_df.drop(['Year'], axis=1, inplace=True)
final_df=pd.get_dummies(final_df,drop_first=True)
print(final_df.corr())
#sns.pairplot(final_df)
#plt.show()
#corrmat=final_df.corr()
#top_corr_feat=corrmat.index
#plt.figure(figsize=(10,10))
#dis=sns.heatmap(final_df[top_corr_feat].corr(), annot=True, cmap='RdYlGn')
#plt.show()
X=final_df.iloc[:,1:]#independent variable
y=final_df.iloc[:,0]#dependent variable
#feature importance
model=ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
#feat_importances=pd.Series(model.feature_importances_, index=X.columns)
#feat_importances.nlargest(5).plot(kind='barh')
#plt.show()
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)
#print(X_train.shape)
rf=RandomForestRegressor()
###Hyperparameters tunning
n_estimators=[int(x) for x in np.linspace(start = 100, stop =1200, num =12)]
#print(n_estimators)#different decision trees
max_features = ['auto','sqrt']#number of feats to consider at every split
max_depth=[int(x) for x in np.linspace(5,30, num =6)]##max no of levels in tree
min_samples_split=[2,5,10,15,100]#min no of samples required to split a node
min_samples_leaf=[1,2,5,10]#min no of samples required to each leaf node
random_grid={'n_estimators':n_estimators,
             'max_features':max_features,
             'max_depth':max_depth,
             'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf}
print(random_grid)     
rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,
verbose=2, random_state=42, n_jobs=1) 
rf_random.fit(X_train,y_train) 
predictions=rf_random.predict(X_test)
print(predictions)
#sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)
plt.show()
file=open('randon_forest_regression_car_model.pkl','wb')
pickle.dump(rf_random,file)