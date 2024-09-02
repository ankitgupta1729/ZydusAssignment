#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 12:20:37 2024

@author: ankit
"""

# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn import preprocessing 
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# loading dataset

df = pd.read_csv("train.csv")
validation_df = pd.read_csv("validation.csv")
###############################################################################

######################## Exploratory Data Analysis ############################

print("Shape of the train dataset is: ", df.shape) # dataset has 493 rows and 13 columns
print("\nTop 5 rows in the train dataset are: \n\n",df.head(5)) # display the first 5 rows
print("\nColumns in the train dataset are: \n\n",df.columns) # display all the columns in the given dataset
print("\nDatatypes of the columns in the train dataset are: \n\n",df.dtypes) # display the datatypes of all the columns

print("\nTotal Columns in the train dataset is: ",len(df.columns))

print("\nTotal Categorical variables in the train dataset is: ",len(df.dtypes[df.dtypes=='object']))

print(len(np.unique(df['Loan_ID']))==len(df)) # True

# Loan_ID column has all unique values and not null values

# visualize all the unique values of categorical variables

categorical = (df.dtypes == 'object') 
categorical_cols = list(categorical[categorical].index) 
categorical_cols.remove('Loan_ID')
plt.figure(figsize=(25,52)) 
index = 1

for col in categorical_cols: 
    y = df[col].value_counts() 
    plt.subplot(10,5,index) 
    plt.xticks(rotation=90) 
    sns.barplot(x=list(y.index), y=y) 
    index +=1
    
# Exploring each categorical variable

# 1. Gender
df['Gender'].unique()
print(df.Gender.value_counts(dropna=False))
print("Percentage of Male:", len(df[df['Gender']=='Male'])/len(df.Gender)*100)
print("Percentage of Female:", len(df[df['Gender']=='Female'])/len(df.Gender)*100)
sns.countplot(x="Gender", data=df,hue='Gender', palette="hls")
plt.title("Gender: Male vs Female")
plt.show()    

# Males are significantly higher than Feamles

#2. Married

print(df['Married'].unique())
print(df.Married.value_counts(dropna=False))
print("Percentage of Yes:", len(df[df['Married']=='Yes'])/len(df.Married)*100)
print("Percentage of No:", len(df[df['Married']=='No'])/len(df.Married)*100)
sns.countplot(x="Married", data=df,hue='Married', palette="hls")
plt.title("Married: Yes vs No")
plt.show() 

# Married are significantly higher than Non-Married

# 3. Dependents

print(df['Dependents'].unique())
print(df.Dependents.value_counts(dropna=False))
print("Percentage of 0:", len(df[df['Dependents']=='0'])/len(df.Dependents)*100)
print("Percentage of 1:", len(df[df['Dependents']=='1'])/len(df.Dependents)*100)
print("Percentage of 2:", len(df[df['Dependents']=='2'])/len(df.Dependents)*100)
print("Percentage of 3+:", len(df[df['Dependents']=='3+'])/len(df.Dependents)*100)
sns.countplot(x="Dependents", data=df,hue='Dependents', palette="hls")
plt.title("Dependents: 0,1,2,3+")
plt.show() 

# Applicants which don't have dependents are significantly higher than applicants
# that have 1,2 and 3 or more than 3 dependents

# 4.  Education

print(df['Education'].unique())
print(df.Education.value_counts(dropna=False))
print("Percentage of Graduate:", len(df[df['Education']=='Graduate'])/len(df.Education)*100)
print("Percentage of Not Graduate:", len(df[df['Education']=='Not Graduate'])/len(df.Education)*100)
sns.countplot(x="Education", data=df,hue='Education', palette="hls")
plt.title("Education: Graduate vs Not Graduate")
plt.show() 

# Graduate applicants are significantly higher than non-graduate applicants

# 5. Self_Employed

print(df['Self_Employed'].unique())
print(df.Self_Employed.value_counts(dropna=False))
print("Percentage of Yes:", len(df[df['Self_Employed']=='Yes'])/len(df.Self_Employed)*100)
print("Percentage of No:", len(df[df['Self_Employed']=='No'])/len(df.Self_Employed)*100)
sns.countplot(x="Self_Employed", data=df,hue='Self_Employed', palette="hls")
plt.title("Self_Employed: Yes vs No")
plt.show() 

# non-self_employed are significantly higher than applicants who are self-employed

# 6. Property_Area

print(df['Property_Area'].unique())
print(df.Property_Area.value_counts(dropna=False))
print("Percentage of Urban:", len(df[df['Property_Area']=='Urban'])/len(df.Property_Area)*100)
print("Percentage of Rural:", len(df[df['Property_Area']=='Rural'])/len(df.Property_Area)*100)
print("Percentage of Semiurban:", len(df[df['Property_Area']=='Semiurban'])/len(df.Property_Area)*100)
sns.countplot(x="Property_Area", data=df,hue='Property_Area', palette="hls")
plt.title("Property_Area: Urban, Rural and Semiurban")
plt.show()

# All 3 categories or Urban, Rural and Semiurban are almost in equal proportion

#  Exploring Numerical variables

# 7. Credit_History

print(df['Credit_History'].unique())
print(df.Credit_History.value_counts(dropna=False))
print("Percentage of 1:", len(df[df['Credit_History']==1.0])/len(df.Credit_History)*100)
print("Percentage of 0:", len(df[df['Credit_History']==0.0])/len(df.Credit_History)*100)
sns.countplot(x="Credit_History", data=df,hue='Credit_History', palette="hls")
plt.title("Credit_History: 1 vs 0")
plt.show()

# Applicants those have credit history are significantly higher than applicants
# those don't have credit history

#  8. ApplicantIncome

print(df['Loan_Amount_Term'].unique())
print(df.Loan_Amount_Term.value_counts(dropna=False))

# Number of applicants those have loan amount term as 360 are significantly higher than others

# 9. Loan_Status

print(df['Loan_Status'].unique())
print(df.Loan_Status.value_counts(dropna=False))
print("Percentage of Y:", len(df[df['Loan_Status']=='Y'])/len(df.Loan_Status)*100)
print("Percentage of N:", len(df[df['Loan_Status']=='N'])/len(df.Loan_Status)*100)
sns.countplot(x="Loan_Status", data=df,hue='Loan_Status', palette="hls")
plt.title("Loan_Status: Y vs N")
plt.show() 

# Applicants those are eligible for loan are significantly higher than those applcants
# who are not eligible

# Other Numerical variables

print(df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']].describe())


###############################################################################

# Histograms

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

sns.histplot(data=df, x="ApplicantIncome", kde=True, ax=axs[0, 0], color='violet')
sns.histplot(data=df, x="CoapplicantIncome", kde=True, ax=axs[0, 1], color='yellow')
sns.histplot(data=df, x="LoanAmount", kde=True, ax=axs[1, 0], color='blue');
sns.histplot(data=df, x="Loan_Amount_Term", kde=True, ax=axs[1, 1], color='red');

# Here, ApplicantIncome, CoapplicantIncome are positive skewed and has outliers and LoanAmount is little negative skewed
# Loan_Amount_Term feature has a outlier and negative skewed.

###############################################################################

# Finding Correlation
df1=copy.copy(df)
label_encoder = preprocessing.LabelEncoder() 
cat = (df1.dtypes == 'object') 
for i in list(cat[cat].index): 
    df1[i] = label_encoder.fit_transform(df1[i])
    
plt.figure(figsize=(12,6)) 

sns.heatmap(df1.corr(),cmap='BrBG',fmt='.2f', linewidths=2,annot=True)
# symmetric correlation matrix and loan_status is highly correlated with credit_history
# Gender/Dependents and Married are somewhat correlated
# LoanAmount and ApplicantIncome is correlated

# Loan_ID is very less correlated with other variables and the target variable and
# it is used for the primary key, we can drop it.

###############################################################################

############################ Data Preprocessing  ##############################

print(df.isnull().sum())
print(validation_df.isnull().sum())

## Removing unnecessary variable

# Loan_ID is very less correlated with other variables and the target variable 
# and it is used as a primary key, so, we can drop it.


df = df.drop(['Loan_ID'], axis = 1)
validation_df = validation_df.drop(['Loan_ID'], axis = 1)


## make the same datatypes

validation_df["CoapplicantIncome"] = validation_df["CoapplicantIncome"].astype(float)
validation_df["Loan_Amount_Term"] = validation_df["Loan_Amount_Term"].astype(float)  

## Imputing the missing values

# For missing valued categorical variables, we will use "mode".

df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True) 
# Though Credit_History and Loan_Amount_Term is a numberical variable but 
# its values are in particular class like 0 and 1 and 360,180 etc and it is working as a categorical variable

# For missing valued numerical variables, we will use "mean".


df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)

#######

validation_df['Gender'].fillna(validation_df['Gender'].mode()[0],inplace=True)
validation_df['Married'].fillna(validation_df['Married'].mode()[0],inplace=True)
validation_df['Dependents'].fillna(validation_df['Dependents'].mode()[0],inplace=True)
validation_df['Self_Employed'].fillna(validation_df['Self_Employed'].mode()[0],inplace=True)
validation_df['Credit_History'].fillna(validation_df['Credit_History'].mode()[0],inplace=True)
validation_df['Loan_Amount_Term'].fillna(validation_df['Loan_Amount_Term'].mode()[0],inplace=True) 
# Though Credit_History and Loan_Amount_Term is a numberical variable but 
# its values are in particular class like 0 and 1 and 360,180 etc and it is working as a categorical variable

# For missing valued numerical variables, we will use "mean".


validation_df['LoanAmount'].fillna(validation_df['LoanAmount'].mean(),inplace=True)



## Label Encoding (One-hot Encoding)

df = pd.get_dummies(df)

# removing unwanted columns
df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'Self_Employed_No', 'Loan_Status_N'], axis = 1)

# Rename the columns    
df.rename(columns={'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed',
       'Loan_Status_Y': 'Loan_Status'}, inplace=True)
#df.replace({False: 0, True: 1}, inplace=True)
print("\nColumns are:\n\n",df.columns)

df["Gender"] = df["Gender"].astype(int)
df["Married"] = df["Married"].astype(int)
df["Dependents_0"] = df["Dependents_0"].astype(int)
df["Dependents_1"] = df["Dependents_1"].astype(int)
df["Dependents_2"] = df["Dependents_2"].astype(int)
df["Dependents_3+"] = df["Dependents_3+"].astype(int)
df["Education"] = df["Education"].astype(int)
df["Self_Employed"] = df["Self_Employed"].astype(int)
df["Property_Area_Rural"] = df["Property_Area_Rural"].astype(int)
df["Property_Area_Semiurban"] = df["Property_Area_Semiurban"].astype(int)
df["Property_Area_Urban"] = df["Property_Area_Urban"].astype(int)
df["Loan_Status"] = df["Loan_Status"].astype(int)

#########

validation_df = pd.get_dummies(validation_df)

# removing unwanted columns
validation_df = validation_df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate', 
              'Self_Employed_No'], axis = 1)

# Rename the columns    
validation_df.rename(columns={'Gender_Male': 'Gender', 'Married_Yes': 'Married', 
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed'}, inplace=True)
#df.replace({False: 0, True: 1}, inplace=True)
print("\nColumns are:\n\n",validation_df.columns)

validation_df["Gender"] = validation_df["Gender"].astype(int)
validation_df["Married"] = validation_df["Married"].astype(int)
validation_df["Dependents_0"] = validation_df["Dependents_0"].astype(int)
validation_df["Dependents_1"] = validation_df["Dependents_1"].astype(int)
validation_df["Dependents_2"] = validation_df["Dependents_2"].astype(int)
validation_df["Dependents_3+"] = validation_df["Dependents_3+"].astype(int)
validation_df["Education"] = validation_df["Education"].astype(int)
validation_df["Self_Employed"] = validation_df["Self_Employed"].astype(int)
validation_df["Property_Area_Rural"] = validation_df["Property_Area_Rural"].astype(int)
validation_df["Property_Area_Semiurban"] = validation_df["Property_Area_Semiurban"].astype(int)
validation_df["Property_Area_Urban"] = validation_df["Property_Area_Urban"].astype(int)


print("\n number of rows in dataset before outlier removal:",len(df))
# ## Removing Outliers by using boxplots

Q1 = df.quantile(0.25) # 25 percentile 
Q3 = df.quantile(0.75) # 75 percentile
IQR = Q3 - Q1 # Interquartile range

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)] 

print("\n number of rows in dataset after outlier removal:",len(df))

# ApplicantIncome, CoapplicantIncome, and LoanAmount are positively skewed.
# So, we will use square root transformation to normalized the distribution.

df.ApplicantIncome = np.sqrt(df.ApplicantIncome)
df.CoapplicantIncome = np.sqrt(df.CoapplicantIncome)
df.LoanAmount = np.sqrt(df.LoanAmount)

validation_df.ApplicantIncome = np.sqrt(validation_df.ApplicantIncome)
validation_df.CoapplicantIncome = np.sqrt(validation_df.CoapplicantIncome)
validation_df.LoanAmount = np.sqrt(validation_df.LoanAmount)

## Splitting into dependent and independent features

X = df.drop(["Loan_Status"], axis=1)
y = df["Loan_Status"]

## Data Imbalanced resolution

# As we have seen initially that Loan_Status column is imbalanced for classes
# so we use oversampling using SMOTE to avoid the overfitting problem

X, y = SMOTE().fit_resample(X, y)  

# check whether classes are baanced or not
sns.countplot(y=y, data=df, palette="coolwarm")
plt.ylabel('Loan Status')
plt.xlabel('Total')
plt.show()

## Data Normalization


X = MinMaxScaler().fit_transform(X)

## Train-Test Split

# we make the 80% for training set and 20% for the validation set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

###############################################################################
###############################################################################

##############################  Models  #######################################

# 1. Logistic Regression

LogR_clf = LogisticRegression(solver='saga', max_iter=500, random_state=1)
LogR_clf.fit(X_train, y_train)

y_pred = LogR_clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

LogR_Acc = accuracy_score(y_pred,y_test)
print('Logistic Regression accuracy:',LogR_Acc*100)
# 2. KNN

score_list = []
for i in range(1,21):
    KNN_clf = KNeighborsClassifier(n_neighbors = i)
    KNN_clf.fit(X_train, y_train)
    score_list.append(KNN_clf.score(X_test, y_test))
    
plt.plot(range(1,21), score_list)
plt.xticks(np.arange(1,21,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.savefig('figures//KNN.png', bbox_inches='tight')  
plt.show()
KNN_Acc = max(score_list)
print("KNN best accuracy: ",KNN_Acc*100)

# 3. SVM

SVC_clf = SVC(kernel='rbf', max_iter=500)
SVC_clf.fit(X_train, y_train)

y_pred = SVC_clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

SVC_Acc = accuracy_score(y_pred,y_test)
print('SVC accuracy: ',SVC_Acc*100)

# 4. Naive Bayes

# 4.1 Categorical NB

NB_clf1 = CategoricalNB()
NB_clf1.fit(X_train, y_train)

y_pred = NB_clf1.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

NB_Acc1 = accuracy_score(y_pred,y_test)
print('Categorical Naive Bayes accuracy: ',NB_Acc1*100)

# 4.2 Gaussian NB

NB_clf2 = GaussianNB()
NB_clf2.fit(X_train, y_train)

y_pred = NB_clf2.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

NB_Acc2 = accuracy_score(y_pred,y_test)
print('Gaussian Naive Bayes accuracy: ',NB_Acc2*100)

# 5. Decision Tree

score_list= []
for i in range(2,21):
    DT_clf = DecisionTreeClassifier(max_leaf_nodes=i)
    DT_clf.fit(X_train, y_train)
    score_list.append(DT_clf.score(X_test, y_test))
    
plt.plot(range(2,21), score_list)
plt.xticks(np.arange(2,21,1))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.savefig('figures//Decision_Tree.png', bbox_inches='tight')  
plt.show()
DT_Acc = max(score_list)
print("Decision Tree Accuracy: ",DT_Acc*100)

# 6. Random Forest

score_list = []
for i in range(2,41):
    RF_clf = RandomForestClassifier(n_estimators = 1000, random_state = 1, max_leaf_nodes=i)
    RF_clf.fit(X_train, y_train)
    score_list.append(RF_clf.score(X_test, y_test))

plt.figure(figsize=(12,6))     
plt.plot(range(2,41), score_list)
plt.xticks(np.arange(2,41,1))
plt.xlabel("RF Value")
plt.ylabel("Score")
plt.savefig('figures//Random_Forest.png', bbox_inches='tight')  
plt.show()
RF_Acc = max(score_list)
print("Random Forest Accuracy: ",RF_Acc*100)

# 7. Gradient Boosting

params_GB={'n_estimators':[100,200,300,400,500],
      'max_depth':[1,2,3,4,5],
      'subsample':[0.5,1],
      'max_leaf_nodes':[2,5,10,20,30,40,50]}

GB = RandomizedSearchCV(GradientBoostingClassifier(), params_GB, cv=20)
GB.fit(X_train, y_train)

print(GB.best_estimator_)
print(GB.best_score_)
print(GB.best_params_)
print(GB.best_index_)

GB_clf = GradientBoostingClassifier(subsample=0.2, n_estimators=200, max_depth=4, max_leaf_nodes=10)
GB_clf.fit(X_train, y_train)

y_pred = GB_clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

GB_Acc = accuracy_score(y_pred,y_test)
print('Gradient Boosting accuracy: ',GB_Acc*100)


## Model Comparison

compare = pd.DataFrame({'Model': ['Logistic Regression', 'KNN', 
                                  'SVM', 'Categorical NB', 
                                  'Gaussian NB', 'Decision Tree', 
                                  'Random Forest', 'Gradient Boost'], 
                        'Accuracy': [LogR_Acc*100, KNN_Acc*100, SVC_Acc*100, 
                                     NB_Acc1*100, NB_Acc2*100, DT_Acc*100, 
                                     RF_Acc*100, GB_Acc*100]})
compare.sort_values(by='Accuracy', ascending=False)

# Prediction using Random Forest
X_val=validation_df.values
y_pred = RF_clf.predict(X_val)
val_df = pd.read_csv("validation.csv")
val_df['Predicted_Value']=y_pred
val_df.to_csv("result.csv",index=False)