
####------------------------Importing data------------------------#############
import pandas as pd
import numpy as np
data = pd.read_csv("C:/Users/Admin/Downloads/XYZCorp_LendingData.txt", encoding='utf-8',sep='\t')

#Looking at Data
data.describe()
data.head()

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

#Droping unnecessary variables which dont have much impact on Dependent variable
data2 = data.drop(['addr_state',
'annual_inc_joint',
'collections_12_mths_ex_med',
'desc',
'dti_joint',
'earliest_cr_line',
'emp_length',
'emp_title',
'funded_amnt',
'id',
'inq_last_6mths',
'last_pymnt_d',
'member_id',
'next_pymnt_d',
'out_prncp_inv',
'policy_code',
'pub_rec',
'revol_util',
'sub_grade',
'title',
'total_acc',
'total_pymnt',
'total_rec_int',
'total_rec_late_fee',
'total_rec_prncp',
'verification_status_joint',
'zip_code',
'open_acc_6m',
'open_il_6m',
'open_il_12m',
'il_util',
'open_rv_12m',
'open_rv_24m',
'inq_fi',
'total_cu_tl',
'inq_last_12m',
'tot_coll_amt',
],axis=1)

#Checking for N.A.'s in our dataset
missval = data2.isna().sum()/int(len(data2))

#Removing NA's having <75%
data3 = data2.drop(missval[missval>0.75].index,axis=1)

#Replacing NA with median
data3['mths_since_last_delinq'].median()
data3['mths_since_last_delinq'].fillna(data3['mths_since_last_delinq'].median(),inplace=True)

data3['tot_cur_bal'].median()
data3['tot_cur_bal'].fillna(data3['tot_cur_bal'].median(),inplace=True)

data3['total_rev_hi_lim'].median()
data3['total_rev_hi_lim'].fillna(data3['total_rev_hi_lim'].median(),inplace=True)

#Dropping variable "last_credit_pull_d"
data4 = data3.drop(['last_credit_pull_d'],axis=1)

#Final check for NA's
data4.isna().sum()

#Converting term variable = 36 month into term = 30, i.e, in integer
data4['term'] = data4['term'].str.split(' ').str[1]

#Treating categorical variables

colname= ['grade','home_ownership','verification_status','purpose','initial_list_status','pymnt_plan','application_type']
data4 = pd.get_dummies(data4,columns = colname)

#Plotting Histogram

plt.hist(data3['default_ind'])
plt.xlabel('default indicator')
plt.ylabel('No. of borrower')
plt.title('Chart of Loan credit defaulters')
plt.show()

plt.hist(data4['verification_status_Source Verified'], bins=10)
plt.xlabel('Verified_status')
plt.ylabel('No. of issuer')
plt.title('Chart of Loan Verified Status')
plt.show()

plt.hist(data4['verification_status_Not Verified'], bins=10)
plt.xlabel('Not Verified status')
plt.ylabel('No. of issuer')
plt.title('Chart of Loan Not-Verified Status')
plt.show()

plt.hist(data4['delinq_2yrs'], bins=10)
plt.xlabel('Delinquency')
plt.ylabel('No. of issuer')
plt.title('Chart of delinquency in the borrowers credit file for the past 2 years')
plt.show()
#####----------------------Performing EDA--------------------##################
   
#Univaraite Analysis

import seaborn as sns
sns.boxplot(data4['loan_amnt'])
sns.boxplot(data4['funded_amnt_inv'])
sns.boxplot(data4['installment'])

#Outlier Treatment
sns.boxplot(data4['int_rate'])
data4['int_rate'].describe()
data4['int_rate'] = np.where(data4['int_rate']>25,25,data4['int_rate'])

sns.boxplot(data4['installment'])
data4['installment'].describe()
data4['installment'] = np.where(data4['installment']>1038,958,data4['int_rate'])

sns.boxplot(data4['annual_inc'])
data4['annual_inc'].describe()
data4['annual_inc'] = np.where(data4['annual_inc']>157500,145000,data4['annual_inc'])

sns.boxplot(data4['dti'])
data4['dti'].describe()
data4['dti'] = np.where(data4['dti']>42,35,data4['annual_inc'])

sns.boxplot(data4['mths_since_last_delinq'])
data4['mths_since_last_delinq'].describe()
data4['mths_since_last_delinq'] = np.where(data4['mths_since_last_delinq']>65,65,data4['mths_since_last_delinq'])

sns.boxplot(data4['open_acc'])
data4['open_acc'].describe()
data4['open_acc'] = np.where(data4['open_acc']>23,22,data4['open_acc'])

sns.boxplot(data4['revol_bal'])
data4['revol_bal'].describe()
data4['revol_bal'] = np.where(data4['revol_bal']>42439,40944,data4['revol_bal'])

sns.boxplot(data4['total_pymnt_inv'])
data4['total_pymnt_inv'].describe()
data4['total_pymnt_inv'] = np.where(data4['total_pymnt_inv']>23802,22645,data4['total_pymnt_inv'])

sns.boxplot(data4['last_pymnt_amnt'])
data4['last_pymnt_amnt'].describe()
data4['last_pymnt_amnt'] = np.where(data4['last_pymnt_amnt']>1694,14866,data4['last_pymnt_amnt'])


#Bivariate Analysis

sns.boxplot(x = 'default_ind', y = 'int_rate', data = data4) ##Default_ind is high for customers who are getting high interest rate.
sns.boxplot(x = 'default_ind', y = 'loan_amnt', data = data4) 
sns.boxplot(x = 'default_ind', y = 'installment', data = data4) 
sns.boxplot(x = 'default_ind', y = 'annual_inc', data = data4) ##We can observe customers having income between 40K-70K are major defaulters.
sns.boxplot(x = 'default_ind', y = 'dti', data = data4) 
sns.boxplot(x = 'default_ind', y = 'loan_amnt', data = data4) 
sns.boxplot(x = 'default_ind', y = 'funded_amnt_inv', data = data4) 
sns.boxplot(x = 'default_ind', y = 'revol_bal', data = data4) 
sns.boxplot(x = 'default_ind', y = 'total_pymnt_inv', data = data4) 



#########---------------------Splitting data-----------------------############
var_list =  ['Jun-2015', 'Jul-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015']

train_original = data4.loc [ -data4.issue_d.isin(var_list) ]
test_original = data4.loc [ data4.issue_d.isin(var_list) ]

#Dropping issue_d column
train_original = train_original.drop(['issue_d'],axis = 1)
test_original = test_original.drop(['issue_d'],axis = 1)

###------Dividing independent variables and dependent variable--------#########

#Independent Var
X = train_original.drop(['default_ind'], axis = 1)
#Dependent Var
Y = np.vstack(train_original['default_ind'].values) 

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)


##########------------Modelling with logistic Regression-----------############

log_reg = LogisticRegression(solver='liblinear',max_iter=10000,random_state=123)

#Fitting model
log_reg.fit(x_train,y_train)

#Predicting model
y_pred_Log = log_reg.predict(x_test)

#Checking Accuracy
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

print(confusion_matrix(y_test,y_pred_Log))
print(accuracy_score(y_test,y_pred_Log))
print(classification_report(y_test,y_pred_Log))
#Accuracy ==== 99%

#Plotting ROC curve
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_Log)
auc = metrics.auc(fpr,tpr)
print(auc)

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


##########--------------Modelling with Decision Tree------------###############

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

#Fitting model
clf = clf.fit(x_train,y_train)

#Predicting model
y_pred = clf.predict(x_test)

#Checking Accuracy of model
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
#Accuracy ===== 99.3%

#Plotting ROC curve
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr,tpr)
print(auc)

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#########---------------Modelling with Random Forest---------------############

from sklearn.ensemble import RandomForestClassifier

model_RandomForest=RandomForestClassifier(15)

#Fitting model
model_RandomForest.fit(x_train,y_train)

#Predicting model
y_pred1=model_RandomForest.predict(x_test)

#Checking Accuracy of model
print(confusion_matrix(y_test,y_pred1))
print(accuracy_score(y_test,y_pred1))
print(classification_report(y_test,y_pred1))
#Accuracy ==== 99.3%

#Plotting ROC curve
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred1)
auc = metrics.auc(fpr,tpr)
print(auc)

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


##########--------------- Modelling with KNN----------------------############

from sklearn.neighbors import KNeighborsClassifier
knnclassifier = KNeighborsClassifier(n_neighbors = 5)

#Fitting model
knnclassifier.fit(x_train,y_train)

#Predicting model
y_pred2 = knnclassifier.predict(x_test)

#Checking Accuracy of model
print(confusion_matrix(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
#Accuracy ==== 96.7%

#Plotting ROC curve
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred2)
auc = metrics.auc(fpr,tpr)
print(auc)

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



##############-----------------CROSS VALIDATION--------------------############

from sklearn.model_selection import cross_val_score
knnclassifier = KNeighborsClassifier(n_neighbors = 5)
print(cross_val_score(knnclassifier,X,Y,cv=50,scoring="accuracy").mean())
##96.8%

log_reg = LogisticRegression()
print(cross_val_score(log_reg,X,Y,cv=50,scoring="accuracy").mean())
###99%

model_RandomForest=RandomForestClassifier(15)
print(cross_val_score(model_RandomForest,X,Y,cv=50,scoring="accuracy").mean())
###99.3%

clf = DecisionTreeClassifier()
print(cross_val_score(clf,X,Y,cv=50,scoring="accuracy").mean())
###99.2%


###Therefore by looking at the mean values from cross validation we can conclude that Random Forest is the best model for this dataset.


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('log', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 =  RandomForestClassifier()
estimators.append(('ID3', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#print(Y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

















