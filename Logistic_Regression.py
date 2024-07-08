#TRAIN/SPLIT
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
from sklearn.metrics import precision_recall_fscore_support,ConfusionMatrixDisplay,accuracy_score

df = pd.read_csv(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2021XPT\diabetes_health_indicators_BRFSS2021_v21.csv")

#Showing current imbalance
"""x=df.drop(["Diabetes"],axis=1)
y=df["Diabetes"]

count_class = y.value_counts() # Count the occurrences of each class
plt.bar(count_class.index, count_class.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(count_class.index, ['Diabetes 0', 'Diabetes 1'])
plt.show()"""

# Diabetes is the dependent value
X = df.iloc[:, df.columns != 'Diabetes']
y = df.iloc[:, df.columns == 'Diabetes']

# Apply SMOTE to create observations for Diabetes
# Source:https://github.com/scikit-learn-contrib/imbalanced-learn
from imblearn.over_sampling import SMOTE #Over sampling
sm = SMOTE(sampling_strategy='minority')
X_sampled,y_sampled = sm.fit_resample(X,y.values.ravel())
y.value_counts()

# Confirm values are similar now
print(np.count_nonzero(y_sampled == 0))
print(np.count_nonzero(y_sampled == 1))

#Percentage of diabetes in original data
Source_data_no_diabetes_count = len(df[df.Diabetes==0])
Source_data_diabetes_count = len(df[df.Diabetes==1])
print('Percentage of diabetes in original dataset:{}%'.format((Source_data_diabetes_count*100)/(Source_data_no_diabetes_count+Source_data_diabetes_count)))

#Percentage of diabetes in sampled data
Sampled_data_no_diabetes_count = len(y_sampled[y_sampled==0])
Sampled_data_diabetes_count = len(y_sampled[y_sampled==1])
print('Percentage of diabetes in the new data:{}%'.format((Sampled_data_diabetes_count*100)/(Sampled_data_no_diabetes_count+Sampled_data_diabetes_count)))

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

print("Number diabetes train dataset: ", len(X_train))
print("Number diabetes test dataset: ", len(X_test))
print("Total number of diabetes: ", len(X_train)+len(X_test))

# Undersampled dataset
X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(X_sampled
                                                                                                   ,y_sampled
                                                                                                   ,test_size = 1/3
                                                                                                   ,random_state = 0)
print("")
print("Number diabetes train dataset: ", len(X_train_sampled))
print("Number diabetes test dataset: ", len(X_test_sampled))
print("Total number of diabetes: ", len(X_train_sampled)+len(X_test_sampled))

X_train_sampled_df = pd.DataFrame(X_train_sampled)
y_train_sampled_df = pd.DataFrame(y_train_sampled)
X_test_sampled_df = pd.DataFrame(X_test_sampled)
y_test_sampled_df = pd.DataFrame(y_test_sampled)


#y = df.iloc[:,0]
#x = df
#X = list(set(list(df)) - set(['Diabetes']))

# From the study being replicated, Training was 2/3 and Test was remaining 1/3
#xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33333333, random_state=0)

#xtrain, xtest, ytrain, ytest = train_test_split(df[X], df['Diabetes'], test_size=0.35, random_state=42)

# Logistic Regression

SC = StandardScaler()
xtrain = SC.fit_transform(X_train_sampled)
xtest = SC.transform(X_test_sampled)
  
LR = LogisticRegression()
LR.fit(xtrain, y_train_sampled)

y_pred_LR = LR.predict(xtest)

print(y_pred_LR)


#Confusion Matrix - Logistic Regression

CM_LR = confusion_matrix(y_test_sampled, y_pred_LR)

disp = ConfusionMatrixDisplay(confusion_matrix=CM_LR)

disp.plot()

precision, recall, fscore, support = precision_recall_fscore_support(y_test_sampled, y_pred_LR)

accuracy = accuracy_score(y_test_sampled, y_pred_LR)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
print('accuracy: {}'.format(accuracy))


# KNN
knn = KNeighborsClassifier(n_neighbors=7) 
  
knn.fit(xtrain, y_train_sampled)
  
y_pred_KNN = knn.predict(xtest)

print(y_pred_KNN)

# Confusion Matrix for KNN

CM_LR = confusion_matrix(y_test_sampled, y_pred_LR)

disp = ConfusionMatrixDisplay(confusion_matrix=CM_LR)

disp.plot()
  
CM_KNN = confusion_matrix(y_test_sampled, y_pred_KNN)

disp = ConfusionMatrixDisplay(confusion_matrix=CM_KNN)

disp.plot()


