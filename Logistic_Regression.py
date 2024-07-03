#TRAIN/SPLIT
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2021XPT\diabetes_health_indicators_BRFSS2021_v21.csv")

# Diabetes is the dependent value
y = df.iloc[:,0]
x = df


# From the study being replicated, Training was 2/3 and Test was remaining 1/3
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33333333, random_state=0)

#Logistic Regression
SC = StandardScaler()
xtrain = SC.fit_transform(xtrain)
xtest = SC.transform(xtest)
  
LR = LogisticRegression()
LR.fit(xtrain, ytrain)

y_pred_LR = LR.predict(xtest)

print(y_pred_LR)

#KNN
knn = KNeighborsClassifier(n_neighbors=7) 
  
knn.fit(xtrain, ytrain)
  
y_pred_KNN = knn.predict(xtest)

print(y_pred_KNN)

#Confusion Matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  
CM_LR = confusion_matrix(ytest, y_pred_LR)

disp = ConfusionMatrixDisplay(confusion_matrix=CM_LR)

disp.plot()

#KNN
  
CM_KNN = confusion_matrix(ytest, y_pred_KNN)

disp = ConfusionMatrixDisplay(confusion_matrix=CM_KNN)

disp.plot()
