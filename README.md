### FINAL EXAM
## MIGUEL MEDRANO
## Librerias requeridas
#!conda update -n base conda -y
#!conda install -c conda-forge pydotplus -y
#!conda install -c conda-forge python-graphviz -y
#!conda install scikit-learn -y

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import sklearn.tree as tree
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
%matplotlib inline

## Loan_train.csv

!wget -O loan_train.csv htps://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv
df = pd.read_csv('loan_train.csv')
df.head()
df.shape
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()

## Data visualization and pre-processing

df['loan_status'].value_counts()
# notice: installing seaborn might takes a few minutes
#!conda install -c anaconda seaborn -y
import seaborn as sns
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

## Pre-processing: Feature selection/extraction

df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()
df.groupby(['education'])['loan_status'].value_counts(normalize=True)
df[['Principal','terms','age','Gender','education']].head()
Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()
X = Feature
X[0:5]
y = df['loan_status'].values
y[0:5]

## Normalizando los datos

X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

## Clasificacion
### K Nearest Neighbor(KNN)
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
Ks = 15
#create array of accuracy scores, or how similar y is to yhat
mean_acc = np.zeros((Ks-1))

#Standard Deviation of the scores
standard_deviation_acc = np.zeros((Ks-1))

#f1 scores for each acc score
f1_scores = np.zeros((Ks-1))

#jaccard scores for each acc score
jac_scores = np.zeros((Ks-1))

#Try 1 to 14 points 
for k in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    yhat = neigh.predict(X_test)    
    #append accuracy
    mean_acc[k-1] = metrics.accuracy_score(y_test, yhat)    
    #append std
    standard_deviation_acc[k-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])  
    #append f1
    f1_scores[k-1] = f1_score(y_test, yhat, average='weighted')
    #append jaccard
    jac_scores[k-1] = jaccard_score(y_test, yhat, pos_label='PAIDOFF')
    print("k=",k,"   Accuracy: ", metrics.accuracy_score(y_test, yhat))
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * standard_deviation_acc,mean_acc + 1 * standard_deviation_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * standard_deviation_acc,mean_acc + 3 * standard_deviation_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
#F1 score and Jaccard for best accuracy
knnF1 = f1_scores[mean_acc.argmax()]
knnJac = jac_scores[mean_acc.argmax()]
print("Avg F1-score: %.4f" % knnF1)
print("Jaccard score: %.4f" % knnJac)

### Arbol de desicion

decisionTree = DecisionTreeClassifier(criterion="entropy", max_depth = 6)
decisionTree.fit(X_train,y_train)
predTree = decisionTree.predict(X_test)
#Test multiple depths, and they all reach the same accuracy score
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
decisionTreeF1 = f1_score(y_test, predTree, average='weighted')
decisionTreeJac = jaccard_score(y_test, predTree, pos_label='PAIDOFF')
print("Avg F1-score: %.4f" % decisionTreeF1)
print("Jaccard score: %.4f" % decisionTreeJac)
#!pip install --upgrade sklearn
plt.figure(figsize=[12,12])
tree.plot_tree(decisionTree)
plt.show()

### Maquinas de vectores de soporte

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']
#create array of accuracy scores, or how similar y is to yhat
mean_acc = np.zeros((4-1))
f1_scores = np.zeros((4-1))
jac_scores = np.zeros((4-1))
for km in range(1,len(kernel_methods)):
    clf = svm.SVC(kernel=kernel_methods[km])
    clf.fit(X_train, y_train) 
    yhat = clf.predict(X_test)    
    #append accuracy
    mean_acc[km-1] = metrics.accuracy_score(y_test, yhat)        
    f1_scores[km-1] = f1_score(y_test, yhat, average='weighted')    
    jac_scores[km-1] = jaccard_score(y_test, yhat, pos_label='PAIDOFF')
print("The best accuracy was with", mean_acc.max(), "with kernel =", kernel_methods[mean_acc.argmax()]) 
svmF1 = f1_scores[mean_acc.argmax()]
svmJac = jac_scores[mean_acc.argmax()]
print("Avg F1-score: %.4f" % svmF1)
print("Jaccard score: %.4f" % svmJac)

### Regresion Logistica

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
#create array of accuracy scores, or how similar y is to yhat
mean_acc = np.zeros((5-1))
f1_scores = np.zeros((5-1))
jac_scores = np.zeros((5-1))
logloss_scores = np.zeros((5-1))
for s in range(1,len(solvers)):
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
    yhat = LR.predict(X_test)
    yhat_prob = LR.predict_proba(X_test)    
    #append accuracy
    mean_acc[s-1] = metrics.accuracy_score(y_test, yhat)        
    f1_scores[s-1] = f1_score(y_test, yhat, average='weighted')    
    jac_scores[s-1] = jaccard_score(y_test, yhat, pos_label='PAIDOFF')    
    logloss_scores[s-1] = log_loss(y_test, yhat_prob)
    print("The best accuracy was with", mean_acc.max(), "with solver =", solvers[mean_acc.argmax()])
lrF1 = f1_scores[mean_acc.argmax()]
lrJac = jac_scores[mean_acc.argmax()]
lrLogLoss = logloss_scores[mean_acc.argmax()]
print("Avg F1-score: %.4f" % lrF1)
print("Jaccard score: %.4f" % lrJac)
print("Log Loss score: %.4f" % lrLogLoss)

### Enaluacion del modelo con prueba

from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv
test_df = pd.read_csv('loan_test.csv')
test_df.head()
#pip install tabulate
from tabulate import tabulate
report = [['', 'Jaccard', 'F1-Score', 'LogLoss'],
          ['k-nearest neighbors', knnJac, knnF1, 'NA'],
          ['Decision Tree', decisionTreeJac, decisionTreeF1, 'NA'],
          ['Support Vector Machine', svmJac, svmF1, 'NA'],
          ['Logistic Regression', lrJac, lrF1, lrLogLoss]]
print(tabulate(report, headers='firstrow'))
