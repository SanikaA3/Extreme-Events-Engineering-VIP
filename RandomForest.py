import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score
#Classification just based on words

data = []
# Read the training data
Xy = np.loadtxt('Datafile_updated.txt',skiprows=1)
y=Xy[:,-1]
X=Xy[:,0:-1]

scores = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RandomState())
rf = RandomForestRegressor(oob_score=False, max_features='log2',n_estimators=10)
myrf=rf.fit(X_train, y_train)

#Cross-Validation Scores
scores=cross_val_score(myrf,X_train, y_train,scoring='r2',cv=10)
print('Cross-Val-mean_r2_square: ',scores.mean())
print('Cross-Val-std_r2_square: ' ,scores.std())


y_pred=rf.predict(X_test)
error=r2(y_test,y_pred)
print('R2 Score: ', error)


