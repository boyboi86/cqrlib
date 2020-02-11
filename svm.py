import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

p = print

#----- SVM Classifer model to train -------------------------------------------
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, test_size = 0.33, random_state = 42)
model = SVC(probability=True)
model.fit(Train_X, Train_Y)
predictions = model.predict(Test_X)

p('=======================================================')
p('Confusion Matrix\n')
p(confusion_matrix(Test_Y, predictions))
p('=======================================================')
p(classification_report(Test_Y, predictions))
p('=======================================================')

#-------- optimization gamma vs C-value ---------------------------------------
# ------- warning! can take forever if use linspace----------------------------

param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(model, param_grid, verbose = 3)
grid.fit(Train_X, Train_Y)
p('=======================================================')
p('Optimized Results')
p(grid.best_params_)
p('=======================================================')
p(grid.best_estimator_)
p('=======================================================')
p('Optimized score')
p(grid.best_score_)
p('=======================================================')

#---------- New prediction based on best param estimators ---------------------
grid_prediction = grid.predict(Test_X)
p('=======================================================')
p('Confusion Matrix\n')
p(confusion_matrix(Test_Y, grid_prediction))
p('=======================================================')
p(classification_report(Test_Y, grid_prediction))
p('=======================================================')

#---------- graphically prediction error --------------------------------------
#cross_val_predict(SVC(), Test_X, Test_Y)

fig, ax =plt.subplots()
ax.scatter(Test_Y, grid_prediction, color='r')
ax.plot([Test_Y.min(), Test_Y.max()], [grid_prediction.min(), grid_prediction.max()], 'k--', lw=4)
ax.set_xlabel('Actual Results')
ax.set_ylabel('Prediction')
ax.set_title('scatter plot')
plt.show()

