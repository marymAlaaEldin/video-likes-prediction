import numpy as np
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle
import os
import time
from sklearn import metrics
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

def DecisionTree_model(X_train,X_test, Y_train,Y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    filename = 'Decision_Tree.sav'
    if (os.path.exists(filename)):
        loaded_model_1 = pickle.load(open(filename, 'rb'))
        begin_time_test_tree=time.time()
        y_prediction = loaded_model_1.predict(X_test)
        accuracy = np.mean(y_prediction == Y_test) * 100
        finish_time_test_tree = time.time()
    else:
        tree_decision = tree.DecisionTreeClassifier(max_depth=12)
        begin_time_train_tree=time.time()
        tree_decision.fit(X_train, Y_train)
        finish_time_train_tree = time.time()
        pickle.dump(tree_decision, open(filename, 'wb'))
        begin_time_test_tree = time.time()
        y_prediction = tree_decision.predict(X_test)
        accuracy = np.mean(y_prediction == Y_test) * 100
        finish_time_test_tree = time.time()
        tranning_time_tree = finish_time_train_tree - begin_time_train_tree
        print(f"Runtime of the train Tree Decision model is {tranning_time_tree}")
    time_test_tree=finish_time_test_tree-begin_time_test_tree
    print(f"Runtime of the test Tree Decision model is {time_test_tree}")
    print('Model Tree Decision Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction))
    filename1 = 'AdaBoost_Decision_Tree.sav'
    if (os.path.exists(filename1)):
        loaded_model_1 = pickle.load(open(filename1, 'rb'))
        begin_time_test_adaboost=time.time()
        y_prediction = loaded_model_1.predict(X_test)
        accuracy1 = np.mean(y_prediction == Y_test) * 100
        finish_time_test_adaboost = time.time()
    else:
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),
                                 algorithm="SAMME",
                                 n_estimators=100)
        begin_time_train_adaboost=time.time()
        bdt.fit(X_train, Y_train)
        finish_time_train_adaboost = time.time()
        pickle.dump(bdt, open(filename1, 'wb'))
        begin_time_test_adaboost = time.time()
        y_prediction = bdt.predict(X_test)
        accuracy1 = np.mean(y_prediction == Y_test) * 100
        finish_time_test_adaboost = time.time()
        tranning_time_adboost=finish_time_train_adaboost-begin_time_train_adaboost
        print(f"Runtime of the train AdaBoost with Tree Decision model is {tranning_time_adboost}")
    time_test_adaboost = finish_time_test_adaboost - begin_time_test_adaboost
    print(f"Runtime of the test AdaBoost with Tree Decision model is {time_test_adaboost}")
    print('Model AdaBoost with Tree Decision Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction))

    return accuracy,accuracy1