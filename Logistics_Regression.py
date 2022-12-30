from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import os
from sklearn import metrics
import time
def logistic_regression(X_train,X_test, Y_train,Y_test):
    filename = 'Logistics_regression.sav'
    if (os.path.exists(filename)):
        loaded_model_1 = pickle.load(open(filename, 'rb'))
        begin_time_test=time.time()
        y_prediction = loaded_model_1.predict(X_test)
        accuracy = np.mean(y_prediction == Y_test) * 100
        finish_time_test = time.time()
    else:
        lr = LogisticRegression(solver='lbfgs', max_iter=800, C=0.1, multi_class='multinomial')
        begin_time_train = time.time()
        lr.fit(X_train, Y_train)
        finish_time_train = time.time()
        pickle.dump(lr, open(filename, 'wb'))
        begin_time_test = time.time()
        y_prediction = lr.predict(X_test)
        accuracy = np.mean(y_prediction == Y_test) * 100
        finish_time_test = time.time()
        time_training = finish_time_train - begin_time_train
        print(f"Runtime of the train Logistics regression model is {time_training}")
    time_test = finish_time_test - begin_time_test
    print(f"Runtime of the test Logistics regression model is {time_test} ")
    print('Model Logistics regression Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction))
    return  accuracy