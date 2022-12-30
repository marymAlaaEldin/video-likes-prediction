from sklearn.naive_bayes import GaussianNB
import numpy as np
import pickle
import os
import time
from sklearn import metrics
def gaussian_nb(X_train,X_test, Y_train,Y_test):
    filename = 'GaussianNB.sav'
    if (os.path.exists(filename)):
        loaded_model_1 = pickle.load(open(filename, 'rb'))

        begin_time_test = time.time()
        y_prediction = loaded_model_1.predict(X_test)
        accuracy = np.mean(y_prediction == Y_test) * 100
        finish_time_test = time.time()
    else:
        nb = GaussianNB()
        begin_time_train = time.time()
        nb.fit(X_train, Y_train)
        finish_time_train = time.time()

        pickle.dump(nb, open(filename, 'wb'))
        begin_time_test = time.time()
        y_prediction = nb.predict(X_test)
        accuracy = np.mean(y_prediction == Y_test) * 100
        finish_time_test = time.time()
        time_training = finish_time_train - begin_time_train
        print(f"Runtime of the train GaussianNB model is {time_training}")
    time_test = finish_time_test - begin_time_test
    print(f"Runtime of the test GaussianNB model is {time_test} ")
    print('Model GaussianNB Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction))
    return accuracy
