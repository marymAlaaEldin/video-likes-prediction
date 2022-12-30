from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os
import time
from numpy import mean
from sklearn import metrics
def random_forest(X_train,X_test, Y_train,Y_test):
    filename = 'RandomForest.sav'
    if (os.path.exists(filename)):
        loaded_model_1 = pickle.load(open(filename, 'rb'))
        begin_time_test=time.time()
        y_prediction = loaded_model_1.predict(X_test)
        accuracy = np.mean(y_prediction == Y_test) * 100
        finish_time_test = time.time()
    else:
        random_f = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=101,
                                          max_features=None, min_samples_leaf=1)
        begin_time_train=time.time()
        random_f.fit(X_train, Y_train)
        finish_time_train=time.time()
        pickle.dump(random_f, open(filename, 'wb'))
        begin_time_test = time.time()
        y_prediction = random_f.predict(X_test)
        accuracy = np.mean(y_prediction == Y_test) * 100
        finish_time_test = time.time()
        time_training = finish_time_train - begin_time_train
        print(f"Runtime of the train Random Forest model is {time_training}")
    time_test = finish_time_test - begin_time_test
    print(f"Runtime of the test Random Forest model is {time_test} ")
    print('Model Random Forest Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction))
    return accuracy