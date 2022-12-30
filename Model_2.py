from sklearn.model_selection import cross_val_score
from sklearn import linear_model, metrics
import time
import os
import pickle
import numpy as np

def train_model_2(x_train, y_train, x_test, y_test):
    filename = 'multi_linear_regression.sav'
    if (os.path.exists(filename)):
        loaded_model = pickle.load(open(filename, 'rb'))
        begin_time_test = time.time()
        prediction_1 = loaded_model.predict(x_train)
        prediction_2 = loaded_model.predict(x_test)
        finish_time_test = time.time()
        scores = cross_val_score(loaded_model, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
    else:
        model_2_multi = linear_model.LinearRegression()
        begin_time_train = time.time()
        model_2_multi.fit(x_train, y_train)
        finish_time_train = time.time()
        pickle.dump(model_2_multi, open(filename, 'wb'))
        begin_time_test = time.time()
        prediction_1 = model_2_multi.predict(x_train)
        prediction_2 = model_2_multi.predict(x_test)
        finish_time_test = time.time()
        time_training = finish_time_train - begin_time_train
        print(f"Runtime of the train multi_linear_regression model is {time_training}")
        scores = cross_val_score(model_2_multi, x_train, y_train, cv=5,scoring='neg_mean_squared_error')
    time_test = finish_time_test - begin_time_test
    print(f"Runtime of the test multi_linear_regression model is {time_test} ")
    print('Model multi_linear_regression Cross Validation  scores: ', abs(scores.mean()))
    print('Model multi_linear_regression  train Mean Square Error : ', metrics.mean_squared_error(y_train, prediction_1))
    print('Model multi_linear_regression test Mean Square Error : ', metrics.mean_squared_error(y_test, prediction_2))
