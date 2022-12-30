from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn import linear_model, metrics
import pickle
import os
import time
import numpy as np
def train_model_1(x_train, y_train, x_test, y_test):
    ##degree 2
    filename = 'polynomial_regression_degree 2.sav'
    model_1_poly_features = PolynomialFeatures(degree=2)
    X_train_poly_model_1 = model_1_poly_features.fit_transform(x_train)
    X_test_poly_model_1 = model_1_poly_features.fit_transform(x_test)
    if (os.path.exists(filename)):
        loaded_model = pickle.load(open(filename, 'rb'))
        begin_time_test = time.time()
        prediction_1 = loaded_model.predict(X_train_poly_model_1)
        prediction_2 = loaded_model.predict(X_test_poly_model_1)
        finish_time_test = time.time()
        scores = cross_val_score(loaded_model, X_train_poly_model_1, y_train, cv=5, scoring='neg_mean_squared_error')
    else:
        poly_model1 = linear_model.LinearRegression()
        begin_time_train = time.time()
        poly_model1.fit(X_train_poly_model_1, y_train)
        finish_time_train = time.time()
        pickle.dump(poly_model1, open(filename, 'wb'))
        begin_time_test = time.time()
        prediction_1 = poly_model1.predict(X_train_poly_model_1)
        prediction_2 = poly_model1.predict(X_test_poly_model_1)
        finish_time_test = time.time()
        time_training = finish_time_train - begin_time_train
        print(f"Runtime of the train polynomial_regression degree=2 model is {time_training}")
        scores = cross_val_score(poly_model1, X_train_poly_model_1, y_train, cv=5,scoring='neg_mean_squared_error')
    time_test = finish_time_test - begin_time_test
    print(f"Runtime of the test polynomial_regression degree=2 model : {time_test} ")
    print('Model polynomial_regression degree=2 Cross Validation  scores : ', abs(scores.mean()))
    print('Model polynomial_regression degree=2 train Mean Square Error : ',metrics.mean_squared_error(y_train, prediction_1))
    print('Model polynomial_regression degree=2 test Mean Square Error : ', metrics.mean_squared_error(y_test, prediction_2))




