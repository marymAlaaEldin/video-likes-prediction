import numpy as np
from sklearn import datasets, svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC
import pickle
import os
import time
from sklearn import metrics
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)


def Svm_model(X_train,X_test, Y_train,Y_test):
    ##inearSVC OneVsOne SVM
    filename1 = 'svm_linear.sav'
    if (os.path.exists(filename1)):
        loaded_model_1 = pickle.load(open(filename1, 'rb'))
        begin_time_test_1 = time.time()
        y_prediction_1 = loaded_model_1.predict(X_test)
        accuracy_1 = np.mean(y_prediction_1 == Y_test) * 100
        finish_time_test_1 = time.time()
    else:
        svm_linear_ovo = OneVsOneClassifier(LinearSVC(C=1,max_iter=1500))
        begin_time_train_1 = time.time()
        svm_linear_ovo.fit(X_train, Y_train)
        finish_time_train_1 = time.time()
        pickle.dump(svm_linear_ovo, open(filename1, 'wb'))
        begin_time_test_1 = time.time()
        y_prediction_1 = svm_linear_ovo.predict(X_test)
        accuracy_1 = np.mean(y_prediction_1 == Y_test) * 100
        finish_time_test_1 = time.time()
        time_training_1 = finish_time_train_1 - begin_time_train_1
        print(f"Runtime of the train LinearSVC OneVsOne SVM model is {time_training_1}")
    time_test_1 = finish_time_test_1 - begin_time_test_1
    print(f"Runtime of the test LinearSVC OneVsOne SVM model is {time_test_1} ")
    print('Model LinearSVC OneVsOne SVM Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction_1))

    ##RBF kernel
    filename2 = 'RBF kernel.sav'
    if (os.path.exists(filename2)):
        loaded_model_2 = pickle.load(open(filename2, 'rb'))
        begin_time_test_2 = time.time()
        y_prediction_2 = loaded_model_2.predict(X_test)
        accuracy_2 = np.mean(y_prediction_2 == Y_test) * 100
        finish_time_test_2 = time.time()
    else:
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=1)
        begin_time_train_2 = time.time()
        rbf_svc.fit(X_train, Y_train)
        finish_time_train_2 = time.time()
        pickle.dump(rbf_svc, open(filename2, 'wb'))
        begin_time_test_2 = time.time()
        y_prediction_2 = rbf_svc.predict(X_test)
        accuracy_2 = np.mean(y_prediction_2 == Y_test) * 100
        finish_time_test_2 = time.time()
        time_training_2 = finish_time_train_2 - begin_time_train_2
        print(f"Runtime of the train SVC with RBF kernel model is {time_training_2}")
    time_test_2 = finish_time_test_2 - begin_time_test_2
    print(f"Runtime of the test SVC with RBF kernel model is {time_test_2} ")
    print('Model SVC with RBF kernel Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction_2))

    ##SVC with polynomial kernel  degree 2
    filename3 = 'polynomial kernel degree=2.sav'
    if (os.path.exists(filename3)):
        loaded_model_3 = pickle.load(open(filename3, 'rb'))
        begin_time_test_3 = time.time()
        y_prediction_3 = loaded_model_3.predict(X_test)
        accuracy_3 = np.mean(y_prediction_3 == Y_test) * 100
        finish_time_test_3 = time.time()
    else:
        poly_svc = svm.SVC(kernel='poly', degree=2, C=1000)
        begin_time_train_3 = time.time()
        poly_svc.fit(X_train, Y_train)
        finish_time_train_3 = time.time()
        pickle.dump(poly_svc, open(filename3, 'wb'))
        begin_time_test_3 = time.time()
        y_prediction_3 = poly_svc.predict(X_test)
        accuracy_3 = np.mean(y_prediction_3 == Y_test) * 100
        finish_time_test_3 = time.time()
        time_training_3 = finish_time_train_3 - begin_time_train_3
        print(f"Runtime of the train SVC with polynomial kernel  degree 2 model is {time_training_3}")
    time_test_3 = finish_time_test_3 - begin_time_test_3
    print(f"Runtime of the test SVC with polynomial kernel  degree 2 model is {time_test_3} ")
    print('Model SVC with polynomial kernel  degree 2 Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction_3))

    ##SVC with polynomial kernel  degree 3
    filename4 = 'polynomial kernel degree=3.sav'
    if (os.path.exists(filename4)):
        loaded_model_4 = pickle.load(open(filename4, 'rb'))
        begin_time_test_4 = time.time()
        y_prediction_4 = loaded_model_4.predict(X_test)
        accuracy_4 = np.mean(y_prediction_4 == Y_test) * 100
        finish_time_test_4 = time.time()
    else:
        poly_svc_1 = svm.SVC(kernel='poly', degree=3, C=500)
        begin_time_train_4 = time.time()
        poly_svc_1.fit(X_train, Y_train)
        finish_time_train_4 = time.time()
        pickle.dump(poly_svc_1, open(filename4, 'wb'))
        begin_time_test_4 = time.time()
        y_prediction_4 = poly_svc_1.predict(X_test)
        accuracy_4 = np.mean(y_prediction_4 == Y_test) * 100
        finish_time_test_4 = time.time()
        time_training_4 = finish_time_train_4 - begin_time_train_4
        print(f"Runtime of the train SVC with polynomial kernel  degree 3 model is {time_training_4}")
    time_test_4 = finish_time_test_4 - begin_time_test_4
    print(f"Runtime of the test SVC with polynomial kernel  degree 3 model is {time_test_4} ")
    print('Model SVC with polynomial kernel  degree 3 Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction_4))

    ##SVC with polynomial kernel  degree 4
    filename5 = 'polynomial kernel degree=4.sav'
    if (os.path.exists(filename5)):
        loaded_model_5 = pickle.load(open(filename5, 'rb'))
        begin_time_test_5 = time.time()
        y_prediction_5 = loaded_model_5.predict(X_test)
        accuracy_5 = np.mean(y_prediction_5 == Y_test) * 100
        finish_time_test_5 = time.time()
    else:
        poly_svc_2 = svm.SVC(kernel='poly', degree=4, C=500)
        begin_time_train_5 = time.time()
        poly_svc_2.fit(X_train, Y_train)
        finish_time_train_5 = time.time()
        pickle.dump(poly_svc_2, open(filename5, 'wb'))
        begin_time_test_5 = time.time()
        y_prediction_5 = poly_svc_2.predict(X_test)
        accuracy_5 = np.mean(y_prediction_5 == Y_test) * 100
        finish_time_test_5 = time.time()
        time_training_5= finish_time_train_5- begin_time_train_5
        print(f"Runtime of the train SVC with polynomial kernel  degree 4 model is {time_training_5}")
    time_test_5 = finish_time_test_5 - begin_time_test_5
    print(f"Runtime of the test SVC with polynomial kernel  degree 4 model is {time_test_5} ")
    print('Model SVC with polynomial kernel  degree 4 Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction_5))

    return accuracy_1,accuracy_2,accuracy_3,accuracy_4,accuracy_5





