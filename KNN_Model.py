from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import os
import time
from sklearn import metrics
def knn_model(X_train,X_test, Y_train,Y_test):
    filename1 = 'KNN1.sav'
    if (os.path.exists(filename1)):
        loaded_model_1 = pickle.load(open(filename1, 'rb'))
        begin_time_test_1=time.time()
        y_prediction_1 = loaded_model_1.predict(X_test)
        accuracy1 = np.mean(y_prediction_1 == Y_test) * 100
        finish_time_test_1 = time.time()
    else:
        knn1 = KNeighborsClassifier(n_neighbors=17, leaf_size=100)
        begin_time_train_1 = time.time()
        knn1.fit(X_train, Y_train)
        finish_time_train_1 = time.time()
        pickle.dump(knn1, open(filename1, 'wb'))
        begin_time_test_1 = time.time()
        y_prediction_1 = knn1.predict(X_test)
        accuracy1 = np.mean(y_prediction_1 == Y_test) * 100
        finish_time_test_1 = time.time()
        time_training_1 = finish_time_train_1 - begin_time_train_1
        print(f"Runtime of the train KNN  with K=17  model is {time_training_1}")
    time_test_1 = finish_time_test_1 - begin_time_test_1
    print(f"Runtime of the test KNN  with K=17 model is {time_test_1} ")
    print('Model KNN  with K=17 Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction_1))

    filename2 = 'KNN2.sav'
    if (os.path.exists(filename2)):
        loaded_model_2 = pickle.load(open(filename2, 'rb'))
        begin_time_test_2=time.time()
        y_prediction_2 = loaded_model_2.predict(X_test)
        accuracy2 = np.mean(y_prediction_2 == Y_test) * 100
        finish_time_test_2 = time.time()
    else:
        knn2 = KNeighborsClassifier(n_neighbors=3, leaf_size=30)
        begin_time_train_2=time.time()
        knn2.fit(X_train, Y_train)
        finish_time_train_2 = time.time()
        pickle.dump(knn2, open(filename2, 'wb'))
        begin_time_test_2 = time.time()
        y_prediction_2 = knn2.predict(X_test)
        accuracy2 = np.mean(y_prediction_2 == Y_test) * 100
        finish_time_test_2 = time.time()
        time_training_2 = finish_time_train_2 - begin_time_train_2
        print(f"Runtime of the train KNN  with K=3  model is {time_training_2}")
    time_test_2 = finish_time_test_2 - begin_time_test_2
    print(f"Runtime of the test KNN  with K=3 model is {time_test_2} ")
    print('Model KNN  with K=3 Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction_2))

    filename3 = 'KNN3.sav'
    if (os.path.exists(filename3)):
        loaded_model_3 = pickle.load(open(filename3, 'rb'))
        begin_time_test_3=time.time()
        y_prediction_3 = loaded_model_3.predict(X_test)
        accuracy3 = np.mean(y_prediction_3 == Y_test) * 100
        finish_time_test_3 = time.time()
    else:
        knn3 = KNeighborsClassifier(n_neighbors=51, algorithm='auto', leaf_size=30, n_jobs=None)
        begin_time_train_3=time.time()
        knn3.fit(X_train, Y_train)
        finish_time_train_3 = time.time()
        pickle.dump(knn3, open(filename3, 'wb'))
        begin_time_test_3 = time.time()
        y_prediction_3 = knn3.predict(X_test)
        accuracy3 = np.mean(y_prediction_3 == Y_test) * 100
        finish_time_test_3 = time.time()
        time_training_3 = finish_time_train_3 - begin_time_train_3
        print(f"Runtime of the train KNN  with K=51  model is {time_training_3}")
    time_test_3 = finish_time_test_3 - begin_time_test_3
    print(f"Runtime of the test KNN  with K=51 model is {time_test_3} ")
    print('Model KNN  with K=51 Test Mean Square Error : ', metrics.mean_squared_error(Y_test, y_prediction_3))

    return accuracy1,accuracy2,accuracy3