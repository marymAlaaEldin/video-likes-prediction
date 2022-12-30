import math

import pandas as pa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt
from DecisionTree import DecisionTree_model
from SVM_Model import Svm_model
from KNN_Model import knn_model
from Logistics_Regression import logistic_regression
from GaussianNP import  gaussian_nb
from Random_Forest import random_forest
import dateutil.parser
import re
'''
    Reading data
'''

data = pa.read_csv('VideoLikesTestingClassification.csv')
x_features = data.iloc[:, 0:13]

'''
    Preprocessing
'''
def reformat_date(date):
    date = str(date).split('.')
    new_date = '20' + date[0] + '-' + date[2] + '-' + date[1]
    return pa.to_datetime(new_date)

def from_timezone_to_date(date):
    to_date = dateutil.parser.parse(date)
    new_date = to_date.strftime('%Y-%m-%d')
    return pa.to_datetime(new_date)
# Categorical encoding
videoId_encoder = LabelEncoder()
x_features.iloc[:, 0] = videoId_encoder.fit_transform(x_features.iloc[:, 0])
channelTitle_encoder = LabelEncoder()
x_features.iloc[:, 3] = channelTitle_encoder.fit_transform(x_features.iloc[:, 3])
comments_disabled_encoder = LabelEncoder()
x_features['comments_disabled'] = comments_disabled_encoder.fit_transform(x_features['comments_disabled'])
ratings_disabled_encoder = LabelEncoder()
x_features['ratings_disabled'] = ratings_disabled_encoder.fit_transform(x_features['ratings_disabled'])
video_error_or_removed_encoder = LabelEncoder()
x_features['video_error_or_removed'] = video_error_or_removed_encoder.fit_transform( x_features['video_error_or_removed'])
VideoPopularity = LabelEncoder()
data['VideoPopularity'] = VideoPopularity.fit_transform(data['VideoPopularity'])

# Reformat trending_date(1) & publish_time(5) & create new column in features called days_to_be_trend
x_features['trending_date'] = x_features['trending_date'].apply(lambda x: reformat_date(x))
x_features['publish_time'] = x_features['publish_time'].apply(lambda x: from_timezone_to_date(x))
days_to_be_trend = x_features['trending_date'] - x_features['publish_time']
x_features['days_to_be_trend'] = days_to_be_trend.apply(lambda x: str(x).split(' ', 1)[0])

#drop trending_date trending_date colnums replace them by days_to_be_trend colnumm
x_features.drop('trending_date', axis=1, inplace=True)
x_features.drop('publish_time', axis=1, inplace=True)

## fill null values previous index of row
data.fillna( method='ffill', axis=0, inplace=True, limit=None, downcast=None)

##correlation
all_data = pa.concat([x_features, data['VideoPopularity']], axis=1)
dataplot = sb.heatmap(all_data.corr(), cmap="YlGnBu", annot=True)
plt.show()
#selection features
select=pa.concat([x_features['category_id'],x_features['views']], axis=1)
select1=pa.concat([select,x_features['comment_count']], axis=1)
select2=pa.concat([select1,x_features['video_id']], axis=1)
select3=pa.concat([select2,x_features['channel_title']], axis=1)
select4=pa.concat([select3,x_features['days_to_be_trend']], axis=1)
select5=pa.concat([select4,x_features['comments_disabled']], axis=1)
select6=pa.concat([select5,x_features['ratings_disabled']], axis=1)
select7=pa.concat([select6,x_features['video_error_or_removed']], axis=1)
Y=data['VideoPopularity']

X_train, X_test, y_train, y_test = train_test_split(select7, Y, test_size=0.2, shuffle=True, random_state=10)
while True:
        val = input("Enter 1 to DecisionTree ,2 to Svm  ,3 to KNN ,4 to LogisticsRegression  ,5 to GaussianNB ,6 to  Random Forest OR -1 to exit :  ")
        val = int(val)
        print(val)
        if val==-1:
            print("exit from program")
            break
        elif val==1:
            decision_acc, Adaboost_acc = DecisionTree_model(X_train, X_test, y_train, y_test)
            print("The achieved accuracy using Decision Tree is " + str(decision_acc))
            print("The achieved accuracy using Adaboost is " + str(Adaboost_acc))
        elif val==2:
            acc_svm1,acc_svm2,acc_svm3,acc_svm4,acc_svm5= Svm_model(X_train, X_test, y_train, y_test)
            print('LinearSVC OneVsOne SVM accuracy: ' + str(acc_svm1))
            print('SVC with RBF kernel accuracy: ' + str(acc_svm2))
            print('SVC with polynomial kernel  degree 2 accuracy: ' + str(acc_svm3))
            print('SVC with polynomial kernel  degree 3 accuracy: ' + str(acc_svm4))
            print('SVC with polynomial kernel  degree 4 accuracy: ' + str(acc_svm5))

        elif val==3:
            acc_knn1,acc_knn2,acc_knn3=knn_model(X_train, X_test, y_train, y_test)
            print("The achieved accuracy using KNN with K=17 is " + str(acc_knn1))
            print("The achieved accuracy using KNN with K=3 is " + str(acc_knn2))
            print("The achieved accuracy using KNN with K=51 is " + str(acc_knn3))
        elif val==4:
            acc_logistics=logistic_regression(X_train, X_test, y_train, y_test)
            print("The achieved accuracy using logistic_regression " + str(acc_logistics))
        elif val==5:
             acc_gaussian= gaussian_nb(X_train, X_test, y_train, y_test)
             print("The achieved accuracy using gaussian np " + str(acc_gaussian))
        elif val==6:
            acc_forest=random_forest(X_train, X_test, y_train, y_test)
            print("The achieved accuracy using random forest " + str(acc_forest))

        else:
            print("invalid choice , please enter valid choice:")
##graph bar tranning time
tranning_time = {'Tree':0.18489, 'AdaBoost': 14.979
    , 'Linear_SVM': 6.808722,'RBF_SVM': 683.62,'Poly2_SVM': 82.84667,'Poly3_SVM': 65.6942,'Poly4_SVM': 202.9398
    ,'KNN 17': 0.184909,'KNN 3': 0.2060890,'KNN 51': 0.201117,
     'Logistic': 6.79898,
     'Gaussian': 0.062498,
     'Random_f': 5.6451,
                 }
models_train_time = list(tranning_time.keys())
train_time = list(tranning_time.values())

fig1 = plt.figure(figsize=(150, 150))

# creating the bar plot
plt.ylim(0.00001,685)
plt.bar(models_train_time, train_time, color='darkblue',
        width=0.4)

plt.xlabel("Classficition Models")
plt.ylabel("Tranning Time for Models")
plt.title("Tranning Time for Classficition Models")
plt.show()

##graph bar testtime
test_time = {'Tree':0.1562380, 'AdaBoost': 0.1609210
    , 'Linear_SVM':  0.0845916271,'RBF_SVM': 0.0689949989,'Poly2_SVM': 8.800754308700,'Poly3_SVM':9.8697569370,'Poly4_SVM': 10.4740283489
    ,'KNN 17': 0.5012683,'KNN 3': 0.4528948,'KNN 51': 0.57028079,
     'Logistic': 0.02712488,
     'Gaussian': 0.00651001,
     'Random_f': 0.13151431,
                 }
models_test_time = list(test_time.keys())
test = list(test_time.values())

fig2 = plt.figure(figsize=(150, 150))

# creating the bar plot
plt.ylim(0.000001,12)
plt.bar(models_test_time, test, color='purple',
        width=0.4)

plt.xlabel("Classficition Models")
plt.ylabel("Test Time for Models")
plt.title("Test Time for Classficition Models")
plt.show()

##graph bar accuracy
data_acc = {'Tree':97.2, 'AdaBoost': 97.2
    , 'Linear_SVM':  76.6103,'RBF_SVM':  76.6103,'Poly2_SVM': 68.4530095,'Poly3_SVM': 59.728088,'Poly4_SVM': 57.510559662
    ,'KNN 17': 81.9,'KNN 3': 80.1,'KNN 51': 81.4,
     'Logistic': 78.2,
     'Gaussian': 74.9,
     'Random_f': 96.1,
                 }
models_acc = list(data_acc.keys())
accuraices = list(data_acc.values())

fig3 = plt.figure(figsize=(150, 150))

# creating the bar plot
plt.ylim(10,100)
plt.bar(models_acc, accuraices, color='maroon',
        width=0.4)

plt.xlabel("Classficition Models")
plt.ylabel("Accuracies for Models")
plt.title("Accuracies for Classficition Models")
plt.show()





