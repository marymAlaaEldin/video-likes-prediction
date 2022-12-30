import pandas as pa
from sklearn.model_selection import train_test_split
from PreProcessing import do_preprocessing
from Model_1 import train_model_1
from Model_2 import train_model_2


'''
    Reading data
'''

data = pa.read_csv('VideoLikesDataset.csv')
X_features = data.iloc[:, 0:13]
Y = data.iloc[:, 13]

'''
    Preprocessing
'''

do_preprocessing(X_features, Y)








