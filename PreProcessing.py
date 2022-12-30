from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import dateutil.parser
import re
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
from Model_1 import train_model_1
from Model_2 import train_model_2


def do_preprocessing(x_features, y):

    # Categorical encoding for video_id(0) & channel_title(3) & comments_disabled(9)
    # & ratings_disabled(10) & video_error_or_removed(10)
    videoId_encoder = LabelEncoder()
    x_features.iloc[:, 0] = videoId_encoder.fit_transform(x_features.iloc[:, 0])
    channelTitle_encoder = LabelEncoder()
    x_features.iloc[:, 3] = channelTitle_encoder.fit_transform(x_features.iloc[:, 3])
    comments_disabled_encoder = LabelEncoder()
    x_features['comments_disabled'] = comments_disabled_encoder.fit_transform(x_features['comments_disabled'])
    ratings_disabled_encoder = LabelEncoder()
    x_features['ratings_disabled'] = ratings_disabled_encoder.fit_transform(x_features['ratings_disabled'])
    video_error_or_removed_encoder = LabelEncoder()
    x_features['video_error_or_removed'] = video_error_or_removed_encoder.fit_transform(x_features['video_error_or_removed'])

    # Reformat trending_date(1) & publish_time(5) & create new column in features called days_to_be_trend
    x_features['trending_date'] = x_features['trending_date'].apply(lambda x: reformat_date(x))
    x_features['publish_time'] = x_features['publish_time'].apply(lambda x: from_timezone_to_date(x))
    days_to_be_trend = x_features['trending_date'] - x_features['publish_time']
    x_features['days_to_be_trend'] = days_to_be_trend.apply(lambda x: str(x).split(' ', 1)[0])

    # Bag of words preprocessing and filter for title(2) & tags(6) & video_description(12)
    x_features['title'] = x_features['title'].apply(lambda x: filter_text(x))
    title_vector = TfidfVectorizer(min_df=1500,  analyzer='word', stop_words='english', ngram_range=(1, 1))
    title_tfidf = title_vector.fit_transform(x_features['title'])
    x_features['tags'] = x_features['tags'].apply(lambda x: filter_text(x))
    tags_vector = TfidfVectorizer(min_df=2000,  analyzer='word', stop_words='english', ngram_range=(1, 1))
    tags_tfidf = tags_vector.fit_transform(x_features['tags'])
    x_features['video_description'] = x_features['video_description'].apply(lambda x: filter_text(x))
    video_description_vector = TfidfVectorizer(min_df=5000, analyzer='word', stop_words='english', ngram_range=(1, 1))
    video_description_tfidf = video_description_vector.fit_transform(x_features['video_description'])

    # Drop Rows if null cells are found
    x_features = x_features.dropna(axis=0)

    # concatenate title & tags & video_description and drop trending_date & publish_time
    # & title & tags & video_description
    x_features.drop('trending_date', axis=1, inplace=True)
    x_features.drop('publish_time', axis=1, inplace=True)
    x_features.drop('title', axis=1, inplace=True)
    x_features.drop('video_description', axis=1, inplace=True)
    x_features.drop('tags', axis=1, inplace=True)
    '''
    df1 = pd.DataFrame(video_description_tfidf.toarray(), columns=video_description_vector.get_feature_names())
    x_features = pd.concat([x_features, df1], axis=1)
    df2 = pd.DataFrame(tags_tfidf.toarray(), columns=tags_vector.get_feature_names())
    x_features = pd.concat([x_features, df2], axis=1)
    df3 = pd.DataFrame(title_tfidf.toarray(), columns=title_vector.get_feature_names())
    x_features = pd.concat([x_features, df3], axis=1)
    '''

    # Normalize dataset
    x_features['video_id'] = (x_features['video_id'] - x_features['video_id'].min()) / (x_features['video_id'].max() - x_features['video_id'].min())
    x_features['channel_title'] = (x_features['channel_title'] - x_features['channel_title'].min()) / (x_features['channel_title'].max() - x_features['channel_title'].min())
    x_features['category_id'] = (x_features['category_id'] - x_features['category_id'].min()) / (x_features['category_id'].max() - x_features['category_id'].min())
    x_features['views'] = (x_features['views'] - x_features['views'].min()) / (x_features['views'].max() - x_features['views'].min())
    x_features['comment_count'] = (x_features['comment_count'] - x_features['comment_count'].min()) / (x_features['comment_count'].max() - x_features['comment_count'].min())
    x_features['days_to_be_trend'] = x_features['days_to_be_trend'].astype(float)
    x_features['days_to_be_trend'] = (x_features['days_to_be_trend'] - x_features['days_to_be_trend'].min()) / (x_features['days_to_be_trend'].max() - x_features['days_to_be_trend'].min())
    y = (y - y.min()) / (y.max() - y.min())

    # Correlation
    all_data = pd.concat([x_features, y], axis=1)
    dataplot = sb.heatmap(all_data.corr(), cmap="YlGnBu", annot=True)
    mp.show()

    # Split the data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x_features, y, test_size=0.30, shuffle=True)
    while True:
        val = input("Enter 1 to polynomial_regression ,2 to multi_linear_regression  OR -1 to exit :  ")
        val = int(val)
        print(val)
        if val == -1:
            print("exit from program")
            break
        elif val == 1:
            # train model 1
            train_model_1(X_train, y_train, X_test, y_test)
        elif val == 2:
            # train model 2
            train_model_2(X_train, y_train, X_test, y_test)
        else:
            print("invalid choice , please enter valid choice:")



def reformat_date(date):
    date = str(date).split('.')
    new_date = '20' + date[0] + '-' + date[2] + '-' + date[1]
    return pd.to_datetime(new_date)


def filter_text(text):

    x = text
    # convert to lowercase
    x = str(x).lower()
    # remove Special Characters
    x = re.sub(r'\W', ' ', str(x))
    # remove Single Characters
    x = re.sub(r'\s+[a-zA-Z]\s+', ' ', str(x))
    # remove Single Characters from the start
    x = re.sub(r'\^[a-zA-Z]\s+', ' ', str(x))
    # Replace multiple spaces with single space
    x = re.sub(r'\s+', ' ', x, flags=re.I)
    # Removing prefixed 'b'
    x = re.sub(r'^b\s+', '', x)
    # Removing links
    x = re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)

    return x


def from_timezone_to_date(date):
    to_date = dateutil.parser.parse(date)
    new_date = to_date.strftime('%Y-%m-%d')
    return pd.to_datetime(new_date)
