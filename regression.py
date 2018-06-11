import json

import numpy as np
import pandas as pd

from six.moves import cPickle as pickle
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from paragraph2vec import Paragraph2Vec
from sif import SIF


def get_category_id_dict() -> dict:
    cat_id_to_idx = {}
    with open('./data/US_category_id.json', 'r') as f:
        data = json.load(f)
        idx = 0
        for category in data['items']:
            cat_id_to_idx[int(category['id'])] = idx
            idx += 1
    return cat_id_to_idx


def add_categorical_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    category_id_to_idx = get_category_id_dict()
    unique_channel_titles = data_frame['channel_title'].unique().tolist()
    channel_title_to_idx = dict(zip(unique_channel_titles,
                                    range(len(unique_channel_titles))))
    data_frame['category_id_idx'] = data_frame['category_id'].map(lambda x: category_id_to_idx[x])
    data_frame['channel_title_idx'] = data_frame['channel_title'].map(lambda x: channel_title_to_idx[x])
    return data_frame


def get_categorical_data(data_frame: pd.DataFrame, use_one_hot: bool = True, scale_data: bool = True):
    if use_one_hot:
        class_num = len(get_category_id_dict())
        x_vals = list(data_frame['category_id_idx'])
        x_data = np.zeros((len(x_vals), class_num), dtype=np.int32)
        ident = np.identity(class_num)
        for key, value in enumerate(x_vals):
            x_data[key] = ident[value - 1]
        if scale_data:
            x_data = StandardScaler().fit(x_data).transform(x_data)
    else:
        x_data = np.array(data_frame['category_id_idx'].tolist()).reshape(-1, 1)
    return x_data


def get_tfidf_data(data_frame: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
    return vectorizer.fit_transform(data_frame['text'].tolist())


def get_par2vec_data(_):
    with open('./data_preprocessed/paragraph_matrix.pkl', 'rb') as f:
        return pickle.load(f)


def get_sif_data(data_frame: pd.DataFrame):
    paragraph_list = data_frame['text'].tolist()
    word_to_id = Paragraph2Vec().build_vocab(paragraph_list).get_word2id_dict()
    paragraph_list = list(map(lambda x: x.lower().split(' '), paragraph_list))
    with open('./data_preprocessed/word_matrix.pkl', 'rb') as f:
        word_matrix = pickle.load(f)
    word_to_vector = {word: word_matrix[idx] for word, idx in word_to_id.items()}
    del word_to_id, word_matrix

    sif = SIF(word_to_vector, paragraph_list)
    return sif.get_paragraph_matrix()


def get_labels(data_frame: pd.DataFrame, column_to_predict: str, use_log: bool = True):
    y_data = data_frame[column_to_predict].tolist()
    if use_log:
        y_data = list(map(lambda val: np.log(val + 1), y_data))
    return y_data


def split_data(x_data, y_data):
    train_part = int(len(y_data) * 0.9)
    x_train = x_data[:train_part]
    x_test = x_data[train_part:]
    y_train = y_data[:train_part]
    y_test = y_data[train_part:]
    return x_train, x_test, y_train, y_test


def mlp_regression(x_train, y_train, x_test):
    mlp = MLPRegressor(solver='adam',
                       # hidden_layer_sizes=(64, 128, 16),
                       # max_iter=10000,
                       activation='relu',
                       learning_rate='adaptive')
    mlp.fit(x_train, y_train)
    return mlp.predict(x_test)


def print_metrics(y_test, y_pred):
    print("Mean squared error: %.2f" % metrics.mean_squared_error(y_test, y_pred))
    print("Explained variance score: %.2f" % metrics.explained_variance_score(y_test, y_pred))
    print("Mean absolute error: %.2f" % metrics.mean_absolute_error(y_test, y_pred))
    print("Mean squared log error: %.2f" % metrics.mean_squared_log_error(y_test, y_pred))
    print("Median absolute error: %.2f" % metrics.median_absolute_error(y_test, y_pred))
    print('R2 score: %.2f' % metrics.r2_score(y_test, y_pred))


def apply_regression(inputs_type: str, data_frame: pd.DataFrame, column_to_predict: str):
    inputs_type_callback = {
        'categorical': get_categorical_data,
        'tfidf': get_tfidf_data,
        'par2vec': get_par2vec_data,
        'sif': get_sif_data,
    }
    assert inputs_type in inputs_type_callback

    y = get_labels(data_frame, column_to_predict)
    x = inputs_type_callback[inputs_type](data_frame)
    x_train, x_test, y_train, y_test = split_data(x, y)
    y_pred = mlp_regression(x_train, y_train, x_test)

    print("Predicting '%s' column (inputs - '%s'):" % (column_to_predict, inputs_type))
    print_metrics(y_test, y_pred)
    print('-------')
    print()
    return
