import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import math
import pickle

DATA_BASE_PATH = '/Users/john/CS411/project/sentiment_analysis/labelled_sentences/'
DATA_FILE_LIST = ['amazon_cells_labelled.txt', 'yelp_labelled.txt', 'imdb_labelled.txt', 'fin_headlines.txt']

def get_data(base_path, file_list):
    df = pd.DataFrame()

    for f in file_list:
        curr_data = pd.read_csv(str(base_path + f), sep='\t', header=None, names=['sentence', 'rating'], quoting=3)
        df = df.append(curr_data)

    return df.values

"""Return counts of each word, term-document matrix, labels for each word"""
def get_bag(corpus, max_df=1.0, min_df=1, stop_words=None):
    vectorizer = CountVectorizer(lowercase=True, max_df=max_df, min_df=min_df, stop_words=stop_words)
    TD_matrix = vectorizer.fit_transform(corpus)
    TD_matrix = TD_matrix.toarray()

    #print(vectorizer.get_stop_words())

    counts = np.sort(np.sum(TD_matrix, axis=0))
    #print(TD_matrix.shape)

    with open('BOW_model.pkl', 'wb') as f:
        pickle_model = pickle.dump(vectorizer, f)

    return counts[::-1], TD_matrix, vectorizer.get_feature_names()

"""Return shuffled data separated into train_data, train_labels, val_data, val_labels"""
def separate_data(data, labels, train_pct=0.8):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    data = data[index]
    labels = labels[index]

    split = math.floor(train_pct * data.shape[0])
    return data[:split], labels[:split].astype(int), data[split:], labels[split:].astype(int)

"""Return trained logistic regression classifier"""
def get_classifier(data, targets):
    classifier = LogisticRegression(fit_intercept=0.7)
    classifier.fit(data, targets)

    return classifier

"""Load and train log model on vectorized bag of words samples. Save model"""
if __name__ == "__main__":
    data = get_data(DATA_BASE_PATH, DATA_FILE_LIST)
    corpus = data[:, 0]

    
    prepositions = ['above', 'across', 'after', 'at', 'around', 'before', 'behind',
                    'below', 'beside', 'between', 'by', 'down', 'during', 'for', 'from', 
                    'in', 'inside', 'onto', 'of', 'off', 'on', 'out', 'through', 'to', 
                    'under', 'up', 'with', 'Apple', 'AMD', 'Amazon', 'Intel', 'Microsoft',
                    'Siri', 'Exxon', 'Nvidia', 'Tesla', 'Square', 'Snapchat', 'Budweiser', 'Facebook',
                    'Twitter', 'Google', 'Alibaba', 'AAPL', 'AMD', 'AMZN', 'INTC', 'MSFT', 'SIRI', 'XOM', 'NVDA',
                    'BUD', 'TSLA', 'FB', 'TWTR', 'GOOG', 'BABA', 'SQ', 'SNAP']
    max_df = 0.15
    min_df = 3
    
    counts, TD_matrix, feature_name = get_bag(corpus, max_df=max_df, min_df=min_df, stop_words=prepositions)

    train_data, train_labels, val_data, val_labels = separate_data(TD_matrix, data[:, 1])
    
    classifier = get_classifier(train_data, train_labels)

    train_acc = classifier.score(train_data, train_labels)
    test_acc = classifier.score(val_data, val_labels)

    print("Train Acc:", train_acc)
    print("Test Acc:", test_acc)

    with open('log_model.pkl', 'wb') as f:
        pickle_model = pickle.dump(classifier, f)
    