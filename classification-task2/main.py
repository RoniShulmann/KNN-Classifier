from collections import Counter
# from sklearn import preprocessing
import time
import json
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

from typing import List


def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    print(
        f'starting classification with {data_trn}, {data_vld}, and {data_tst}')

    # todo: return a list of your predictions for test instances
    predictions = KNN(k=3, data_trn=data_trn,
                      data_vld=data_vld, data_tst=data_tst)
    return predictions


def KNN(k, data_trn, data_vld, data_tst):
    data_trn_df = pd.read_csv(data_trn)
    data_vld_df = pd.read_csv(data_vld)
    data_tst_df = pd.read_csv(data_tst)

    # Get the feature and label arrays
    X_trn, y_trn = data_trn_df.values[:, :-1], data_trn_df['class'].values
    X_vld, y_vld = data_vld_df.values[:, :-1], data_vld_df['class'].values
    X_tst, y_tst = data_tst_df.values[:, :-1], data_tst_df['class'].values

    # Initialize an empty list to store the predictions
    predictions = []

    # Iterate over the test instances
    for i in range(len(X_tst)):
        # Find the euclidean distance between the test point and all training points
        distances = np.sqrt(
            np.sum((X_trn.astype('float64') - X_tst[i][np.newaxis, :].astype('float64'))**2, axis=1))

        # Get the indices of the k nearest neighbors
        k_nearest_indices = np.argsort(distances)[: k]

        # Get the labels of the k nearest neighbors
        k_nearest_labels = y_trn[k_nearest_indices]

        # Get the most common label among the k nearest neighbors
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]

        # Append the prediction to the list
        predictions.append(most_common_label)

    return predictions


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    # make sure you predict label for all test instances
    assert (len(labels) == len(predicted))
    print(
        f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
