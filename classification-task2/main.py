import time
import json
import pandas as pd

from sklearn.metrics import accuracy_score
from typing import List


def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    # todo: implement this function
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')


    predictions = list()  # todo: return a list of your predictions for test instances
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

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
