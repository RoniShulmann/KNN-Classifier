# K-Nearest Neighbors Classifier

This is a Python implementation of a K-Nearest Neighbors (KNN) classifier for a given dataset. 

## Dependencies

* Python 3.6+
* pandas
* numpy
* scikit-learn

## Usage

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Prepare your dataset in the required format, with separate files for the training, validation, and test sets.
3. Create a `config.json` file with the following keys and values:
   * `"data_file_train"`: the file path of the training set.
   * `"data_file_validation"`: the file path of the validation set.
   * `"data_file_test"`: the file path of the test set.
4. Run the classifier using `python knn_classifier.py`.
5. The classifier will output the classification accuracy on the test set, as well as the total time taken to classify the test instances.

## Feature Scaling
In order to ensure that all the features are treated equally, 
it is important to scale them.
Here, we use the StandardScaler from the sklearn.preprocessing module to scale the features.

```python
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the StandardScaler on the training data
scaler.fit(X_trn)

# Standardize the training, validation, and test data
X_trn = scaler.transform(X_trn)
X_vld = scaler.transform(X_vld)
X_tst = scaler.transform(X_tst)
```

## K-Nearest Neighbors Algorithm
The K-Nearest Neighbors algorithm is a simple, yet effective machine learning algorithm 
used for classification and regression problems. 

```python
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
```


This project was created by [Ori Nurieli](https://github.com/orinurieli) and [Roni Shulman](https://github.com/RoniShulmann). 
