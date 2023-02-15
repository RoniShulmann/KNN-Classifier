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

## License

This project is licensed under the MIT License. See `LICENSE` for more information.
