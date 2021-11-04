#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Template script for exercise 3."""
import re
import sys

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


def load_dataset(filepath='./corpus.txt', length_label=10):
    """Load the dataset of reviews.

    The dataset contains two columns---label and text.
    """
    with open(filepath, 'r') as f:
        data = f.readlines()

        texts, labels = [], []
        for line in data:
            texts.append(line[length_label + 1:])
            labels.append(line[:length_label])

        return texts, labels


def preprocess_dataset(texts, remove_non_alphanumeric=True, to_lower_case=True,
                       stem=True, remove_stop_words=True,
                       stopwords_filepath='./stopwords_english.txt'):
    """Preprocesses the data in 'text'.

    Note that the stemming algorithm automatically converts all characters to
    lowercase. Hence, 'to_lower_case' has no effect if 'stem' is set to True.
    """
    if remove_non_alphanumeric:
        # Replace all none alphanumeric characters with spaces.
        texts = [re.sub(r'[^a-zA-Z0-9\s]', ' ', review) for review in texts]

    if to_lower_case:
        # Split review into single words and convert to lowercase.
        texts = [review.lower().split() for review in texts]
    else:
        # Split review into single words.
        texts = [review.split() for review in texts]

    if remove_stop_words:
        # Here we remove the stop words, that is 'the', 'a', 'I', 'is', etc.
        # Stop words are those words which occur extremely frequently in any
        # text (and hence do not give us any information).
        with open(stopwords_filepath, 'r') as f:
            sw = f.read().split()
        texts = [[word for word in review if word not in sw]
                 for review in texts]

    if stem:
        # Here we perform a stemming algorithm (Porter Stammer). The words like
        # 'go', 'goes', 'going' indicate the same activity. We can replace all
        # these words by a single word ‘go’.
        stemmer = PorterStemmer()
        texts = [[stemmer.stem(word) for word in review] for review in texts]

    return texts


def split_train_test(texts, labels):
    """Splits the data in 'text' to randomly chosen disjoint sets for training
    and testing."""
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test, idc_train, idc_test = train_test_split(
        texts, labels, range(len(labels)), random_state=0xDEADBEEF)

    return x_train, x_test, y_train, y_test, idc_test


def count_vectorizer(texts, k=1):
    """Returns all unique features of 'text' and their frequency. Frequencies
    are sorted in descending order. Features follow the sorting order of
    frequencies."""
    y = []
    c = []
    # TODO begin
    entries = 0
    vals = {}
    for text in texts:
        counter, len_ = 0, len(text)
        while counter <= (len_-k):  # info to avoid running index over array bound
            word = " ".join(text[counter:(counter+k)]
                            ) if k != 1 else text[counter]
            try:
                vals[word] += 1
            except KeyError:
                vals[word] = 1
            counter += 1
        entries += counter
    y, c = list(vals.keys()), list(vals.values())
    # check if already sorted
    arrlen_ = len(y)
    if arrlen_ == entries:  # info already sorted fast return
        return y, c

    def merge_sort(words, weights, start, end):
        if start < end:
            mid = int((start+end)/2)
            merge_sort(words, weights, start, mid)
            merge_sort(words, weights, mid+1, end)
            return merge_(words, weights, start, mid, end)
        else:
            return words, weights

    def merge_(words, weights, start, mid, end):
        left_arr, right_arr = [], []
        for left in range(start, mid+1):  # range-stop is exclusive in python
            left_arr.append((words[left], weights[left]))  # append ist O(1)
        for right in range(mid+1, end+1):
            right_arr.append((words[right], weights[right]))
        left_arr.append(("", 0))  # 0 is per design not possible either a word
        right_arr.append(("", 0))  # is in the list (count = 1) or not
        left, right = 0, 0
        for counter in range(start, end+1):
            if left_arr[left][1] >= right_arr[right][1]:  # order
                words[counter], weights[counter] = left_arr[left][0], left_arr[left][1]
                left += 1
            else:
                words[counter], weights[counter] = right_arr[right][0], right_arr[right][1]
                right += 1
        return words, weights
    y, c = merge_sort(y, c, 0, arrlen_-1)  # call functions
    # TODO end
    return y, c


def transform_features_to_token_counts(features, total_counts, texts,
                                       ngram_range=[1, 1]):
    """Returns token counts for each feature and each text in 'texts'.

    Token counts are computed for each text in 'texts'. This results in a
    "number of texts" x "number of features" matrix of token counts.
    """
    features = np.asarray(features)
    if not any(isinstance(x, list) for x in texts):
        texts = [texts]

    token_counts = np.zeros((len(texts), len(total_counts)), dtype=np.int)
    for i, text in enumerate(texts):
        for k in range(ngram_range[0], ngram_range[1] + 1):
            tokens, counts = count_vectorizer([text], k)
            for token, count in zip(tokens, counts):
                idx = np.where(features == token)
                token_counts[i, idx] += count

    return token_counts


def count_vectors_as_features(texts_train, texts_test,
                              ngram_range=[1, 1]):
    """Computes the features and returns a matrix of token counts for the train
    and test set."""
    features = []
    counts = []
    for k in range(ngram_range[0], ngram_range[1] + 1):
        # Here we extract the feature names, i.e., all unique text fragments of
        # the whole training set.
        ret = count_vectorizer(texts_train, k)
        features += ret[0]
        counts += ret[1]

    # Here we convert train and test data to a matrix of token counts.
    texts_train_count = transform_features_to_token_counts(features, counts,
                                                           texts_train,
                                                           ngram_range)
    texts_test_count = transform_features_to_token_counts(features, counts,
                                                          texts_test,
                                                          ngram_range)

    return texts_train_count, texts_test_count


def train_model(classifier, feature_vector_train, labels_train):
    """Trains the classifier on the feature vectors."""
    classifier.fit(feature_vector_train, labels_train)

    return classifier


def test_model(classifier, feature_vector_test):
    """Returns the class labels predicted by the classifier."""
    predictions = classifier.predict(feature_vector_test)

    return predictions


def main_test():
    """Tests your algorithm with a simple example."""
    k = 1
    texts = [
        ["stuning", "even", "for", "the", "non", "gamer", "this", "sound",
         "track", "was", "beautiful"],
        ["cannot", "recommend", "as", "a", "former", "alaskan", "i", "did",
         "not", "want", "to", "have", "to", "do", "this"]
    ]

    y, c = count_vectorizer(texts, k=k)

    print("X: {0}".format(texts))
    print("y: {0}".format(y))
    print("c: {0}\n".format(c))


def main():
    """Main function for exercise 3."""
    ngram_range = [1, 1]  # Range of ks. Can be [1,2], [1,3], [2, 4] etc.
    num_examples = 500  # There are 10000 reviews in the dataset.
    # If you want to test the classification algorithm set to False.
    test_algorithm = True
    np.random.seed(0xDEADBEEF)

    if test_algorithm:
        # Test your implementation.
        main_test()
        sys.exit(0)

    # Load the dataset of reviews.
    raw_texts, raw_labels = load_dataset()
    num_reviews = len(raw_labels)

    # Choose a random subset of the dataset.
    idc = np.random.choice(num_reviews, size=num_examples, replace=False)
    raw_texts = [raw_texts[idx] for idx in idc]
    raw_labels = [raw_labels[idx] for idx in idc]

    # Perform pre-processing of the texts.
    texts = preprocess_dataset(raw_texts, remove_non_alphanumeric=True,
                               to_lower_case=True, stem=True,
                               remove_stop_words=True)

    # Split the dataset into training and test set. Here we also convert the
    # labels to numeric values.
    texts_train, texts_test, labels_train, labels_test, idc_test = \
        split_train_test(texts, raw_labels)

    # Create the feature vectors.
    texts_train_count, texts_test_count = count_vectors_as_features(
        texts_train, texts_test, ngram_range=ngram_range)

    # Train the classifier.
    classifier = train_model(MultinomialNB(), texts_train_count, labels_train)

    # Predict the labels on the test data.
    predictions = test_model(classifier, texts_test_count)

    # Print the test accuracy.
    # NOTE: This is the classification accuracy and it can vary. If it is not 100%, which it is most probably,
    # it is most likely not an error in your algorithm. To test your algorithm use `test_algorithm = True` and
    # check for correct output of y and c.
    accuracy = accuracy_score(predictions, labels_test)
    print("\n*** Test accuracy {0:.2%} ***\n".format(accuracy))

    # Print some example reviews along with the true label and the predicted
    # label.
    idc = np.random.choice(predictions.size, size=5, replace=False)
    for idx in idc:
        print("Review #{0} of test set:".format(idc_test[idx]))
        print(raw_texts[idc_test[idx]].rstrip())
        print("True label: {0}".format(labels_test[idx]))
        print("Predicted label: {0}\n".format(predictions[idx]))


if __name__ == '__main__':
    main()
