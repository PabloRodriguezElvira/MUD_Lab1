import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from utils import *
from classifiers import *
from preprocess import preprocess
from collections import Counter

seed = 42
random.seed(seed)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input data in csv format", type=str)
    parser.add_argument("-v", "--voc_size", help="Vocabulary size", type=int)
    parser.add_argument("-a", "--analyzer",
                        help="Tokenization level: {word, char}",
                        type=str, choices=['word','char'])
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    raw = pd.read_csv(args.input)

    languages = set(raw['language'])
    print('========')
    print('Languages', languages)
    print('========')

    X = raw['Text']
    y = raw['language']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    print('========')
    print('Split sizes:')
    print('Train:', len(X_train))
    print('Test:', len(X_test))
    print('========')

    # Preprocess
    if args.analyzer == 'word':
        X_train, y_train = preprocess(X_train, y_train)
        X_test,  y_test  = preprocess(X_test,  y_test)

    X_test_text = list(X_test)
    y_test_list = list(y_test)

    # Compute text features
    features, X_train_raw, X_test_raw = compute_features(
        X_train, X_test,
        analyzer=args.analyzer,
        max_features=args.voc_size
    )

    print('========')
    print('Number of tokens in the vocabulary:', len(features))

    print('Coverage: ', compute_coverage(features, X_test, analyzer=args.analyzer))
    print('========')

    # Apply classifier
    X_train_vec, X_test_vec = normalizeData(X_train_raw, X_test_raw)
    #y_predict = applyNaiveBayes(X_train_vec, y_train, X_test_vec) # Naive Bayes classifier
    y_predict = applySVM(X_train_vec, y_train, X_test_vec) # SVM classifier


    print('========')
    print('Prediction Results:')
    plot_F_Scores(y_test, y_predict)
    print('========')

    plot_Confusion_Matrix(y_test, y_predict, "Greens")

    # Plot PCA
    print('========')
    print('PCA and Explained Variance:')
    plotPCA(X_train_vec, X_test_vec, y_test, languages)
    print('========')

    # Error report
    errors = []
    for txt, y_true, y_hat in zip(X_test_text, y_test_list, y_predict):
        if y_true != y_hat:
            errors.append((y_true, y_hat, txt))

    print("Num errores:", len(errors), "de", len(y_test_list))

    pairs = Counter((yt, yh) for yt, yh, _ in errors)
    print("Top confusiones:")
    for (yt, yh), c in pairs.most_common(20):
        print(f"{yt} -> {yh}: {c}")

    # Examples of a specific confusion
    target_true = "Latin"
    target_pred = "English"
    examples = [txt for yt, yh, txt in errors if yt == target_true and yh == target_pred][:10]
    print(f"\nEjemplos {target_true}->{target_pred}:")
    for e in examples:
        print("-", e[:200])