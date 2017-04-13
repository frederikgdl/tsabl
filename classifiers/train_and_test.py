import logging
import numpy as np
from time import time
from os import path

from classifiers.models.svm import SVM
from classifiers.models.log_res import LogRes

import classifiers.config as config
import classifiers.funcs as funcs
from classifiers.word_embedding_dict import WordEmbeddingDict
from classifiers.models.afinn import AfinnModel
from classifiers.models.combo_average import ComboAverage
from classifiers.models.random_uniform import RandomUniform
from classifiers.models.random_weighted import RandomWeighted
from classifiers.models.textblob import Textblob
from classifiers.models.vader import Vader

embedding_file = config.EMBEDDING_FILE
train_file = config.TRAIN_FILE
test_file = config.TEST_FILE

classifiers = [
    SVM(),
    LogRes()
]

baselines = [
    RandomUniform(),
    RandomWeighted(),
    AfinnModel(),
    Vader(threshold=0.1),
    Textblob(subjectivity_threshold=0.1, polarity_threshold=0.5),
    ComboAverage()
]


def load_word_embeddings():
    logging.info("Loading word embeddings")
    t = time()
    md = WordEmbeddingDict(embedding_file)
    logging.debug("Done. " + str(time() - t) + "s")
    return md


def load_training_data():
    logging.info("Loading training data")
    t = time()
    tweets_train, labels_train_txt = funcs.load_labeled_data(train_file)
    logging.debug("Done. " + str(time() - t) + "s")
    return tweets_train, labels_train_txt


def load_test_data():
    logging.info("Loading test data")
    t = time()
    tweets_test, labels_test_txt = funcs.load_labeled_data(test_file)
    logging.debug("Done. " + str(time() - t) + "s")
    return tweets_test, labels_test_txt


def calculate_tweet_embeddings(md, tweets):
    logging.info("Calculating tweet embeddings")
    t = time()
    embeddings = list(map(md.get_tweet_embedding, tweets))
    embeddings = np.array(embeddings)
    logging.debug("Done. " + str(time() - t) + "s")
    return embeddings


def scale_word_embeddings(embeddings_train):
    logging.info("Scaling word embedding vectors")
    t = time()
    embeddings_train_scaled = [funcs.scale_vector(emb) for emb in embeddings_train]
    logging.debug("Done. " + str(time() - t) + "s")
    # embeddings_train_scaled = funcs.regularize_hor(embeddings_train)
    return embeddings_train_scaled


def convert_labels_to_numerical(labels_train_txt):
    logging.info("Converting labels to numerical")
    t = time()
    labels_train_num = funcs.get_labels_numerical(labels_train_txt)
    logging.debug("Done. " + str(time() - t) + "s")
    return labels_train_num


# TODO: Parallelize
def train(models, tweets, embeddings_train_scaled, labels_train_num):
    for model in models:
        logging.info("Training " + model.name + " classifier on training data")
        t = time()
        model.train(tweets, embeddings_train_scaled, labels_train_num)  # 7.396183688299606)
        logging.debug("Done. " + str(time() - t) + "s")


def train_classifiers(tweets_train, embeddings_train_scaled, labels_train_num):
    train(classifiers, tweets_train, embeddings_train_scaled, labels_train_num)


def train_baselines(tweets_train, embeddings_train_scaled, labels_train_num):
    train(baselines, tweets_train, embeddings_train_scaled, labels_train_num)


# TODO: Parallelize
def load_classifier_models():
    for classifier in classifiers:
        model_file = path.join(config.MODELS_DIR, classifier.name + ".pickle")
        logging.info("Loading " + classifier.name + " model from " + model_file)
        t = time()
        classifier.load_model(model_file)
        logging.debug("Done. " + str(time() - t) + "s")


# TODO: Parallelize
def save_classifier_models():
    for classifier in classifiers:
        logging.info("Saving " + classifier.name + " model")
        t = time()
        classifier.save_model(path.join(config.MODELS_DIR, classifier.name + ".pickle"))
        logging.debug("Done. " + str(time() - t) + "s")


# TODO: Parallelize
def test(models, tweets, embeddings, numeric_test_labels):
    for model in models:
        model.predict(tweets, embeddings)
        model.test(numeric_test_labels)


def test_classifiers(tweets, embeddings, numeric_test_labels):
    test(classifiers, tweets, embeddings, numeric_test_labels)


def test_baselines(tweets, embeddings, numeric_test_labels):
    test(baselines, tweets, embeddings, numeric_test_labels)


# TODO: Parallelize
def save_classifier_results():
    for classifier in classifiers:
        classifier.save_results(path.join(config.RESULTS_DIR, classifier.name.lower() + ".txt"))


def print_results(models):
    for model in models:
        model.print()


def main(arguments):
    md = load_word_embeddings()

    if not arguments.no_train:
        # Prepare training data
        tweets_train, labels_train_txt = load_training_data()
        embeddings_train = calculate_tweet_embeddings(md, tweets_train)
        embeddings_train_scaled = scale_word_embeddings(embeddings_train)
        labels_train_num = convert_labels_to_numerical(labels_train_txt)

        # Train classifiers and save the models
        train_classifiers(tweets_train, embeddings_train_scaled, labels_train_num)
        save_classifier_models()

        # Train baselines
        train_baselines(tweets_train, embeddings_train_scaled, labels_train_num)

    else:
        load_classifier_models()

    if not arguments.no_test:
        # Prepare test data
        tweets_test, labels_test_txt = load_test_data()
        embeddings_test = calculate_tweet_embeddings(md, tweets_test)
        embeddings_test_scaled = scale_word_embeddings(embeddings_test)
        labels_test_num = convert_labels_to_numerical(labels_test_txt)

        # Test classifiers and save the results
        test_classifiers(tweets_test, embeddings_test_scaled, labels_test_num)
        save_classifier_results()

        # Test baselines
        test_baselines(tweets_test, embeddings_test_scaled, labels_test_num)

        # Print results
        print_results(classifiers)
        print_results(baselines)


def print_intro(arguments):
    print()
    if not arguments.no_train and not arguments.no_test:
        print('Training and testing classifiers')
    if arguments.no_test:
        print('Training classifiers')
    if arguments.no_train:
        print('Testing classifiers')

    print()
    print('config.py settings:')
    print()
    print('{:24}'.format('Word embedding file:'), config.EMBEDDING_FILE)
    print('{:24}'.format('Training data file:'), config.TRAIN_FILE)
    print()
    print('{:24}'.format('SVM model file:'), config.SVM_MODEL_FILE)
    print('{:24}'.format('LogRes model file:'), config.LOGRES_MODEL_FILE)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='tsabl - train classifiers')

    parser.add_argument('--no-train', action='store_true', help='do not train classifiers')
    parser.add_argument('--no-test', action='store_true', help='do not test classifiers')

    # Logger verbosity parameters
    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level, repeat to increase')
    parser.add_argument('-q', '--quiet', action='store_true', help='no print to console')

    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, args.verbose + 2)]  # capped to number of levels
    logging.basicConfig(level=level, format="%(asctime)s\t%(levelname)s\t%(message)s")

    if args.quiet:
        logging.disable(levels[0])
    else:
        print_intro(args)

    main(args)
