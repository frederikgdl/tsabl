import logging
from os import path
from time import time

import numpy as np

import classifiers.config as config
import classifiers.funcs as funcs
from classifiers.k_fold import KFoldValidator
from classifiers.word_embedding_dict import WordEmbeddingDict
from utils import file_ops

embedding_file = config.EMBEDDING_FILE
train_file = config.TRAIN_FILE
test_file = config.TEST_FILE
k = config.K
skip_training = config.SKIP_TRAINING
skip_testing = config.SKIP_TESTING
models_dir = config.MODELS_DIR
results_dir = config.RESULTS_DIR
classifiers = config.CLASSIFIERS
baselines = config.BASELINES
verbose = config.VERBOSE
quiet = config.QUIET

logger = None


def load_word_embeddings():
    logger.info("Loading word embeddings")
    t = time()
    word_embeddings = WordEmbeddingDict(embedding_file)
    logger.debug("Done. " + str(time() - t) + "s")
    return word_embeddings


def load_training_data():
    logger.info("Loading training data")
    t = time()
    tweets_train, labels_train_txt = file_ops.load_labeled_data(train_file)
    logger.debug("Done. " + str(time() - t) + "s")
    return tweets_train, labels_train_txt


def load_test_data():
    logger.info("Loading test data")
    t = time()
    tweets_test, labels_test_txt = file_ops.load_labeled_data(test_file)
    logger.debug("Done. " + str(time() - t) + "s")
    return tweets_test, labels_test_txt


def calculate_tweet_embeddings(word_embeddings, tweets):
    logger.info("Calculating tweet embeddings")
    t = time()
    tweets = [tweet.split() for tweet in tweets]
    embeddings = list(map(word_embeddings.get_tweet_embedding, tweets))
    embeddings = np.array(embeddings)
    logger.debug("Done. " + str(time() - t) + "s")
    return embeddings


def scale_word_embeddings(embeddings):
    logger.info("Scaling word embedding vectors")
    t = time()
    embeddings_scaled = [funcs.scale_vector(emb) for emb in embeddings]
    logger.debug("Done. " + str(time() - t) + "s")
    # embeddings_scaled = funcs.regularize_hor(embeddings)
    return embeddings_scaled


def convert_labels_to_numerical(labels_txt):
    logger.info("Converting labels to numerical")
    t = time()
    labels_num = funcs.get_labels_numerical(labels_txt)
    logger.debug("Done. " + str(time() - t) + "s")
    return labels_num


def train(models, tweets, embeddings, labels):
    for model in models:
        logger.info("Training " + model.name + " classifier on training data")
        t = time()
        model.train(tweets, embeddings, labels)  # 7.396183688299606)
        logger.debug("Done. " + str(time() - t) + "s")


def train_classifiers(tweets, tweet_embeddings, labels):
    train(classifiers, tweets, tweet_embeddings, labels)


def train_baselines(tweets, tweet_embeddings, labels):
    train(baselines, tweets, tweet_embeddings, labels)


def do_k_fold_validation(k, tweets, tweet_embeddings, labels):
    kfold = KFoldValidator(k, tweets, tweet_embeddings, labels)
    for classifier in classifiers:
        res = kfold.run(classifier)
        file_ops.write_to_file(str(res),
                               path.join(config.RESULTS_DIR, classifier.name.lower() + ".kfold" + str(k) + ".txt"))
    for baseline in baselines:
        kfold.run(baseline)


def load_classifier_models():
    for classifier in classifiers:
        model_file = path.join(config.MODELS_DIR, classifier.name + ".pickle")
        logger.info("Loading " + classifier.name + " model from " + model_file)
        t = time()
        classifier.load_model(model_file)
        logger.debug("Done. " + str(time() - t) + "s")


def save_classifier_models():
    for classifier in classifiers:
        logger.info("Saving " + classifier.name + " model")
        t = time()
        classifier.save_model(path.join(config.MODELS_DIR, classifier.name + ".pickle"))
        logger.debug("Done. " + str(time() - t) + "s")


def test(models, tweets, embeddings, labels):
    for model in models:
        model.predict(tweets, embeddings)
        model.test(labels)


def test_classifiers(tweets, embeddings, numeric_test_labels):
    test(classifiers, tweets, embeddings, numeric_test_labels)


def test_baselines(tweets, embeddings, numeric_test_labels):
    test(baselines, tweets, embeddings, numeric_test_labels)


def save_results(models):
    for model in models:
        model.save_results(path.join(results_dir, model.name.lower()))


def print_results(models):
    for model in models:
        model.print()


def main():
    global logger
    logger = setup_logger()

    word_embeddings = load_word_embeddings()

    if skip_training:
        # Load saved classifier models from files
        load_classifier_models()
    else:
        # Prepare training data
        tweets_train, labels_train_txt = load_training_data()
        embeddings_train = calculate_tweet_embeddings(word_embeddings, tweets_train)
        embeddings_train_scaled = scale_word_embeddings(embeddings_train)
        labels_train_num = convert_labels_to_numerical(labels_train_txt)

        # Do K-fold validation. This does both training and testing, so we return after it's done.
        if k > 1:
            do_k_fold_validation(k, tweets_train, embeddings_train_scaled, labels_train_num)
            print_results(classifiers)
            print_results(baselines)
            return

        # Train classifiers and save the models
        train_classifiers(tweets_train, embeddings_train_scaled, labels_train_num)
        save_classifier_models()

        # Train baselines
        train_baselines(tweets_train, embeddings_train_scaled, labels_train_num)

    # If we are to skip testing, there's not more to do, so we can return from the function
    if skip_testing:
        return

    # Prepare test data
    tweets_test, labels_test_txt = load_test_data()
    embeddings_test = calculate_tweet_embeddings(word_embeddings, tweets_test)
    embeddings_test_scaled = scale_word_embeddings(embeddings_test)
    labels_test_num = convert_labels_to_numerical(labels_test_txt)

    # Test classifiers and save the results
    test_classifiers(tweets_test, embeddings_test_scaled, labels_test_num)
    save_results(classifiers)

    # Test baselines
    test_baselines(tweets_test, embeddings_test_scaled, labels_test_num)
    save_results(baselines)

    # Print results
    if not quiet and verbose >= 0:
        print_results(classifiers)
        print_results(baselines)


def print_intro(arguments):
    print()
    if not arguments.skip_training and not arguments.skip_testing:
        print('Training and testing classifiers')
    if arguments.skip_testing:
        print('Training classifiers')
    if arguments.skip_training:
        print('Testing classifiers')

    print()
    print('config.py settings:')
    print()
    print('{:24}'.format('Word embedding file:'), config.EMBEDDING_FILE)
    print('{:24}'.format('Training data file:'), config.TRAIN_FILE)
    print('{:24}'.format('Test data file:'), config.TEST_FILE)
    print()
    print('{:24}'.format('Model directory:'), config.MODELS_DIR)
    print('{:24}'.format('Results directory:'), config.RESULTS_DIR)
    print()


def setup_logger():
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, verbose + 2)]  # capped to number of levels

    # create logger
    new_logger = logging.getLogger('train_and_test')
    new_logger.setLevel(level)
    new_logger.propagate = False

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    new_logger.addHandler(ch)
    return new_logger

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='tsabl - train classifiers')

    # File arguments
    parser.add_argument('-e', '--embedding-file', default=config.EMBEDDING_FILE, help='file containing word embeddings')
    parser.add_argument('-tr', '--train-file', default=config.TRAIN_FILE, help='file containing training data')
    parser.add_argument('-te', '--test-file', default=config.TEST_FILE, help='file containing test data')

    # Directory arguments
    parser.add_argument('-m', '--models-dir', default=config.MODELS_DIR, help='file containing training data')
    parser.add_argument('-r', '--results-dir', default=config.RESULTS_DIR, help='file containing test data')

    # Training and testing arguments
    parser.add_argument('-k', '--k-fold', type=int, default=config.K,
                        help='perform k-fold validation with given k, must be greater than 1')
    parser.add_argument('--skip-training', action='store_true', help='do not train classifiers')
    parser.add_argument('--skip-testing', action='store_true', help='do not test classifiers')

    # Logger verbosity parameters
    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level, repeat to increase')
    parser.add_argument('-q', '--quiet', action='store_true', help='no print to console')

    args = parser.parse_args()

    embedding_file = args.embedding_file
    train_file = args.train_file
    test_file = args.test_file
    k = args.k_fold
    skip_training = args.skip_training
    skip_testing = args.skip_testing
    models_dir = args.models_dir
    results_dir = args.results_dir
    verbose = args.verbose
    quiet = args.quiet

    if args.quiet:
        logging.disable(logging.ERROR)
    else:
        print_intro(args)

    main()
