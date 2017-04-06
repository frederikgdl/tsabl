import logging
from time import time
import numpy as np

import classifiers.config as config
import classifiers.funcs as funcs
import classifiers.metrics as metrics
import classifiers.word_embedding_dict as wedict
from classifiers.baselines.afinn_tweets import AfinnTweets
from classifiers.baselines.combo_tweets import ComboTweets
from classifiers.baselines.random_uniform import RandomUniform
from classifiers.baselines.random_weighted import RandomWeighted
from classifiers.baselines.textblob_tweets import TextblobTweets
from classifiers.baselines.vader_tweets import VaderTweets

test_file = config.TEST_FILE
svm_model_file = config.SVM_MODEL_FILE
logres_model_file = config.LOGRES_MODEL_FILE
word_embed_file = config.EMBEDDING_FILE


def test_baselines(tweets, tokenized_tweets, labels):
    RandomUniform(tweets, tokenized_tweets, labels).run().test().print()
    RandomWeighted(tweets, tokenized_tweets, labels).run().test().print()
    AfinnTweets(tweets, tokenized_tweets, labels).run().test().print()
    VaderTweets(tweets, tokenized_tweets, labels, threshold=0.1).run().test().print()
    TextblobTweets(tweets, tokenized_tweets, labels, subjectivity_threshold=0.1, polarity_threshold=0.5).run().test().print()
    ComboTweets(tweets, tokenized_tweets, labels).run().test().print()


def main():
    data_file_name = test_file

    logging.info("Loading word embeddings")
    t = time()
    md = wedict.WordEmbeddingDict(word_embed_file)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Loading SVM model")
    t = time()
    clf_svm = funcs.load_model(svm_model_file)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Loading LogRes model")
    t = time()
    clf_logres = funcs.load_model(logres_model_file)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Loading test data")
    t = time()
    full_tweets = funcs.load_tweets_full(data_file_name)
    tweets_test, labels_test_txt = funcs.load_labeled_data(data_file_name)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Calculating tweet embeddings")
    t = time()
    embeddings_test = list(map(md.get_tweet_embedding, tweets_test))
    embeddings_test = np.array(embeddings_test)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Scaling word embedding vectors")
    t = time()
    embeddings_test_scaled = [funcs.scale_vector(emb) for emb in embeddings_test]
    embeddings_test_scaled = np.array(embeddings_test_scaled)
    logging.debug("Done. " + str(time() - t) + "s")
    # embeddings_test_scaled = funcs.regularize_hor(embeddings_test)

    logging.info("Converting labels to numerical")
    t = time()
    labels_test_num = funcs.get_labels_numerical(labels_test_txt)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Classifying test samples using the SVM model")
    t = time()
    predictions_svm = clf_svm.predict(embeddings_test_scaled)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Calculating scores for SVM classifier")
    t = time()
    F1_score_svm = metrics.calc_F1_score_with_neutrals(predictions_svm, labels_test_num)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Classifying test samples using the LogRes model")
    t = time()
    predictions_logres = clf_logres.predict(embeddings_test_scaled)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Calculating scores for LogRes classifier")
    t = time()
    F1_score_logres = metrics.calc_F1_score_with_neutrals(predictions_logres, labels_test_num)
    logging.debug("Done. " + str(time() - t) + "s")

    print("F1-score SVM classifier:\t" + str(F1_score_svm))
    print("F1-score LogRes classifier:\t" + str(F1_score_logres))

    logging.info("\nTesting baselines")
    t = time()
    test_baselines(full_tweets, tweets_test, labels_test_num)
    logging.debug("Done. " + str(time() - t) + "s")


def print_intro():
    print()
    print('Testing classifiers')
    print()
    print('config.py settings:')
    print()
    print('Testing data file:\t{}'.format(config.TRAIN_FILE))
    print()
    print('SVM model file:\t\t{}'.format(config.SVM_MODEL_FILE))
    print('LogRes model file:\t{}'.format(config.LOGRES_MODEL_FILE))
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='tsabl - test classifiers')

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
        print_intro()

    main()
