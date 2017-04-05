import logging
from time import time
import numpy as np

import classifiers.LogRes as LogRes
import classifiers.SVM as SVM
import classifiers.config as config
import classifiers.funcs as funcs
import classifiers.word_embedding_dict as wedict

embedding_file = config.EMBEDDING_FILE
train_file = config.TRAIN_FILE

out_file_svm = config.SVM_MODEL_FILE
out_file_logres = config.LOGRES_MODEL_FILE


def main():
    logging.info("Loading word embeddings")
    t = time()
    md = wedict.WordEmbeddingDict(embedding_file)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Loading training data")
    t = time()
    tweets_train, labels_train_txt = funcs.load_labeled_data(train_file)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Calculating tweet embeddings")
    t = time()
    embeddings_train = list(map(md.get_tweet_embedding, tweets_train))
    embeddings_train = np.array(embeddings_train)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Scaling word embedding vectors")
    t = time()
    embeddings_train_scaled = [funcs.scale_vector(emb) for emb in embeddings_train]
    logging.debug("Done. " + str(time() - t) + "s")
    # embeddings_train_scaled = funcs.regularize_hor(embeddings_train)

    logging.info("Converting labels to numerical")
    t = time()
    labels_train_num = funcs.get_labels_numerical(labels_train_txt)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Training SVM classifier on training data")
    t = time()
    clf_svm = SVM.train(embeddings_train_scaled, labels_train_num, c=1)  # 7.396183688299606)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Saving model")
    t = time()
    funcs.save_model(clf_svm, out_file_svm)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Training Logistic Regression classifier on training data")
    t = time()
    clf_logres = LogRes.train(embeddings_train_scaled, labels_train_num, c=1)
    logging.debug("Done. " + str(time() - t) + "s")

    logging.info("Saving model")
    t = time()
    funcs.save_model(clf_logres, out_file_logres)
    logging.debug("Done. " + str(time() - t) + "s")


def print_intro():
    print()
    print('Training classifiers')
    print()
    print('config.py settings:')
    print()
    print('Word embedding file:\t{}'.format(config.EMBEDDING_FILE))
    print('Training data file:\t{}'.format(config.TRAIN_FILE))
    print('Embedding length:\t{}'.format(config.EMBEDDING_LENGTH))
    print()
    print('SVM model file:\t\t{}'.format(config.SVM_MODEL_FILE))
    print('LogRes model file:\t{}'.format(config.LOGRES_MODEL_FILE))
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='tsabl - train classifiers')

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
