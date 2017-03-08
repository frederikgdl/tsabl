import logging
from time import time

import numpy as np
import twitty.metrics as metrics
import twitty.word_embedding_dist as wedict
import twitty.utils as utils

test_file = config.TEST_FILE
svm_model_file = config.SVM_MODEL_FILE
logres_model_file = config.LOGRES_MODEL_FILE
word_embed_file = EMBEDDING_FILE


def main():
    data_file_name = test_file

    print("Loading word embeddings")
    t = time()
    md = wedict.WordEmbeddingDict(word_embed_file)
    print("Done. " + str(time() - t) + "s")

    print("Loading SVM model")
    t = time()
    clf_svm = utils.load_model(svm_model_file)
    print("Done. " + str(time() - t) + "s")

    print("Loading LogRes model")
    t = time()
    clf_logres = utils.load_model(logres_model_file)
    print("Done. " + str(time() - t) + "s")

    print("Loading test data")
    t = time()
    tweets_test, labels_test_txt = utils.load_labeled_data(data_file_name)
    print("Done. " + str(time() - t) + "s")

    print("Calculating tweet embeddings")
    t = time()
    embeddings_test = list(map(md.get_tweet_embedding, tweets_test))
    embeddings_test = np.array(embeddings_test)
    print("Done. " + str(time() - t) + "s")

    print("Scaling word embedding vectors")
    t = time()
    embeddings_test_scaled = [utils.scale_vector(emb) for emb in embeddings_test]
    embeddings_test_scaled = np.array(embeddings_test_scaled)
    print("Done. " + str(time() - t) + "s")
    # embeddings_test_scaled = utils.regularize_hor(embeddings_test)

    print("Converting labels to numerical")
    t = time()
    labels_test_num = utils.get_labels_numerical(labels_test_txt)
    print("Done. " + str(time() - t) + "s")

    print("Classifying test samples using the SVM model")
    t = time()
    predictions_svm = clf_svm.predict(embeddings_test_scaled)
    print("Done. " + str(time() - t) + "s")

    print("Calculating scores for SVM classifier")
    t = time()
    F1_score_svm = metrics.calc_F1_score_with_neutrals(predictions_svm, labels_test_num)
    print("Done. " + str(time() - t) + "s")

    print("Classifying test samples using the LogRes model")
    t = time()
    predictions_logres = clf_logres.predict(embeddings_test_scaled)
    print("Done. " + str(time() - t) + "s")

    print("Calculating scores for LogRes classifier")
    t = time()
    F1_score_logres = metrics.calc_F1_score_with_neutrals(predictions_logres, labels_test_num)
    print("Done. " + str(time() - t) + "s")

    print("F1-score SVM classifier:\t" + str(F1_score_svm))
    print("F1-score LogRes classifier:\t" + str(F1_score_logres))
