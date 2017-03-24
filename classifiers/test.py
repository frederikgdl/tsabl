from time import time
import numpy as np

from embeddings import word_embedding_dict as wedict
import classifiers.config as config
import classifiers.methods as methods
import classifiers.metrics as metrics

from classifiers.baselines.random_uniform import RandomUniform
from classifiers.baselines.random_weighted import RandomWeighted
from classifiers.baselines.afinn_tweets import AfinnTweets
from classifiers.baselines.vader_tweets import VaderTweets
from classifiers.baselines.textblob_tweets import TextblobTweets
from classifiers.baselines.combo_tweets import ComboTweets

test_file = config.TEST_FILE
svm_model_file = config.SVM_MODEL_FILE
logres_model_file = config.LOGRES_MODEL_FILE
word_embed_file = config.EMBEDDING_FILE


def test_baselines(tweets, tokenized_tweets, labels):
    print("\nTesting baselines")
    RandomUniform(tweets, tokenized_tweets, labels).run().test().print()
    RandomWeighted(tweets, tokenized_tweets, labels).run().test().print()
    AfinnTweets(tweets, tokenized_tweets, labels).run().test().print()
    VaderTweets(tweets, tokenized_tweets, labels, threshold=0.1).run().test().print()
    TextblobTweets(tweets, tokenized_tweets, labels, subjectivity_threshold=0.1, polarity_threshold=0.5).run().test().print()
    ComboTweets(tweets, tokenized_tweets, labels).run().test().print()


def main():
    data_file_name = test_file

    print("Loading word embeddings")
    t = time()
    md = wedict.WordEmbeddingDict(word_embed_file)
    print("Done. " + str(time() - t) + "s")

    print("Loading SVM model")
    t = time()
    clf_svm = methods.load_model(svm_model_file)
    print("Done. " + str(time() - t) + "s")

    print("Loading LogRes model")
    t = time()
    clf_logres = methods.load_model(logres_model_file)
    print("Done. " + str(time() - t) + "s")

    print("Loading test data")
    t = time()
    full_tweets = methods.load_tweets_full(data_file_name)
    tweets_test, labels_test_txt = methods.load_labeled_data(data_file_name)
    print("Done. " + str(time() - t) + "s")

    print("Calculating tweet embeddings")
    t = time()
    embeddings_test = list(map(md.get_tweet_embedding, tweets_test))
    embeddings_test = np.array(embeddings_test)
    print("Done. " + str(time() - t) + "s")

    print("Scaling word embedding vectors")
    t = time()
    embeddings_test_scaled = [methods.scale_vector(emb) for emb in embeddings_test]
    embeddings_test_scaled = np.array(embeddings_test_scaled)
    print("Done. " + str(time() - t) + "s")
    # embeddings_test_scaled = methods.regularize_hor(embeddings_test)

    print("Converting labels to numerical")
    t = time()
    labels_test_num = methods.get_labels_numerical(labels_test_txt)
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

    test_baselines(full_tweets, tweets_test, labels_test_num)

main()
