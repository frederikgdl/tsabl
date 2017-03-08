import numpy as np
from time import time
from embeddings import word_embedding_dict as wedict
import config
import LogRes
import SVM
import methods

embedding_file = config.EMBEDDING_FILE
train_file = config.TRAIN_FILE

out_file_svm = config.OUT_FILE_SVM
out_file_logres = config.OUT_FILE_LOGRES


def main():
    print("Loading word embeddings")
    t = time()
    md = wedict.WordEmbeddingDict(embedding_file)
    print("Done. " + str(time() - t) + "s")

    print("Loading training data")
    t = time()
    tweets_train, labels_train_txt = methods.load_labeled_data(train_file)
    print("Done. " + str(time() - t) + "s")

    print("Calculating tweet embeddings")
    t = time()
    embeddings_train = list(map(md.get_tweet_embedding, tweets_train))
    embeddings_train = np.array(embeddings_train)
    print("Done. " + str(time() - t) + "s")

    print("Scaling word embedding vectors")
    t = time()
    embeddings_train_scaled = [methods.scale_vector(emb) for emb in embeddings_train]
    print("Done. " + str(time() - t) + "s")
    # embeddings_train_scaled = methods.regularize_hor(embeddings_train)

    print("Converting labels to numerical")
    t = time()
    labels_train_num = methods.get_labels_numerical(labels_train_txt)
    print("Done. " + str(time() - t) + "s")

    print("Training SVM classifier on training data")
    t = time()
    clf_svm = SVM.train(embeddings_train_scaled, labels_train_num, c=1)  # 7.396183688299606)
    print("Done. " + str(time() - t) + "s")

    print("Saving model")
    t = time()
    methods.save_model(clf_svm, out_file_svm)
    print("Done. " + str(time() - t) + "s")

    print("Training Logistic Regression classifier on training data")
    t = time()
    clf_logres = LogRes.train(embeddings_train_scaled, labels_train_num, c=1)
    print("Done. " + str(time() - t) + "s")

    print("Saving model")
    t = time()
    methods.save_model(clf_logres, out_file_logres)
    print("Done. " + str(time() - t) + "s")

main()
