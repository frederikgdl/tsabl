import classifiers.train_and_test as train_and_test
from classifiers.models.svm import SVM
from os import path, listdir

path_of_this_file = path.dirname(path.realpath(__file__))

# Directory containing embeddings of different epochs
embeddings_dir = path.join(path_of_this_file, "../data/embeddings/binary_sa_embedding")

for embeddings_file in sorted(listdir(embeddings_dir)):
    # Configure test_and_train
    train_and_test.classifiers = [SVM()]
    train_and_test.baselines = []
    train_and_test.save_format = 'data'
    train_and_test.embedding_file = path.join(embeddings_dir, embeddings_file)
    train_and_test.verbose = -2
    train_and_test.quiet = True

    print(embeddings_file)

    # Run
    train_and_test.main()
