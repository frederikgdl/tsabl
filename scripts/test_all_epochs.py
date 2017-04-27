import classifiers.train_and_test as train_and_test
from classifiers.models.svm import SVM
from os import path, listdir
import matplotlib.pyplot as plt

path_of_this_file = path.dirname(path.realpath(__file__))

# Directory containing embeddings of different epochs
embeddings_dir = path.join(path_of_this_file, "../data/embeddings/binary_sa_embedding")

# Sort by suffix number of files. Turn to int so that '7' is treated as less that '18', for instance.
embeddings_files = sorted(listdir(embeddings_dir), key=lambda f: int(f.split("-")[-1]))

results = []

for embeddings_file in embeddings_files:
    print(embeddings_file)

    # Configure test_and_train
    train_and_test.classifiers = [SVM()]
    train_and_test.baselines = []
    train_and_test.embedding_file = path.join(embeddings_dir, embeddings_file)
    train_and_test.verbose = -2
    train_and_test.quiet = True

    # Run
    train_and_test.main()

    # Save results
    result = train_and_test.classifiers[0].results.clone()
    results.append(result)

# Plot
x = list(range(1, 1 + len(embeddings_files)))
plt.plot(x, list(map(lambda r: r['ternary_macro_f1_score'], results)), 'b-', label='Macro F1')
plt.plot(x, list(map(lambda r: r['f1_pn_score'], results)), 'r-', label='F1 PN')
# plt.axis([0, len(embeddings_files), 0, 1])
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Scores')
plt.legend()
plt.show()
