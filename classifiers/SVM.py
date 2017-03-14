from sklearn import svm


# Creates and returns SVM classifier trained on samples and labels
# Parameters sample and labels of type array-like
def train(samples, labels, c=1):
    # Create classifier
    clf = svm.LinearSVC(C=c, class_weight="balanced")

    # Fit classifier to samples and labels
    clf.fit(samples, labels)

    return clf
