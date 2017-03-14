from sklearn import linear_model


# Creates and returns Logistic Regression classifier trained on samples and labels
# Parameters sample and labels of type array-like
def train(samples, labels, c=1, solver='sag', multi_class='multinomial'):
    clf = linear_model.LogisticRegression(C=c, solver=solver, multi_class=multi_class, max_iter=1000)

    clf.fit(samples, labels)

    return clf
