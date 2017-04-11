from sklearn import linear_model
from classifiers.models.model import Model


class LogRes(Model):

    def __init__(self):
        Model.__init__(self, "LogRes")

    # Creates and returns Logistic Regression classifier trained on samples and labels
    # Parameters sample and labels of type array-like
    def train(self, tweets, embeddings, labels, c=1, solver='sag', multi_class='multinomial'):
        clf = linear_model.LogisticRegression(C=c, solver=solver, multi_class=multi_class, max_iter=1000)
        clf.fit(embeddings, labels)
        self.model = clf
        return self

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = self.model.predict(embeddings_train_scaled)
        return self.predictions
