from sklearn import linear_model
from classifiers.models.model import Model


class LogRes(Model):
    def __init__(self, name="LogRes", c=1, solver='sag', multi_class='multinomial', max_iter=1000):
        Model.__init__(self, name=name)

        # Create classifier
        self.model = linear_model.LogisticRegression(C=c, solver=solver, multi_class=multi_class, max_iter=max_iter)

    # Creates and returns Logistic Regression classifier trained on samples and labels
    # Parameters sample and labels of type array-like
    def train(self, tweets, embeddings, labels):
        self.model.fit(embeddings, labels)
        return self

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = self.model.predict(embeddings_train_scaled)
        return self.predictions
