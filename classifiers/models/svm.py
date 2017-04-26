from classifiers.models.model import Model
from sklearn import svm


class SVM(Model):

    def __init__(self, c=1, class_weight='balanced'):
        Model.__init__(self, "SVM")

        # Create classifier
        self.model = svm.LinearSVC(C=c, class_weight=class_weight)

    def train(self, tweets, embeddings, labels):
        """
        Creates and returns SVM classifier trained on samples and labels
        Parameters sample and labels of type array-like
        
        :param tweets: 
        :param embeddings: 
        :param labels: 
        :return: 
        """
        # Fit classifier to samples and labels
        self.model.fit(embeddings, labels)
        return self

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = self.model.predict(embeddings_train_scaled)
        return self.predictions
