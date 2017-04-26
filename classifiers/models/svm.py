from classifiers.models.model import Model
from sklearn import svm


class SVM(Model):
    def __init__(self, name="SVM", c=1, class_weight='balanced'):
        Model.__init__(self, name=name)

        # Create classifier
        self.model = svm.LinearSVC(C=c, class_weight=class_weight)

    def train(self, tweets, embeddings, labels):
        """
        Traines SVM classifier on samples and labels
        Parameters embeddings and labels of type array-like
        
        :param tweets: 
        :param embeddings: 
        :param labels: 
        :return: 
        """
        self.model.fit(embeddings, labels)
        return self

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = self.model.predict(embeddings_train_scaled)
        return self.predictions
