from classifiers.models.model import Model
from sklearn import svm


class SVM(Model):

    def __init__(self):
        Model.__init__(self, "SVM")

    def train(self, tweets, train_embeddings, labels_train_num, c=1):
        """
        Creates and returns SVM classifier trained on samples and labels
        Parameters sample and labels of type array-like
        
        :param tweets: 
        :param train_embeddings: 
        :param labels_train_num: 
        :param c: 
        :return: 
        """
        # Create classifier
        clf = svm.LinearSVC(C=c, class_weight="balanced")

        # Fit classifier to samples and labels
        clf.fit(train_embeddings, labels_train_num)

        self.model = clf
        return self

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = self.model.predict(embeddings_train_scaled)
        return self.predictions
