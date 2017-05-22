from classifiers.models.model import Model
from sklearn import svm


class SVM(Model):
    def __init__(self, name="SVM", c=1, class_weight='balanced', penalty='l2', loss='squared_hinge', dual=True,
                 tol=0.0001, multi_class='ovr', fit_intercept=True, intercept_scaling=1, verbose=0,
                 random_state=None, max_iter=1000):
        Model.__init__(self, name=name)

        # Create classifier
        self.model = svm.LinearSVC(C=c, class_weight=class_weight, penalty=penalty, loss=loss, dual=dual, tol=tol,
                                   multi_class=multi_class, fit_intercept=fit_intercept,
                                   intercept_scaling=intercept_scaling, verbose=verbose, random_state=random_state,
                                   max_iter=max_iter)

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
