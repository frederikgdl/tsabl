from random import choice

from classifiers.models.model import Model


class RandomUniform(Model):
    def __init__(self, name="RandomUniform"):
        Model.__init__(self, name=name)

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = [choice([0, 1, -1]) for _ in tweets]
        return self.predictions
