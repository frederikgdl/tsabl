from afinn import Afinn
from utils.misc import sign

from classifiers.models.model import Model


class AfinnModel(Model):
    def __init__(self):
        Model.__init__(self, "AFINN")
        self.afinn = Afinn(emoticons=True)

    def predict(self, tweets, embeddings_train_scaled):
        self.predictions = list(map(lambda tweet: sign(self.afinn.score(tweet)), tweets))
        return self.predictions
