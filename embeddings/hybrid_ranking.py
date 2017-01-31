from layers.lookup_layer import LookupLayer


class HybridRanking(object):
    def __init__(self, window_size, vocab_size, hidden_size, embedding_length):
        # self.window_size = window_size
        # self.vocab_size = vocab_size
        # self.hidden_size = hidden_size
        # self.embedding_length = embedding_length

        self.lookup = LookupLayer(embedding_length, vocab_size, window_size)
        # self.linear1 = linear_layer.LinearLayer(window_size * embedding_length, hidden_size)
        # self.tanh = tanh_layer.TanhLayer(hidden_size)
        # self.sentiment_linear2 = linear_layer(hidden_size, 2)

        # self.train_function = self._generate_train_function()

    def _generate_train_function(self):
        # return theano.function()
        pass
