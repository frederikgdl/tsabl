import numpy as np

import utils
import config
import hybrid_ranking


def train():

    rng = np.random.RandomState(config.SEED)
    window_size = config.WINDOW_SIZE
    hidden_size = config.HIDDEN_SIZE
    embedding_length = config.EMBEDDING_LENGTH

    vocab = utils.get_vocab(config.VOCAB_FILE)
    assert vocab
    vocab_size = len(vocab)

    model = hybrid_ranking.HybridRanking(window_size, vocab_size, hidden_size, embedding_length)


if __name__ == "__main__":
    train()