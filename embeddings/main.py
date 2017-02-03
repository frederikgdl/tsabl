import numpy as np

from utils import file_ops
import config
import funcs
import hybrid_ranking
from twitter.api import TwitterApi

twitter_api = TwitterApi()


def init():
    ids = file_ops.read_twitter_id_file(config.DATA_FILE, config.DATA_FILE_LABELED)
    print("Fetching", len(ids), "tweets.")
    tweets = twitter_api.bulk_get_statuses(ids)
    print("Got ", len(tweets), "tweets.")
    texts = list(map(lambda t: t["text"], tweets))

    # TODO: Do preprocessing here (lowercasing, tokenizing, etc.)

    vocab = funcs.get_vocab(texts, config.MIN_WORD_FREQUENCY)
    print("Vocabulary size:", len(vocab))


def train():

    rng = np.random.RandomState(config.SEED)
    window_size = config.WINDOW_SIZE
    hidden_size = config.HIDDEN_SIZE
    embedding_length = config.EMBEDDING_LENGTH

    vocab, corpus = funcs.get_vocab(config.VOCAB_FILE)
    assert vocab
    vocab_size = len(vocab)

    model = hybrid_ranking.HybridRanking(window_size, vocab_size, hidden_size, embedding_length)


if __name__ == "__main__":
    init()
    #train()
