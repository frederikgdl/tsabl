import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense

from utils import file_ops
from tokenizer import Tokenizer
import config
import funcs


def main():
    window_size = config.WINDOW_SIZE
    hidden_size = config.HIDDEN_SIZE
    embedding_length = config.EMBEDDING_LENGTH

    data_file = config.DATA_FILE
    data_file_labeled = config.DATA_FILE_LABELED

    min_freq = config.MIN_WORD_FREQUENCY
    max_nb_words = config.MAX_NUMBER_WORDS
    lowercase = config.LOWERCASE

    #

    if data_file_labeled:
        texts, labels = file_ops.read_labeled_file(data_file)
    else:
        texts = file_ops.read_lines(data_file)

    # TODO: Do preprocessing here (lowercasing, tokenizing, etc.)

    tokenizer = Tokenizer(nb_words=max_nb_words, lower=lowercase, min_freq=min_freq)
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)

    context_windows = funcs.get_context_windows(seqs, window_size)

    # TODO: Unknown words must be handled
    # Add 1 for unknown words
    vocab_size = len(tokenizer.word_counts) + 1

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_length, input_length=window_size))

    # Linear layer
    # Init from a uniform distribution U(-0.01/InputLength, 0.01/InputLength), see Tang16 3.6.2
    model.add(Dense(hidden_size, activation='linear'))

    # hTanh layer
    model.add(Dense(hidden_size, activation='tanh'))

    # Sentiment linear 2
    model.add(Dense(2, activation='linear'))

    input_array = np.array(context_windows)
    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)

    # print(len(model.get_weights()))
    print(output_array.shape)


if __name__ == "__main__":
    main()
