import numpy as np
from keras.models import Model
from keras.layers import Embedding, Dense, Reshape, Input, merge
import keras.backend as K
import theano.tensor as T

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
    output_file = config.OUTPUT_FILE

    min_freq = config.MIN_WORD_FREQUENCY
    max_nb_words = config.MAX_NUMBER_WORDS
    lowercase = config.LOWERCASE

    nb_epochs = config.EPOCHS
    margin = config.MARGIN
    batch_size = config.BATCH_SIZE

    # Read data

    texts, labels = file_ops.read_labeled_file(data_file)

    # if data_file_labeled:
    #     texts, labels = file_ops.read_labeled_file(data_file)
    # else:
    #     texts = file_ops.read_lines(data_file)

    # TODO: Do preprocessing here (lowercasing, tokenizing, etc.)

    tokenizer = Tokenizer(nb_words=max_nb_words, lower=lowercase, min_freq=min_freq)
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    vocab_map = tokenizer.word_index
    inverse_vocab_map = {v: k for k, v in vocab_map.items()}

    # TODO: Unknown words must be handled
    # Add 1 for reserved index 0
    vocab_size = len(vocab_map) + 1

    # Turn 'positive' to [1, -1, -1], 'neutral' to [-1, 1, -1] and negative to [-1, -1, 1].
    labels = funcs.get_numeric_labels(labels)

    context_windows, labels = funcs.get_context_windows_labels(seqs, labels, window_size)
    negative_samples = funcs.get_negative_samples(context_windows, vocab_size)

    input_array = np.array(context_windows)
    neg_input_array = np.array(negative_samples)
    input_labels = np.array(labels)

    # Init functions
    # TODO: verify functions

    def init_embeddings(shape, name=None):
        return K.random_uniform_variable(shape=shape, low=-0.01, high=0.01, name=name)

    # Init from a uniform distribution U(-0.01/InputLength, 0.01/InputLength), see Tang16 3.6.2
    def init_hidden_layer(shape, name=None):
        return K.random_uniform_variable(shape=shape, low=-0.01/shape[0], high=0.01/shape[0], name=name)

    # Model

    # Input

    main_input = Input((window_size,))
    neg_input = Input((window_size,))

    # Embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_length, input_length=window_size,
                                init=init_embeddings)
    # Reshape to concat embeddings in context windows
    reshaped_embedding_layer = Reshape((embedding_length*window_size,))

    # PosMain

    # Linear layer
    linear_layer = Dense(hidden_size, activation='linear', init=init_hidden_layer)

    # hTanh layer
    # TODO: tanh vs hTanh
    tanh_layer = Dense(hidden_size, activation='tanh', init=init_hidden_layer)

    # Context linear 2
    context_layer = Dense(1, activation='linear', init=init_hidden_layer)

    # Sentiment linear 2
    # sentiment_layer = Dense(2, activation='linear', init=init_hidden_layer, name='sentiment_output')
    sentiment_layer = Dense(3, activation='linear', init=init_hidden_layer, name='sentiment_output')

    embeddings = embedding_layer(main_input)
    reshaped_embeddings = reshaped_embedding_layer(embeddings)
    lin_output = linear_layer(reshaped_embeddings)
    tanh_output = tanh_layer(lin_output)
    context_output = context_layer(tanh_output)
    sentiment_output = sentiment_layer(tanh_output)

    # NegMain

    # Linear layer
    neg_linear_layer = Dense(hidden_size, activation='linear', weights=linear_layer.get_weights())

    # hTanh layer
    neg_tanh_layer = Dense(hidden_size, activation='tanh', weights=tanh_layer.get_weights())

    # Context linear 2
    neg_context_layer = Dense(1, activation='linear', weights=context_layer.get_weights())

    # Sentiment linear 2
    # neg_sentiment_layer = Dense(2, activation='linear', weights=sentiment_layer.get_weights())

    neg_embeddings = embedding_layer(neg_input)
    neg_reshaped_embeddings = reshaped_embedding_layer(neg_embeddings)
    neg_lin_output = neg_linear_layer(neg_reshaped_embeddings)
    neg_tanh_output = neg_tanh_layer(neg_lin_output)
    neg_context_output = neg_context_layer(neg_tanh_output)
    # neg_sentiment_output = neg_sentiment_layer(neg_tanh_output)

    merged_context_output = merge([context_output, neg_context_output], mode='concat', concat_axis=-1,
                                  name='merged_context_output')

    model = Model(input=[main_input, neg_input], output=[merged_context_output, sentiment_output])
    # model = Model(input=[main_input, neg_input], output=merged_context_output)
    # model = Model(input=main_input, output=context_output)

    def context_loss_function(y_true, y_pred):
        # TODO: verify function
        # y_len = y_pred.shape[0]
        # TODO: sizes = 1? not y_len?
        y_pos, y_neg = T.split(y_pred, [1, 1], 2, axis=1)
        # return K.sum(K.maximum(0., 1. - y_pos + y_neg), axis=-1)
        return K.maximum(0., 1. - y_pos + y_neg)

    def sentiment_loss_function(y_true, y_pred):
        # TODO: verify function
        # y_true is [1, -1, -1] for positive, [-1, 1, -1] for neutral etc.
        return K.maximum(0., 1. - K.sum(y_true*y_pred, axis=1))

    model.compile(optimizer='sgd', loss={'merged_context_output': context_loss_function,
                                         'sentiment_output': sentiment_loss_function})
    # model.compile(optimizer='sgd', loss=context_loss_function)
    # model.compile(optimizer='sgd', loss='mse')

    model.fit([input_array, neg_input_array], [input_labels, input_labels], nb_epoch=nb_epochs, batch_size=batch_size)
    # output_array = model.predict([input_array, neg_input_array])
    # output_array = model.predict(input_array)

    funcs.dump_embed_file(output_file, inverse_vocab_map, embedding_layer.get_weights()[0])
    print('Done')


if __name__ == "__main__":
    main()
