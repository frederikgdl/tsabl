from os import getenv
import logging
from time import time

import numpy as np
from keras.models import Model
from keras.layers import Embedding, Dense, Reshape, Input, Dropout, merge
from keras.optimizers import Adagrad, SGD
import keras.backend as K

import embeddings.config as config
import embeddings.funcs as funcs

KERAS_BACKEND = getenv('KERAS_BACKEND')
if KERAS_BACKEND == 'theano':
    import theano.tensor as T
else:
    import tensorflow


# Split context output using theano or tensorflow depending on backend
def split(tensor, size_splits, n_splits, axis):
    if KERAS_BACKEND == 'theano':
        return T.split(tensor, size_splits, n_splits, axis=axis)
    else:
        return tensorflow.split(tensor, size_splits, axis=axis)


# Create Keras model
def create_model(window_size, vocab_size, embedding_length, hidden_size, dropout_p, sentiment_classes):
    def init_embeddings(shape, name=None):
        return K.random_uniform_variable(shape=shape, low=-0.01, high=0.01, name=name)

    # Init from a uniform distribution U(-0.01/InputLength, 0.01/InputLength), see Tang16 3.6.2
    def init_hidden_layer(shape, name=None):
        return K.random_uniform_variable(shape=shape, low=-0.01/shape[0], high=0.01/shape[0], name=name)

    # Activation function for htanh layers
    def htanh(x):
        # return K.min(K.max(x, -1, keepdims=True), 1)
        return K.clip(x, -1, 1)

    # Input
    main_input = Input((window_size,))
    neg_input = Input((window_size,))

    # Embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_length, input_length=window_size,
                                init=init_embeddings, name='embedding_layer')
    # Reshape to concat embeddings in context windows
    reshaped_embedding_layer = Reshape((embedding_length*window_size,))

    # PosMain

    # Linear layer
    linear_layer = Dense(hidden_size, activation='linear', init=init_hidden_layer)

    # hTanh layer
    # TODO: tanh vs hTanh
    tanh_layer = Dense(hidden_size, activation=htanh, init=init_hidden_layer)

    # Context linear 2
    context_layer = Dense(1, activation='linear', init=init_hidden_layer)

    # Sentiment linear 2
    sentiment_layer = Dense(sentiment_classes, activation='linear', init=init_hidden_layer, name='sentiment_output')

    embeddings = embedding_layer(main_input)
    reshaped_embeddings = reshaped_embedding_layer(embeddings)
    lin_output = Dropout(dropout_p)(linear_layer(reshaped_embeddings))
    tanh_output = Dropout(dropout_p)(tanh_layer(lin_output))
    context_output = context_layer(tanh_output)
    sentiment_output = sentiment_layer(tanh_output)

    # NegMain

    # Linear layer
    neg_linear_layer = Dense(hidden_size, activation='linear', weights=linear_layer.get_weights())

    # hTanh layer
    neg_tanh_layer = Dense(hidden_size, activation=htanh, weights=tanh_layer.get_weights())

    # Context linear 2
    neg_context_layer = Dense(1, activation='linear', weights=context_layer.get_weights())

    # Sentiment linear 2
    # neg_sentiment_layer = Dense(2, activation='linear', weights=sentiment_layer.get_weights())

    neg_embeddings = embedding_layer(neg_input)
    neg_reshaped_embeddings = reshaped_embedding_layer(neg_embeddings)
    neg_lin_output = Dropout(dropout_p)(neg_linear_layer(neg_reshaped_embeddings))
    neg_tanh_output = Dropout(dropout_p)(neg_tanh_layer(neg_lin_output))
    neg_context_output = neg_context_layer(neg_tanh_output)
    # neg_sentiment_output = neg_sentiment_layer(neg_tanh_output)

    merged_context_output = merge([context_output, neg_context_output], mode='concat', concat_axis=-1,
                                  name='merged_context_output')

    model = Model(input=[main_input, neg_input], output=[merged_context_output, sentiment_output])

    return model


def main():
    min_freq = config.MIN_WORD_FREQUENCY

    window_size = config.WINDOW_SIZE
    hidden_size = config.HIDDEN_SIZE
    embedding_length = config.EMBEDDING_LENGTH
    nb_epochs = config.EPOCHS
    margin = float(config.MARGIN)
    batch_size = config.BATCH_SIZE
    dropout_p = config.DROPOUT_P
    alpha = config.ALPHA
    learning_rate = config.LEARNING_RATE
    sentiment_classes = config.SENTIMENT_CLASSES
    use_adagrad = config.USE_ADAGRAD

    if sentiment_classes not in [2, 3]:
        logging.critical('The number of supported sentiment classes is 2 or 3. Number given: {}'
                         .format(sentiment_classes))
        exit(1)

    output_file = config.OUTPUT_FILE

    pos_file = config.POS_DATA_FILE
    neg_file = config.NEG_DATA_FILE

    file_paths = [pos_file, neg_file]
    sentiment_labels = ['positive', 'negative']

    if sentiment_classes == 3:
        neu_file = config.NEU_DATA_FILE
        file_paths.append(neu_file)
        sentiment_labels.append('neutral')

    logging.info('Loading tweet data')
    t = time()
    tweets, labels = funcs.get_training_data(file_paths, sentiment_labels)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Shuffling tweet data')
    t = time()
    tweets, labels = funcs.shuffle_data(tweets, labels)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Creating vocabulary')
    t = time()
    vocab_map = funcs.get_vocab(tweet_texts=tweets, min_freq=min_freq)
    inverse_vocab_map = {v: k for k, v in vocab_map.items()}
    vocab_size = len(vocab_map)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Padding tweets')
    t = time()
    tweets = funcs.pad_tweets(tweets)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Converting labels')
    t = time()
    # Turn 'positive' to [1, -1, -1], 'neutral' to [-1, 1, -1] and negative to [-1, -1, 1].
    labels = funcs.get_numeric_labels(labels, sentiment_classes)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Creating context windows and negative samples')
    t = time()
    context_windows, labels = funcs.get_context_windows_labels(tweets, labels, window_size, vocab_map)
    negative_samples = funcs.get_negative_samples(context_windows, vocab_size)
    logging.debug('Done. {}s'.format(str(time() - t)))

    input_array = np.array(context_windows)
    neg_input_array = np.array(negative_samples)
    input_labels = np.array(labels)

    # Create model
    logging.info('Creating model')
    t = time()
    model = create_model(window_size, vocab_size, embedding_length, hidden_size, dropout_p, sentiment_classes)
    logging.debug('Done. {}s'.format(str(time() - t)))

    # Loss functions
    def context_loss_function(y_true, y_pred):
        # TODO: verify function
        # y_len = y_pred.shape[0]
        # TODO: sizes = 1? not y_len?
        # y_pos, y_neg = y_pred
        y_pos, y_neg = split(y_pred, [1, 1], 2, axis=1)
        # y_pos, y_neg = T.split(y_pred, [1, 1], 2, axis=1)
        # return K.sum(K.maximum(0., 1. - y_pos + y_neg), axis=-1)
        # return (1 - alpha) * K.maximum(0., 1. - y_pos + y_neg)

        # Add 0*y_true to add y_true to graph
        return (1 - alpha) * K.maximum(0., margin - y_pos + y_neg) + 0*y_true

    def sentiment_loss_function(y_true, y_pred):
        # TODO: verify function
        # ([1, -1, 0], [1, 0, -1])
        # y_true is [1, -1, -1] for positive, [-1, 1, -1] for neutral etc.
        # return alpha * K.maximum(0., 1. - K.sum(y_true*y_pred, axis=1))
        labels_one, labels_two = split(y_true, [3, 3], 2, axis=1)
        return alpha * (K.maximum(0., margin - K.sum(labels_one*y_pred, axis=1))
                        + K.maximum(0., margin - K.sum(labels_two*y_pred, axis=1)))

    # Optimizer
    if use_adagrad:
        optimizer = Adagrad(lr=learning_rate)
    else:
        optimizer = SGD(lr=learning_rate)

    logging.info('Compiling model')
    t = time()
    model.compile(optimizer=optimizer, loss={'merged_context_output': context_loss_function,
                                             'sentiment_output': sentiment_loss_function})
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Fitting model')
    t = time()
    model.fit([input_array, neg_input_array], [input_labels, input_labels],
              nb_epoch=nb_epochs, batch_size=batch_size, shuffle=True)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Writing word embeddings to file')
    t = time()
    funcs.dump_embed_file(output_file, inverse_vocab_map, model.get_layer('embedding_layer').get_weights()[0])
    logging.debug('Done. {}s'.format(str(time() - t)))

    print('Done')


def print_intro():
    print()
    print('Training sentiment enhanced word embeddings')
    print()
    print('config.py settings:')
    print()
    print('Output file:\t\t{}'.format(config.OUTPUT_FILE))
    print()
    print('Positive file:\t\t{}'.format(config.POS_DATA_FILE))
    print('Negative file:\t\t{}'.format(config.NEG_DATA_FILE))
    if config.SENTIMENT_CLASSES == 3:
        print('Neutral file:\t\t{}'.format(config.NEG_DATA_FILE))
    print()
    print('Min frequency:\t\t{}'.format(config.MIN_WORD_FREQUENCY))
    print()
    print('Number of epochs:\t{}'.format(config.EPOCHS))
    print('Margin:\t\t\t{}'.format(config.MARGIN))
    print('Batch size:\t\t{}'.format(config.BATCH_SIZE))
    print('Dropout p:\t\t{}'.format(config.DROPOUT_P))
    print('Alpha:\t\t\t{}'.format(config.ALPHA))
    print('Learning rate:\t\t{}'.format(config.LEARNING_RATE))
    print('Sentiment classes:\t{}'.format(config.SENTIMENT_CLASSES))
    print('Window size:\t\t{}'.format(config.WINDOW_SIZE))
    print('Hidden size:\t\t{}'.format(config.HIDDEN_SIZE))
    print('Embedding length:\t{}'.format(config.EMBEDDING_LENGTH))
    print('Using Adagrad:\t\t{}'.format(config.USE_ADAGRAD))
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='tsabl - creating sentiment enhanced word embeddings')

    # Logger verbosity parameters
    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level, repeat to increase')
    parser.add_argument('-q', '--quiet', action='store_true', help='no print to console')

    args = parser.parse_args()

    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, args.verbose + 2)]  # capped to number of levels
    logging.basicConfig(level=level, format="%(asctime)s\t%(levelname)s\t%(message)s")

    if args.quiet:
        logging.disable(levels[0])
    else:
        print_intro()

    main()
