import logging
from time import time

import numpy as np
from keras.models import Model
from keras.layers import Embedding, Dense, Reshape, Input, Dropout, merge
from keras.optimizers import Adagrad
import keras.backend as K
import theano.tensor as T

from utils import file_ops, text_processing
import config
import funcs
from tokenizer import Tokenizer


def create_model(window_size, vocab_size, embedding_length, hidden_size, dropout_p):
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
                                init=init_embeddings, name='embedding_layer')
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
    sentiment_layer = Dense(2, activation='linear', init=init_hidden_layer, name='sentiment_output')
    # sentiment_layer = Dense(3, activation='linear', init=init_hidden_layer, name='sentiment_output')

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
    neg_tanh_layer = Dense(hidden_size, activation='tanh', weights=tanh_layer.get_weights())

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
    window_size = config.WINDOW_SIZE
    hidden_size = config.HIDDEN_SIZE
    embedding_length = config.EMBEDDING_LENGTH

    data_file = config.DATA_FILE
    data_file_labeled = config.DATA_FILE_LABELED
    output_file = config.OUTPUT_FILE

    pos_file = config.POS_DATA_FILE
    neg_file = config.NEG_DATA_FILE

    min_freq = config.MIN_WORD_FREQUENCY
    max_nb_words = config.MAX_NUMBER_WORDS
    lowercase = config.LOWERCASE

    nb_epochs = config.EPOCHS
    margin = config.MARGIN
    batch_size = config.BATCH_SIZE
    dropout_p = config.DROPOUT_P
    alpha = config.ALPHA
    adagrad_lr = config.ADAGRAD_LR

    # Read data
    logging.info('Loading tweet data')
    t = time()
    texts, labels = funcs.get_training_data(pos_file, neg_file)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Shuffling tweet data')
    t = time()
    texts, labels = funcs.shuffle_data(texts, labels)
    logging.debug('Done. {}s'.format(str(time() - t)))

    # Use Twokenize (https://github.com/myleott/ark-twokenize-py) to tokenize tweets
    # print("Twokenizing and removing urls, @-mentions, hashtags...")
    # texts = list(map(lambda tweet: ' '.join(text_processing.clean_and_twokenize(tweet)), texts))
    # print("Done.")

    logging.info('Creating vocabulary and one-hot vectors')
    t = time()
    tokenizer = Tokenizer(nb_words=max_nb_words, lower=lowercase, min_freq=min_freq)
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    vocab_map = tokenizer.word_index
    inverse_vocab_map = {v: k for k, v in vocab_map.items()}

    # Add 1 for reserved index 0
    vocab_size = len(vocab_map) + 1
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Converts labels')
    t = time()
    # Turn 'positive' to [1, -1, -1], 'neutral' to [-1, 1, -1] and negative to [-1, -1, 1].
    labels = funcs.get_numeric_labels(labels)
    logging.debug('Done. {}s'.format(str(time() - t)))

    logging.info('Creating context windows and negative samples')
    t = time()
    context_windows, labels = funcs.get_context_windows_labels(seqs, labels, window_size)
    negative_samples = funcs.get_negative_samples(context_windows, vocab_size)
    logging.debug('Done. {}s'.format(str(time() - t)))

    input_array = np.array(context_windows)
    neg_input_array = np.array(negative_samples)
    input_labels = np.array(labels)

    # Create model
    logging.info('Creating model')
    t = time()
    model = create_model(window_size, vocab_size, embedding_length, hidden_size, dropout_p)
    logging.debug('Done. {}s'.format(str(time() - t)))

    # Loss functions
    def context_loss_function(y_true, y_pred):
        # TODO: verify function
        # y_len = y_pred.shape[0]
        # TODO: sizes = 1? not y_len?
        y_pos, y_neg = T.split(y_pred, [1, 1], 2, axis=1)
        # return K.sum(K.maximum(0., 1. - y_pos + y_neg), axis=-1)
        return (1 - alpha) * K.maximum(0., 1. - y_pos + y_neg)

    def sentiment_loss_function(y_true, y_pred):
        # TODO: verify function
        # y_true is [1, -1, -1] for positive, [-1, 1, -1] for neutral etc.
        return alpha * K.maximum(0., 1. - K.sum(y_true*y_pred, axis=1))

    # Optimizer
    optimizer = Adagrad(lr=adagrad_lr)

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
    print('Window size: {}'.format(config.WINDOW_SIZE))
    print('Hidden size: {}'.format(config.HIDDEN_SIZE))
    print('Embedding length: {}'.format(config.EMBEDDING_LENGTH))
    print()
    print('Data file: {}'.format(config.DATA_FILE))
    print('Data file labeled: {}'.format(config.DATA_FILE_LABELED))
    print('Output file: {}'.format(config.OUTPUT_FILE))
    print()
    print('Pos file: {}'.format(config.POS_DATA_FILE))
    print('Neg file: {}'.format(config.NEG_DATA_FILE))
    print()
    print('Min frequency: {}'.format(config.MIN_WORD_FREQUENCY))
    print('Max number of words: {}'.format(config.MAX_NUMBER_WORDS))
    print('Lowercase: {}'.format(config.LOWERCASE))
    print()
    print('Number of epochs: {}'.format(config.EPOCHS))
    print('Margin: {}'.format(config.MARGIN))
    print('Batch size: {}'.format(config.BATCH_SIZE))
    print('Dropout p: {}'.format(config.DROPOUT_P))
    print('Alpha: {}'.format(config.ALPHA))
    print('Adagrad learning rate: {}'.format(config.ADAGRAD_LR))
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

    main()
