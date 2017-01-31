import numpy
import theano
import theano.tensor as T

from .abstract_layer import AbstractLayer


class LookupLayer(AbstractLayer):
    def __init__(self, embedding_length, vocab_size, input_length):
        self.embedding_length = embedding_length
        self.vocab_size = vocab_size
        self.input_length = input_length

        table_values = numpy.zeros((self.vocab_size, self.embedding_length), dtype=theano.config.floatX)
        self.table = theano.shared(value=table_values, name='table', borrow=True)

        input_values = numpy.zeros((self.input_length,), dtype=theano.config.floatX)
        self.input = theano.shared(value=input_values, name='input', borrow=True)

        output_values = numpy.zeros((self.embedding_length * self.input_length,), dtype=theano.config.floatX)
        self.output = theano.shared(value=output_values, name='output', borrow=True)

        output_g_values = numpy.zeros((self.embedding_length * self.input_length,), dtype=theano.config.floatX)
        self.output_g = theano.shared(value=output_g_values, name='outputG', borrow=True)

        # ada_lr_values = numpy.zeros((self.vocab_size, self.embedding_length), dtype=theano.config.floatX)
        # self.ada_lr = theano.shared(value=ada_lr_values, name='input', borrow=True)

    def randomize(self, rng, low, high):
        table_values = numpy.asarray(
            rng.uniform(
                low=-low,
                high=high,
                size=(self.vocab_size, self.embedding_length)
            ),
            dtype=theano.config.floatX
        )
        self.table = theano.shared(value=table_values, name='table', borrow=True)