import keras.preprocessing.text as keras_text


class Tokenizer(keras_text.Tokenizer):
    def __init__(self, *args, **kwargs):
        self.min_freq = kwargs.pop('min_freq', None)
        super().__init__(*args, **kwargs)

    # Override
    def texts_to_sequences_generator(self, texts):
        """Transforms each text in texts in a sequence of integers.

        Only top "nb_words" most frequent words will be taken into account.
        Only words with frequencies higher than "min_freq" will be taken into account.
        Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of texts (strings).

        # Yields
            Yields individual sequences.
        """
        nb_words = self.nb_words
        min_freq = self.min_freq
        for text in texts:
            seq = text if self.char_level else keras_text.text_to_word_sequence(text,
                                                                     self.filters,
                                                                     self.lower,
                                                                     self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if nb_words and i >= nb_words:
                        continue

                    # Filter for minimum frequency
                    elif min_freq and self.word_counts.get(w) < min_freq:
                        continue

                    else:
                        vect.append(i)
            yield vect
