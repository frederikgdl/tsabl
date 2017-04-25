import unittest
from os import path

from classifiers.word_embedding_dict import WordEmbeddingDict


class TestWordEmbeddingDict(unittest.TestCase):

    def setUp(self):
        self.we_file_txt = path.join(path.dirname(path.realpath(__file__)), "assets/test_word_embeddings.txt")
        self.we_file_tsv = path.join(path.dirname(path.realpath(__file__)), "assets/test_word_embeddings.tsv")
        self.word_embeddings = [
            WordEmbeddingDict(self.we_file_txt),
            WordEmbeddingDict(self.we_file_tsv),
        ]

    def test_init(self):
        for word_embeddings_dict in self.word_embeddings:
            self.assertEqual(4, word_embeddings_dict.n_vecs)
            self.assertEqual(4, word_embeddings_dict.d)
            self.assertEqual(4, len(word_embeddings_dict.embeddings))
            for word in word_embeddings_dict.embeddings.keys():
                self.assertEqual(4, len(word_embeddings_dict.embeddings[word]))

if __name__ == '__main__':
    unittest.main()
