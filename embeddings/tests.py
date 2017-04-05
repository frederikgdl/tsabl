from utils import file_ops
from scipy import spatial

import embeddings.config as config


test_words = [
    'good',
    'bad',
    'happy',
    'sad',
    'paris',
    'london',
    'queen',
    'king',
    'man',
    'woman',
    'Kappa123'
]


def get_closest_words(word, embeddings):
    if word not in embeddings:
        return []

    scores = []
    for other_word, emb in embeddings.items():
        if word == other_word:
            continue

        similarity = 1 - spatial.distance.cosine(embeddings[word], emb)
        scores.append((other_word, similarity))

    scores.sort(key=lambda x: x[-1], reverse=True)
    return scores


def main():
    file_path = config.OUTPUT_FILE

    embeddings = file_ops.load_word_embeddings(file_path)

    for word in test_words:
        close_words = get_closest_words(word, embeddings)
        print('-'*40)
        print(word)
        for i, other_word in enumerate(close_words):
            if i > 10:
                break
            print("{}\t{}: {}".format(i+1, other_word[0], other_word[1]))

if __name__ == "__main__":
    main()
