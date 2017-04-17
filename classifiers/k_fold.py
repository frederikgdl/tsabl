from sklearn.model_selection import KFold

from classifiers.models.random_weighted import RandomWeighted


class KFoldValidator:

    def __init__(self, k, tweets, embeddings, labels, shuffle=True):

        if not len(tweets) == len(embeddings) == len(labels):
            print("Inconsistent length found in training data")
            raise ValueError

        self.kfold = KFold(n_splits=k, shuffle=shuffle)
        self.tweets = tweets
        self.embeddings = embeddings
        self.labels = labels
        self.k = k

        indexes = self.split()
        self.train_tweets, self.test_tweets = self.map_indexes_to_values(indexes, tweets)
        self.train_embeddings, self.test_embeddings = self.map_indexes_to_values(indexes, embeddings)
        self.train_labels, self.test_labels = self.map_indexes_to_values(indexes, labels)

    def split(self):
        return list(self.kfold.split(self.tweets))

    @staticmethod
    def map_indexes_to_values(indexes, data):
        train_partitions = []
        test_partitions = []
        for train_index, test_index in indexes:
            train_partitions.append(list(map(lambda i: data[i], train_index)))
            test_partitions.append(list(map(lambda i: data[i], test_index)))
        return train_partitions, test_partitions

    def get_training_data(self, partition_num):
        return self.train_tweets[partition_num], self.train_embeddings[partition_num], self.train_labels[partition_num]

    def get_test_data(self, partition_num):
        return self.test_tweets[partition_num], self.test_embeddings[partition_num], self.test_labels[partition_num]

    def run(self, classifier):
        """
        Train and test classifier on partitions and return average score
        :type classifier: classifiers.models.Model

        """

        results = []

        for i in range(self.k):
            train_tweets, train_embeddings, train_labels = self.get_training_data(i)
            test_tweets, test_embeddings, test_labels = self.get_test_data(i)
            classifier.train(train_tweets, train_embeddings, train_labels)
            classifier.predict(test_tweets, test_embeddings)
            classifier.test(test_labels)
            results.append(classifier.results.clone())

        print(sum(results) / self.k)

"""

kfold = KFoldValidator(10, list(range(40)))

kfold.run(RandomWeighted())
for x in range(kfold.k):
    print(kfold.train_partitions[x], kfold.test_partitions[x])
"""
