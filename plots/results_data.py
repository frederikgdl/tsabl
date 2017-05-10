import os
from os import path, listdir
from typing import List

from plots import config


class ResultsData:

    def __init__(self, methods: List[str], embeddings: List[str], classifiers: List[str], num_epochs: int):
        """
        Creates a ResultsData instance that holds results for the given methods, embeddings and classifiers with 
        a certain number of epochs trained.
        
        :param methods:     List of methods to collect results from ("binary", "ternary", "agg_ternary", ...)
        :param embeddings:  List of embeddings/params to collect results from ("TextBlob", "AFINN", "windowsize=1", ...)
        :param classifiers: Names of classifiers to collect results from ("SVM c=1", "lexicon classifier", ...)
        :param num_epochs:  The minimum number of epochs a method+embedding combination must have been trained for.
        """
        # { method: { embedding: { classifier_name: [ { metric: value } ] } } }
        self.data = {}

        self.methods = methods
        self.embeddings = embeddings
        self.classifiers = classifiers
        self.num_epochs = num_epochs

        # Load results!
        for method in methods:

            embs = embeddings
            if embs == "all":
                method_path = path.join(config.EMBEDDINGS_DIR, method)
                embs = [d for d in os.listdir(method_path) if os.path.isdir(os.path.join(method_path, d))]

            for embedding in embs:
                selected_embeddings = path.join(method, embedding)
                results_dir = path.join(config.RESULT_DIR, selected_embeddings)

                if not self.filter_tweet(method, embedding, results_dir):
                    continue

                for classifier in classifiers:
                    self.load_results(method, embedding, classifier, results_dir)

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return str(self.data)

    def filter_tweet(self, method, embedding, results_dir):
        """Returns True if method and embedding combination is valid (basically done training for num_epochs)"""
        if not path.exists(results_dir):
            print("Skipping", method, embedding, "because its results_dir does not exist")
            return False

        if len([d for d in listdir(results_dir) if path.isdir(path.join(results_dir, d))]) < self.num_epochs:
            print("Skipping", method, embedding, "because its results do not contain enough epoch directories")
            return False

        return True

    def add_epoch_results(self, method, embedding, classifier, epoch_results):
        """Adds results for a given epoch to the data set"""
        if method not in self.data:
            self.data[method] = {}
        if embedding not in self.data[method]:
            self.data[method][embedding] = {}
        if classifier not in self.data[method][embedding]:
            self.data[method][embedding][classifier] = []
        self.data[method][embedding][classifier].append(epoch_results)

    def load_results(self, method, embedding, classifier, results_dir):
        """Loads results for all epochs of a method + embedding + classifier combination"""
        for epoch_dir in [d for d in listdir(results_dir) if path.isdir(path.join(results_dir, d))]:
            file_path = path.join(results_dir, epoch_dir, classifier.lower())
            with open(file_path) as f:
                epoch_results = {}
                for line in f:
                    metric = line.split()[0]
                    value = float(line.split()[1])
                    epoch_results[metric] = value
                self.add_epoch_results(method, embedding, classifier, epoch_results)
