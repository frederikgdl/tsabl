import logging
import os
from os import path, listdir

import classifiers.train_and_test as train_and_test
import plots.config as config
from plots.config import classifiers

logger = None
verbose = 0
quiet = False


def setup_logger():
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, verbose + 2)]  # capped to number of levels

    # create logger
    global logger
    logger = logging.getLogger('test_all_epochs')
    logger.setLevel(level)
    logger.propagate = False

    if quiet:
        logging.disable(logging.ERROR)
        return logger

    if len(logger.handlers) > 0:
        return logger

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def test_all_epochs(embeddings_dir, results_dir):
    # Sort by suffix number of files. Turn to int so that '7' is treated as less that '18', for instance.
    embeddings_files = sorted(listdir(embeddings_dir), key=lambda f: int(f.split("-")[-1]))

    for embeddings_file in embeddings_files:
        logger.info(embeddings_file)

        # Configure test_and_train
        train_and_test.classifiers = classifiers()
        train_and_test.baselines = []
        train_and_test.embedding_file = path.join(embeddings_dir, embeddings_file)
        train_and_test.verbose = -2
        train_and_test.quiet = True
        train_and_test.results_dir = path.join(results_dir, embeddings_file)

        # Run
        train_and_test.main()


def main():
    setup_logger()

    for method in config.METHODS:

        if not path.exists(path.join(config.EMBEDDINGS_DIR, method)):
            logger.warning(
                "Skipping " + method + " because " + path.join(config.EMBEDDINGS_DIR, method) + " does not exist")
            continue

        logger.info("Doing method " + method)
        embeddings = config.EMBEDDINGS
        if embeddings == "all":
            embeddings = os.listdir(path.join(config.EMBEDDINGS_DIR, method))

        for embedding in embeddings:
            selected_embeddings = path.join(method, embedding)
            embeddings_dir = path.join(config.EMBEDDINGS_DIR, selected_embeddings)
            results_dir = path.join(config.RESULT_DIR, selected_embeddings)

            if not path.exists(embeddings_dir):
                logger.warning("Skipping", method, embedding, "because its embeddings_dir does not exist")
                continue

            epoch_files = len(listdir(embeddings_dir))
            if epoch_files < config.NUM_EPOCHS:
                msg = "Skipping " + method + " " + embedding + "because it does not contain enough epoch files (" + str(
                    epoch_files) + "/" + str(config.NUM_EPOCHS) + ")"
                logger.warning(msg)
                continue

            logger.info("Doing " + method + " " + embedding)
            test_all_epochs(embeddings_dir, results_dir)


if __name__ == "__main__":
    main()
