from os import path, listdir

from scripts import config
from scripts import test_all_epochs


def main():
    for method in config.METHODS:
        for embedding in config.EMBEDDINGS:
            embeddings_dir = path.join(config.EMBEDDINGS_DIR, method, embedding)

            if not path.exists(embeddings_dir):
                print("Skipping", method, embedding, "because its embeddings_dir does not exist")
                continue

            if len(listdir(embeddings_dir)) < config.NUM_EPOCHS:
                print("Skipping", method, embedding, "because it does not contain enough epoch files")
                continue

            print("Doing", method, embedding)

            test_all_epochs.embeddings_dir = embeddings_dir
            test_all_epochs.main()


if __name__ == "__main__":
    main()
