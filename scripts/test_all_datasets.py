from importlib import reload
from os import path, listdir

from scripts import config
from scripts import test_all_epochs


def main():
    for method in config.METHODS:
        for embedding in config.EMBEDDINGS:
            selected_embeddings = path.join(method, embedding)
            embeddings_dir = path.join(config.EMBEDDINGS_DIR, selected_embeddings)
            results_dir = path.join(config.RESULT_DIR, selected_embeddings)

            if not path.exists(embeddings_dir):
                print("Skipping", method, embedding, "because its embeddings_dir does not exist")
                continue

            if len(listdir(embeddings_dir)) < config.NUM_EPOCHS:
                print("Skipping", method, embedding, "because it does not contain enough epoch files")
                continue

            if path.exists(results_dir) and any([f for f in listdir(results_dir) if f.endswith(".png")]):
                print("Skipping", method, embedding,
                      "because a png was found in " + results_dir + ", which means it has been tested already.")
                continue

            print("Doing", method, embedding)

            # Reload reimports test_all_epochs, so its state is reset.
            reload(test_all_epochs)
            test_all_epochs.selected_embeddings = selected_embeddings
            test_all_epochs.embeddings_dir = embeddings_dir
            test_all_epochs.results_dir = results_dir
            test_all_epochs.main()


if __name__ == "__main__":
    main()
